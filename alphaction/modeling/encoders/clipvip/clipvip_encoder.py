import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.configuration_clip import CLIPConfig
from transformers import CLIPTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPooling
from .CLIP_ViP import CLIPModel, _expand_mask
from alphaction.modeling.backbone.vit_utils import interpolate_pos_embed_online
from alphaction.cam.hilacam import construct_attention
from alphaction.cam.mhsa import get_multi_head_mask
from contextlib import nullcontext
from typing import Any, Optional, Tuple, Union
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
from einops import rearrange
from collections import OrderedDict
import re


def load_clipmodel(cfg):
    clipconfig = CLIPConfig.from_pretrained(cfg.CLIP_NAME)  # "openai/clip-vit-base-patch16"
    args = {"temporal_size": cfg.TEMPORAL_SIZE,
            "if_use_temporal_embed": "1" if cfg.USE_TEMPORAL_EMBED else "0",
            "logit_scale_init_value": cfg.LOGIT_SCALE_INIT,
            "add_cls_num": cfg.ADD_CLS_NUM,
            "st_cross_attn": cfg.ST_CROSS_ATTN,
            "num_xattn": cfg.NUM_XATTN}
    setattr(clipconfig, "vision_additional_config", Dict2Class(args))
    setattr(clipconfig.vision_config, "temporal_winsize", cfg.TEMPORAL_WINSIZE)

    if getattr(cfg, 'USE_ATTN', False):
        setattr(clipconfig.vision_config, 'attn_prob', True)
    
    if getattr(cfg, 'USE_GRAD', False):
        setattr(clipconfig.vision_config, 'attn_grad', True)
    
    if getattr(cfg, 'USE_ATTN_LAST', False):
        setattr(clipconfig.vision_config, 'attn_last_only', True)
    
    if cfg.CLIP_NAME:
        model = CLIPModel.from_pretrained(cfg.CLIP_NAME, config=clipconfig)
    else:
        model = CLIPModel(clipconfig)
    
    # init logit scale from 
    logit_scale_value = cfg.LOGIT_SCALE_INIT
    model.logit_scale.data.fill_(logit_scale_value)

    return model


class Dict2Class(object):
    def __init__(self, my_dict):
        for k in my_dict:
            setattr(self, k, my_dict[k])


class CLIPViPVisualEncoder(nn.Module):
    def __init__(self, cfg, clipmodel):
        super(CLIPViPVisualEncoder, self).__init__()
        self.use_cls_feat = cfg.USE_CLS_FEAT
        self.requires_backprop = False

        self.output_attentions = clipmodel.config.output_attentions
        self.output_hidden_states = clipmodel.config.output_hidden_states
        self.use_return_dict = clipmodel.config.use_return_dict

        self.vision_model = clipmodel.vision_model
        if self.use_cls_feat:
            self.visual_projection = clipmodel.visual_projection
    

    def get_video_features(self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        if_norm: Optional[bool] = None,
        interpolator=None,
    ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        vision_outputs, ws = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolator=interpolator,
            return_ws=True,
        )
        last_hidden_states = vision_outputs['last_hidden_state']  # (1, 5636, D=768)
        
        st_feats = last_hidden_states[:, 4:, :]
        out_dict = {'st_feats': st_feats, 'ws': ws}

        if self.use_cls_feat:
            # get the features of video proxy tokens
            pooled_output = vision_outputs[1]  # pooled_output
            cls_feats = self.visual_projection(pooled_output)
            # normalize
            if_norm = if_norm if if_norm is not None else False
            if if_norm:
                cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
            out_dict.update({'cls_feats': cls_feats})
        
        return out_dict


    def project_patch_features(self, patch_features):
        """ patch_features: (B, D, T, h, w)
        """
        x = patch_features.permute(0, 2, 3, 4, 1).contiguous()
        x = self.visual_projection(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, video):
        inputs = {"pixel_values": video, 
                  "interpolator": interpolate_pos_embed_online}
        with torch.no_grad() if not self.requires_backprop else nullcontext():
            video_features = self.get_video_features(**inputs)
        return video_features


class CLIPViPTextEncoder(nn.Module):
    def __init__(self, cfg, clipmodel):
        super(CLIPViPTextEncoder, self).__init__()
        self.context_prompt = cfg.CONTEXT_INIT
        self.context_len = cfg.LEN_CONTEXT
        self.device = torch.device('cuda')
        self.use_soft_prompt = cfg.SOFT_PROMPT
        self.requires_backprop = False

        self.output_attentions = clipmodel.config.output_attentions
        self.output_hidden_states = clipmodel.config.output_hidden_states
        self.use_return_dict = clipmodel.config.use_return_dict

        # modules
        self.tokenizer = CLIPTokenizerFast.from_pretrained(cfg.CLIP_NAME)
        self.embedder = clipmodel.text_model.embeddings.to(self.device)  # token_embedding(), position_embedding(), position_ids
        self.encoder = clipmodel.text_model.encoder.to(self.device)  # Transformer encoder
        self.layer_norm = clipmodel.text_model.final_layer_norm.to(self.device)  # nn.LayerNorm
        self._build_causal_attention_mask = clipmodel.text_model._build_causal_attention_mask  # function
        self.text_projection = clipmodel.text_projection.to(self.device)  # nn.Linear

        # get the initialized position embeddings given the context length
        self.position_embeddings = self.get_pos_embed()  # (1, L, D)

        if cfg.CONTEXT_INIT:
            # initialize learnable context
            prompt_init = self.init_soft_prompt(cfg.CONTEXT_INIT)
            if self.use_soft_prompt:  # make sure model.named_parameters() and model.state_dict() can find it
                self.register_parameter("soft_prompt", nn.Parameter(prompt_init))
            else:
                self.soft_prompt = prompt_init
        else:
            self.soft_prompt = None

        self.vocab_token_embeddings = {}  # dynamically updated in self.set_vocabulary()
    

    def get_pos_embed(self):
        # get the positional embedding
        with torch.no_grad() if not self.requires_backprop else nullcontext():
            position_ids = self.embedder.position_ids[:, :self.context_len]
            position_embeddings = self.embedder.position_embedding(position_ids)
        return position_embeddings


    def init_soft_prompt(self, context_init=None):
        """ context_init: "a video of "
        """
        # get the initial context embeddings
        batch_enc = self.tokenizer.batch_encode_plus(
            [context_init.strip()],  # 'a video of'
            max_length=self.context_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        token_ctx = batch_enc.input_ids.to(self.device)  # (1, L)
        n_ctx = token_ctx[0].argmax() - 1  # the number of context tokens (=3)

        # the embeding of "[SOS] a video of [EOS]..."
        with torch.no_grad() if not self.requires_backprop else nullcontext():
            embedding = self.embedder.token_embedding(token_ctx)
        embed_ctx = embedding[0, 1 : 1 + n_ctx, :]  # (n_ctx, D)
        return embed_ctx
    

    def get_token_embeddings(self, batch_text):
        batch_enc = self.tokenizer.batch_encode_plus(
            batch_text,  # e.g., "a video of person golfing"
            max_length=self.context_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )  # tokens of "[SOS] a video of person golfing [EOS]..."
        token_ids = batch_enc.input_ids.to(self.device)  # (K_new, L)
        token_masks = batch_enc.attention_mask.to(self.device)  # (K_new, L)
        eos_idx = token_ids.argmax(dim=-1)

        # embed the input token sequence with masks
        with torch.no_grad() if not self.requires_backprop else nullcontext():
            token_embeds = self.embedder.token_embedding(token_ids)  # (K_new, L, D)  # No position embedding
        
        return token_embeds, token_masks, eos_idx


    def set_vocabulary(self, text_data, embeddings=None):
        # update vocabulary list
        self.text_data = text_data
        if embeddings is not None:
            self.vocab_token_embeddings = embeddings
        else:
            new_classes = list(set(self.text_data.keys()).difference(set(self.vocab_token_embeddings.keys())))
            if len(new_classes) > 0:  # when eval_open=True
                if isinstance(list(self.text_data.values())[0]['caption'], str):  # single caption
                    text_captions = [(self.context_prompt.strip() + ' ' + self.text_data[vocab]['caption']).strip()
                                    for vocab in new_classes]
                    token_embeds, token_masks, eos_idx = self.get_token_embeddings(text_captions)
                    # store info
                    self.vocab_token_embeddings.update({vocab: {'embed': token_embeds[k],
                                                                'mask': token_masks[k],
                                                                'eos_id': eos_idx[k]} 
                                                        for k, vocab in enumerate(new_classes)})
                else:
                    for vocab in new_classes:
                        # e.g., 'Climb stairs: He ascended the staircase with haste, not wanting to waste any time.'
                        # text_captions = [re.sub(r'\d+.', '{}:'.format(re.sub(r'_', ' ', vocab.capitalize())), cap) 
                        #     for cap in self.text_data[vocab]['caption']]
                        # text_captions = [re.sub(r'\d+. ', '', cap) for cap in self.text_data[vocab]['caption']]
                        text_captions = self.text_data[vocab]['caption']
                        token_embeds, token_masks, eos_idx = self.get_token_embeddings(text_captions)
                        # store info
                        self.vocab_token_embeddings.update({vocab: {'embed': token_embeds,
                                                                    'mask': token_masks,
                                                                    'eos_id': eos_idx}})


    def construct_token_embeds(self, use_soft_prompt=False, cond=None):
        """ Construct the input embeddings of "[SOS] a video of person golfing [EOS] ..."      
        """ 
        class_token_embed = [self.vocab_token_embeddings[vocab]['embed'] for vocab in self.text_data.keys()]
        class_token_embed = torch.stack(class_token_embed, dim=0)  # (K, L, D)
        num_cls, num_tokens, num_feat = class_token_embed.size()

        token_masks = torch.stack([self.vocab_token_embeddings[vocab]['mask'] for vocab in self.text_data.keys()], dim=0)  # (K, L)
        eos_ids = torch.stack([self.vocab_token_embeddings[vocab]['eos_id'] for vocab in self.text_data.keys()], dim=0)  # (K,)
        
        if use_soft_prompt:
            # insert the learnable soft prompt (or hard prompt embedding)
            class_token_embed[:, 1: (1+self.soft_prompt.size(0)), :] = self.soft_prompt
        
        # add the positional embedding
        class_token_embed = class_token_embed + self.position_embeddings
        
        return class_token_embed, token_masks, eos_ids
    

    def tokenize_input(self, text_input, device=torch.device('cuda')):
        batch_enc = self.tokenizer.batch_encode_plus(
            text_input,
            max_length=self.context_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids.to(device)  # (B, L)
        text_input_mask = batch_enc.attention_mask.to(device)  # (B, L)
        return text_input_ids, text_input_mask


    def embed_tokens(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad() if not self.requires_backprop else nullcontext():
            # use the embedder directly
            hidden_states = self.embedder(input_ids=input_ids, position_ids=position_ids)  # position embedding added

        return hidden_states


    def build_attention_masks(self, input_shape, attention_mask, type=torch.float16):
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        bsz, seq_len = input_shape
        if_fp16 = type == torch.float16
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, fp16=if_fp16).to(self.device)
        
        # expand attention_mask
        attention_mask = _expand_mask(attention_mask, type)  # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        
        return attention_mask, causal_attention_mask
    

    def forward_with_embeds(self, 
        hidden_states,
        attn_mask,
        eos_ids,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        # causal attention mask
        attention_mask, causal_attention_mask = self.build_attention_masks(hidden_states.size()[:2], attn_mask, type=hidden_states.dtype)
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # encoder
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), eos_ids]

        if not self.use_return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    
    def get_text_features(self, encoder_outs, if_norm: Optional[bool] = None):
        """ post process after transformer encoder (linear projection, normalization)
        """
        pooled_output = encoder_outs[1]
        text_features = self.text_projection(pooled_output)
        
        if_norm = if_norm if if_norm is not None else False
        if if_norm:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def get_text_token_feats(self, encoder_outs, if_norm: Optional[bool] = None):
        """ get the text token features after transformer encoder (linear projection, normalization)
        """
        hidden_states = encoder_outs[0]  # (K, L=24, D)
        token_features = self.text_projection(hidden_states)

        if_norm = if_norm if if_norm is not None else False
        if if_norm:
            token_features = token_features / token_features.norm(dim=-1, keepdim=True)

        return token_features
    

    def encode_person(self, prompt_ensemble=False):
        if prompt_ensemble:  # 85 templates from CLIP_Surgery
            prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
        
        input_text = [prompt.format("person") for prompt in prompt_templates] if prompt_ensemble else ["a photo of person"]
        input_ids, input_mask = self.tokenize_input(input_text)
        with torch.no_grad():
            # token embedding
            hidden_states = self.embed_tokens(input_ids, input_mask)
            # get text feature
            encoder_outs = self.forward_with_embeds(hidden_states, input_mask, input_ids.argmax(dim=-1))
            text_features = self.get_text_features(encoder_outs, if_norm=False)  # (85, D)
        
        if prompt_ensemble:
            # aggregation
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_feat_out = text_features.mean(dim=0)
            text_feat_out /= text_feat_out.norm()
        else:
            text_feat_out = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
        
        return text_feat_out



class CLIPViPModel(nn.Module):
    def __init__(self, cfg, dtype=torch.float32):
        super(CLIPViPModel, self).__init__()
        self.dtype = dtype

        # load the CLIPViP model
        clipmodel = load_clipmodel(cfg)
        self.finetune_added_cls = cfg.FINETUNE_ADDED_CLS

        self.num_pathways = 1
        self.tau_inv = clipmodel.logit_scale.exp()
        self.dim_embed = clipmodel.vision_embed_dim  # pre-projection dim
        self.use_soft_prompt = cfg.SOFT_PROMPT

        # clip model config choices
        self.output_attentions = clipmodel.config.output_attentions
        self.output_hidden_states = clipmodel.config.output_hidden_states
        self.use_return_dict = clipmodel.config.use_return_dict

        self.visual_encoder = CLIPViPVisualEncoder(cfg, clipmodel)
        self.text_encoder = CLIPViPTextEncoder(cfg, clipmodel)

        self.cam_method = cfg.CAM_METHOD
        self.use_attn = getattr(cfg, 'USE_ATTN', False)
        self.use_grad = getattr(cfg, 'USE_GRAD', False)
        if self.cam_method == 'HilaCAM': assert self.use_attn

        self.use_person_embed = cfg.USE_PERSON_EMBED
        if self.use_person_embed:
            self.person_embed = self.text_encoder.encode_person(prompt_ensemble=True)
        self.cam_sampling = cfg.CAM_SAMPLING
        
        self.max_prematch = getattr(cfg, 'MAX_PRE_MATCH', False)
        

    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)


    def freeze_modules(self, mismatched_keys):
        trainable_params =  copy.deepcopy(mismatched_keys)
        if self.finetune_added_cls:
            trainable_params.append('visual_encoder.vision_model.embeddings.added_cls')
        
        # freeze the vision model
        for n, p in self.visual_encoder.named_parameters():
            if 'visual_encoder.' + n not in trainable_params:
                p.requires_grad=False
            else:
                # has some learnable parameters that need backprop
                self.visual_encoder.requires_backprop = True

        # freeze the text model
        for n, p in self.text_encoder.named_parameters():
            if 'text_encoder.' + n not in trainable_params:
                p.requires_grad=False
            else:
                # has some learnable parameters that need backprop
                self.text_encoder.requires_backprop = True
        # freeze others
        self.tau_inv.requires_grad = False
        
        if self.use_grad:
            self.visual_encoder.requires_backprop = True
    

    def forward(self, x_list):
        x = x_list[0]  # slow video (b, C=3, T=16, H=256, W=?)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, T = x.size()[:2]

        # encoder forward
        x_dict = self.visual_encoder(x)

        # parse output
        x, ws = x_dict['st_feats'], x_dict['ws']
        x = x.reshape(B, T, ws[0], ws[1], -1).permute(0, 4, 1, 2, 3).contiguous()  # (B, D, T, h, w)

        if self.visual_encoder.use_cls_feat:
            cls_feat = x_dict['cls_feats']
            return [x, x, x, x], cls_feat
        
        return [x, x, x, x]


    def forward_text(self, device=torch.device('cuda'), cond=None):
        single_cap = isinstance(list(self.text_encoder.text_data.values())[0]['caption'], str)  # single caption
        if not single_cap:
            input_texts = {vocab: data['caption'] for vocab, data in self.text_encoder.text_data.items()}
            return self.forward_multi_cap(input_texts, device)  # return a list of (M, D), __len__=K

        if self.use_soft_prompt:
            hidden_states, text_input_mask, text_eos_id = self.text_encoder.construct_token_embeds(use_soft_prompt=self.use_soft_prompt, cond=cond)  # (K, L=8, D)
        else:
            text_input = [(self.text_encoder.context_prompt.strip() + " " + data['caption']).strip() for vocab, data in self.text_encoder.text_data.items()]
            text_input_ids, text_input_mask = self.text_encoder.tokenize_input(text_input, device=device)  # (K, L=24)
            hidden_states = self.text_encoder.embed_tokens(text_input_ids, text_input_mask)
            text_eos_id = text_input_ids.argmax(dim=-1)
        
        with torch.no_grad() if not self.text_encoder.requires_backprop else nullcontext():
            # get text feature
            encoder_outs = self.text_encoder.forward_with_embeds(hidden_states, text_input_mask, text_eos_id)
            text_features = self.text_encoder.get_text_features(encoder_outs, if_norm=False)
        
        if self.training and 'hardneg' in list(self.text_encoder.text_data.values())[0]:
            input_texts = {vocab: data['hardneg'] for vocab, data in self.text_encoder.text_data.items()}
            hardneg = self.forward_multi_cap(input_texts, device) 
            text_features = torch.cat((text_features.unsqueeze(1), torch.stack(hardneg, dim=0)), dim=1)  # (K, M, D)

        return text_features
    

    def forward_multi_cap(self, input_texts, device=torch.device('cuda')):
        text_features = []
        for vocab, text_captions in input_texts.items():
            # text_captions = [re.sub(r'\d+. ', '', cap) for cap in data['caption']]
            # text_captions = data['caption']
            input_ids, input_mask = self.text_encoder.tokenize_input(text_captions)
            # token embedding
            hidden_states = self.text_encoder.embed_tokens(input_ids, input_mask)
            with torch.no_grad() if not self.text_encoder.requires_backprop else nullcontext():
                # get text feature
                encoder_outs = self.text_encoder.forward_with_embeds(hidden_states, input_mask, input_ids.argmax(dim=-1))
                cap_features = self.text_encoder.get_text_features(encoder_outs, if_norm=False)
            text_features.append(cap_features)
        return text_features
    

    def get_ritsm_coarse(self, cls_token_feat, patch_token_feat, text_features):
        """ cls_token_feat: (B, D)
            patch_token_feat: (B, D, T, h, w)
            text_features: (K, D)
        """
        if self.use_person_embed:
            batch_size = patch_token_feat.size(0)
            cls_feat = self.person_embed.unsqueeze(0).expand(batch_size, -1)  # (B, D)
        else:
            # normalize
            cls_token_feat = cls_token_feat / cls_token_feat.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # pre-matching
            pred_cls = (cls_token_feat @ text_features.t()).argmax(dim=-1)  # (B,)
            cls_feat = text_features[pred_cls] # (B, D)
        
        T, h, w = patch_token_feat.size()[-3:]
        # layernorm, projection, and normalization
        patch_features = rearrange(patch_token_feat, 'b d t h w -> b (t h w) d')  # (B, Thw, D)
        patch_features = self.visual_encoder.vision_model.post_layernorm(patch_features)  # layernorm over D
        patch_features = self.visual_encoder.visual_projection(patch_features)  # 768 --> 512
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)  # (B, Thw, 512)

        # patch-text similarity
        pt_sim = 1 - torch.bmm(patch_features, cls_feat.unsqueeze(-1))  # (B, Thw, 1)
        attn_maps = rearrange(pt_sim[:, :, 0], 'b (t h w) -> b t h w', t=T, h=h, w=w)
        # we use the CAM of middle frame for sampling
        attn_maps = attn_maps[:, int(T//2)]  # (B, H, W)

        # 3: normalization
        attn_maps = rearrange(attn_maps, 'b h w -> b (h w)')  # (B, H*W)
        attn_maps -= attn_maps.min(dim=-1, keepdim=True)[0]
        attn_maps /= attn_maps.max(dim=-1, keepdim=True)[0]
        attn_maps = rearrange(attn_maps, 'b (h w) -> b h w', h=h, w=w)

        return attn_maps



    def get_ritsm(self, cls_token_feat, patch_token_feat, text_features, input_sizes):
        """ cls_token_feat: (B, D)
            patch_token_feat: (B, D, T, h, w)
            text_features: (K, D)
            input_sizes: (B, 2), (height, width)
        """
        if self.use_person_embed:
            batch_size = patch_token_feat.size(0)
            cls_feat = self.person_embed.unsqueeze(0).expand(batch_size, -1)  # (B, D)
        elif self.max_prematch:
            dim_feat = cls_token_feat.size(-1)
            text_features = torch.stack(text_features, dim=0).view(-1, dim_feat)  # (K*M, D)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            pred_idx = (cls_token_feat @ text_features.t()).argmax(dim=-1)  # (B,)
            cls_feat = text_features[pred_idx] # (B, D)
        else:
            if isinstance(text_features, list):
                # text_features = torch.stack([feat.mean(dim=0) for feat in text_features], dim=0)
                text_features = torch.stack([feat[0] for feat in text_features], dim=0)
            pred_cls = (cls_token_feat @ text_features.t()).argmax(dim=-1)  # (B,)
            cls_feat = text_features[pred_cls] # (B, D)
        T, h, w = patch_token_feat.size()[-3:]

        # layernorm, projection, and normalization
        patch_features = rearrange(patch_token_feat, 'b d t h w -> b (t h w) d')  # (B, Thw, D)
        patch_features = self.visual_encoder.vision_model.post_layernorm(patch_features)  # layernorm over D
        patch_features = self.visual_encoder.visual_projection(patch_features)  # 768 --> 512
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)  # (B, Thw, 512)

        # image-text similarity
        it_sim = torch.bmm(patch_features, cls_feat.unsqueeze(-1))  # (B, Thw, 1)

        # 1: reshape
        featmap_attention = rearrange(it_sim[:, :, 0], 'b (t h w) -> b t h w', t=T, h=h, w=w)

        # since the target size could be different in a batch, we have to for-loop each sample here:
        batch_attention_maps = []
        for feat_attn, target_size in zip(featmap_attention, input_sizes):
            # 2: resize
            H, W = int(target_size[0]), int(target_size[1])
            attn_map = F.interpolate(feat_attn.unsqueeze(0).float(), size=(H, W), mode='bilinear')  # (1, T, H, W)
            # 3: normalization
            attn_map = rearrange(attn_map[0], 't h w -> t (h w)')  # (T, H*W)
            attn_map -= attn_map.min(dim=-1, keepdim=True)[0]
            attn_map /= attn_map.max(dim=-1, keepdim=True)[0]
            attn_map = rearrange(attn_map, 't (h w) -> t h w', h=H, w=W)
            # 4: reverse attention
            attn_map = 1 - attn_map
            # we use the CAM of middle frame for sampling
            batch_attention_maps.append(attn_map[int(T//2)])

        return batch_attention_maps
    
    
    def get_hilacam(self, cls_token_feat, patch_token_feat, text_features, input_sizes):
        """ cls_token_feat: (B, D)
            patch_token_feat: (B, D, T, h, w)
            text_features: (K, D)
            input_sizes: (B, 2), (height, width)
        """
        if isinstance(text_features, list):
            text_features = torch.stack(text_features, dim=0).mean(dim=1)
        batch_size, num_cls = cls_token_feat.size(0), text_features.size(0)
        patch_h, patch_w = patch_token_feat.size()[-2:]
        device = cls_token_feat.device
        # backward
        if self.use_grad:
            logits = cls_token_feat @ text_features.t()  # (B, K)
            index = logits.argmax(dim=-1)  # (B,)
            # create pseudo label
            one_hot = torch.zeros((batch_size, num_cls), dtype=torch.float32).to(device)
            one_hot[torch.arange(batch_size), index] = 1
            one_hot = torch.sum(one_hot * logits)
            self.zero_grad()
            # back propergate to the network
            one_hot.requires_grad_(True)
            one_hot.backward(retain_graph=True)
            
        # create a diagonal matrix
        image_attn_blocks = list(dict(self.visual_encoder.vision_model.encoder.layers.named_children()).values())
        num_proxy, num_tokens = image_attn_blocks[0].attn_probs['inter'].shape[-2:]  # number of video proxy tokens and all tokens
        num_patches = image_attn_blocks[0].attn_probs['intra'].size(1)  # number of patches within each frame
        num_frames = int((num_tokens - num_proxy) / num_patches)
        num_heads = self.visual_encoder.vision_model.config.num_attention_heads  # =12
        R = torch.stack([torch.eye(num_tokens, num_tokens, dtype=torch.float32) 
                         for _ in range(batch_size)], dim=0).to(device) # (B, M+N*L, M+N*L)
        # weighted activation
        for blk in image_attn_blocks:
            cam = construct_attention(blk.attn_probs['inter'], blk.attn_probs['intra'], num_frames)  # (B*num_heads, M+N*L, M+N*L)
            cam = cam.reshape(-1, num_heads, cam.shape[-1], cam.shape[-1])
            if self.use_grad:
                grad = construct_attention(blk.attn_grads['inter'], blk.attn_grads['intra'], num_frames)  # (B*num_heads, M+N*L, M+N*L)
                grad = grad.reshape(-1, num_heads, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=1)  # (B, M+N*L, M+N*L), average cam over 12 heads
            R += torch.bmm(cam.detach(), R)  # accumulative inner-product over layers
            # after usage of the attn_probs, delete them from model since they are too large
            blk.attn_probs['inter'] = None
            blk.attn_probs['intra'] = None

        image_relevance_all = R[:, 0, num_proxy:].view(batch_size, num_frames, num_patches)  # the 1st row (B, T, L)
        
        # since the target size could be different in a batch, we have to for-loop each sample here:
        batch_attention_maps = []
        for image_relevance, target_size in zip(image_relevance_all, input_sizes):
            # 2: resize
            H, W = int(target_size[0]), int(target_size[1])
            image_relevance = image_relevance.reshape(1, num_frames, patch_h, patch_w)
            image_relevance = F.interpolate(image_relevance.float(), size=(H, W), mode='bilinear')
            # 3: normalization
            attn_map = rearrange(image_relevance[0], 't h w -> t (h w)')  # (T, H*W)
            attn_map -= attn_map.min(dim=-1, keepdim=True)[0]
            attn_map /= attn_map.max(dim=-1, keepdim=True)[0]
            attn_map = rearrange(attn_map, 't (h w) -> t h w', h=H, w=W)
            # we use the CAM of middle frame for sampling
            batch_attention_maps.append(attn_map[int(num_frames // 2)])
            
        return batch_attention_maps
    

    def get_mhsacam(self, patch_token_feat, input_sizes, attn_type='intra', use_mask=False, threshold=0.6):
        """ patch_token_feat: (B, D, T, h, w)
            input_sizes: (B, 2), (height, width)
            threshold: the threshold to clean the attention map
        """
        num_proxy = self.visual_encoder.vision_model.embeddings.add_cls_num + 1  # M
        num_heads = self.visual_encoder.vision_model.config.num_attention_heads  # num_heads
        num_frames = patch_token_feat.size(2)  # N
        heatmap_size = patch_token_feat.size()[-2:]
        num_patches = heatmap_size[0] * heatmap_size[1]  # L
        
        # get the last MHSA attention layer
        last_block = list(dict(self.visual_encoder.vision_model.encoder.layers.named_children()).values())[-1]
        
        if attn_type == 'intra':
            # get the self-attention within each frame
            attn_intra = last_block.attn_probs['intra']  # [B*num_heads*N, L, M+L] where L=196 if input_size=224
            attentions = attn_intra[:, 0, num_proxy:].reshape(-1, num_heads, num_frames, num_patches)  # [B*num_heads*N, L] --> [B, num_heads, N, L]
        
        elif attn_type == 'inter':
            # get the self-attention across frames
            attn_inter = last_block.attn_probs['inter']  # [B*num_heads, M, M+N*L] where M=4
            attentions = attn_inter[:, 0, num_proxy:].reshape(-1, num_heads, num_frames, num_patches)  # [B*num_heads, N*L] --> [B, num_heads, N, L]
        
        # post process
        batch_attention_maps = []
        for attn, target_size in zip(attentions, input_sizes):
            H, W = int(target_size[0]), int(target_size[1])
            if use_mask:
                # 1. masking: apply mask on attention map
                th_attn = get_multi_head_mask(attn, threshold)
                attn = attn * th_attn.float()  # (num_heads, N, L)
            # 2. aggregating: average over multi-heads as the final attention
            attn_map = attn.reshape(num_heads, num_frames, heatmap_size[0], heatmap_size[1]).mean(dim=0, keepdim=True)
            # 3. resizing
            attn_map = F.interpolate(attn_map, size=(H, W), mode="bilinear")[0]
            # 4. normalizing
            attn_map -= attn_map.min(dim=-1, keepdim=True)[0]
            attn_map /= attn_map.max(dim=-1, keepdim=True)[0]
            # we use the CAM of middle frame for sampling
            batch_attention_maps.append(attn_map[int(num_frames // 2)])
        
        return batch_attention_maps
        
    
    def get_cam(self, cls_token_feat, patch_token_feat, text_features, input_sizes):
        """ cls_token_feat: (B, D)
            patch_token_feat: (B, D, T, h, w)
            text_features: (K, D)
            input_sizes: (B, 2), (height, width)
        """
        need_backprop = self.text_encoder.requires_backprop or self.visual_encoder.requires_backprop
        with torch.no_grad() if not need_backprop else nullcontext():
            if self.cam_method == 'RITSM':
                if self.cam_sampling == 'topk':
                    return self.get_ritsm(cls_token_feat, patch_token_feat, text_features, input_sizes)
            elif self.cam_method == 'HilaCAM':
                return self.get_hilacam(cls_token_feat, patch_token_feat, text_features, input_sizes)
            elif self.cam_method == 'MHSA':
                return self.get_mhsacam(patch_token_feat, input_sizes)
            else:
                raise NotImplementedError


def key_transform(key):
    if 'text_model.embeddings' in key:
        return 'text_encoder.embedder' + key.split('text_model.embeddings')[-1]  # 3 keys
    if 'text_model.encoder' in key:
        return 'text_encoder.encoder' + key.split('text_model.encoder')[-1]  # 192 keys
    if 'text_model.final_layer_norm' in key:
        return 'text_encoder.layer_norm' + key.split('text_model.final_layer_norm')[-1]  # 2 keys
    if 'text_projection' in key:
        return 'text_encoder.text_projection' + key.split('text_projection')[-1]  # 1 key
    if ('vision_model' in key) or ('visual_projection' in key):
        return 'visual_encoder' + key.split('clipmodel')[-1]  # 203 keys


def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
    """operated in-place, no need to return `model`"""

    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(
            loaded_state_dict_or_path, map_location="cpu")
    else:
        loaded_state_dict = loaded_state_dict_or_path
    
    # key transformation
    load_transformed = dict()
    for k, v in loaded_state_dict.items():
        load_transformed[key_transform(k)] = v  # logit_scale missed here

    model_keys = set([k for k in list(model.state_dict().keys())])  # state_dict() ignores the tau_inv
    load_keys = set(load_transformed.keys())

    toload = {}
    mismatched_keys = []
    for k in model_keys:
        if (k in load_keys) and (model.state_dict()[k].shape == load_transformed[k].shape):
            toload[k] = load_transformed[k]  # has weight to load
        else:
            mismatched_keys.append(k)  # weights that need to train
    
    model.tau_inv = loaded_state_dict['clipmodel.logit_scale'].exp()  # used the pretrained logit_scale
    model.load_state_dict(toload, strict=False)

    return mismatched_keys


def load_pretrained_clipvip(model, cfg):
    if cfg.WEIGHT:
        mismatched_keys = load_state_dict_with_mismatch(model, cfg.WEIGHT)
    
    if getattr(cfg, "overload_logit_scale", False):  # by default, we do not overload logit scale
        model.overload_logit_scale(cfg.overload_logit_scale)
    
    # freeze components
    model.freeze_modules(mismatched_keys)


def build_clipvip_backbone(cfg):
    cfg.MODEL.CLIPViP.update(USE_CLS_FEAT=cfg.MODEL.STM.USE_CLS_FEAT)

    model = CLIPViPModel(cfg.MODEL.CLIPViP)

    # load the pre-trained CLIPViP model
    load_pretrained_clipvip(model, cfg.MODEL.CLIPViP)

    return model