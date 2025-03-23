import torch
import torch.nn as nn
from .clip_loader import load
import clip
from alphaction.modeling.backbone.vit_utils import interpolate_pos_embed_online
import torch.utils.checkpoint as checkpoint
from contextlib import nullcontext
from einops import rearrange
import torch.nn.functional as F



class CLIPVisualEncoder(nn.Module):
    def __init__(self, cfg, clip_model, dtype=torch.float16):
        super(CLIPVisualEncoder, self).__init__()
        self.dtype = dtype
        self.device = torch.device('cuda')

        # clip model components
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

        pretrain_img_size = clip_model.visual.input_resolution  # 224
        patch_size = clip_model.visual.conv1.kernel_size[0]  # 14 for ViT/L14, 16 for ViT/B16
        self.grid_size = [pretrain_img_size // patch_size, pretrain_img_size // patch_size]  # [16,16], or [14,14]
        self.use_checkpoint = cfg.MODEL.CLIP.USE_CHECKPOINT
        self.use_cls_feat = cfg.MODEL.STM.USE_CLS_FEAT
        self.requires_backprop = False
    
    
    def project_patch_features(self, patch_features):
        """ patch_features: (B, D, T, h, w)
        """
        x = patch_features.permute(0, 2, 3, 4, 1).contiguous()
        x = x @ self.proj.type(x.dtype)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    

    def forward(self, x_list):
        dtype = x_list[0].dtype
        x = x_list[0].type(self.dtype)  # slow video (b, 3, 16, 256, ?)
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)  # (b*16, 3, 256, ?)
        x = self.conv1(x)  # shape = [b*16, width=1024, 18, ?]
        ws = x.size()[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [b*16, width, 18*]
        x = x.permute(0, 2, 1).contiguous()  # shape = [N=b*16, L=18*, D=width]
        x = torch.cat([self.class_embedding.to(x.dtype) + 
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [N, L+1, D]
        # interpolate positional embedding online
        pos_embed = self.positional_embedding.unsqueeze(0)  # (1, 257, D)
        if pos_embed.shape[1] != ws[0] * ws[1] + 1:
            pos_embed = interpolate_pos_embed_online(pos_embed, self.grid_size, ws, 1)  # (1, 18*?+1, D)
        x = x + pos_embed.to(x.dtype)
        x = self.ln_pre(x) # (16*B, 18*?+1, D)

        x = x.permute(1, 0, 2).contiguous()  # NLD -> LND
        # x = self.transformer(x)
        for blk in self.transformer.resblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        xseq = x.permute(1, 0, 2).contiguous()  # LND -> NLD=(B*16, 18*?+1, D)

        x = xseq[:, 1:, :].reshape(B, T, ws[0], ws[1], -1).permute(0, 4, 1, 2, 3).contiguous().type(dtype)  # (B, D, T, h, w)

        if self.use_cls_feat:
            y = self.ln_post(xseq[:, 0, :].reshape(B, T, -1))  # semantic features, (B, T, D)
            if self.proj is not None:
                y = (y @ self.proj).mean(dim=1).type(dtype)  # average CLS feat over multiple frames
            return [x, x, x, x], y
        
        return [x, x, x, x]



class CLIPTextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, freeze=True, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device('cuda')

        # clip model components
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.requires_backprop = False
        
        # hyperparamters
        self.context_length_cls = cfg.MODEL.CLIP.LEN_CONTEXT_CLS  # need to be large, e.g., 24
        self.context_length_prompt = cfg.MODEL.CLIP.LEN_CONTEXT_PROMPT  # small length is okay, e.g., 8
        self.enable_pos_emb = cfg.MODEL.CLIP.ENABLE_POS_EMBED
        self.dim_embed = self.token_embedding.weight.size(-1)

        if cfg.MODEL.CLIP.CONTEXT_INIT:
            # initialize learnable context
            embedding_ctx, self.token_ids = self.init_soft_prompt(cfg.MODEL.CLIP.CONTEXT_INIT)
            if cfg.MODEL.CLIP.SOFT_PROMPT:  # make sure model.named_parameters() and model.state_dict() can find it
                self.register_parameter("soft_prompt", nn.Parameter(embedding_ctx))
            else:
                self.soft_prompt = embedding_ctx
        else:
            self.soft_prompt = None
        self.vocab_token_embeddings = {}  # dynamically updated in self.set_vocabulary()


    def init_soft_prompt(self, context_init=None):
        """ context_init: "a photo of "
        """
        # get the initial context embeddings
        n_ctx = len(context_init.split())
        token_ctx = self.tokenize([context_init],
                                  context_length=self.context_length_prompt).to(self.device)
        with torch.no_grad():
            embedding_ctx = self.token_embedding(token_ctx)
        embedding_ctx = embedding_ctx[0, 1 : 1 + n_ctx, :]
        # get the tokens of complete sentence input
        token_ids = self.tokenize([context_init.strip() + ' x'],  # 'a photo of x'
                                  context_length=self.context_length_prompt).to(self.device)  # (1, L)
        return embedding_ctx, token_ids
    

    def get_token_embeddings(self, batch_text):
        tokenized = self.tokenize(batch_text, context_length=self.context_length_cls).to(self.device)
        frozen_embedding = torch.zeros((len(batch_text), self.dim_embed))
        with torch.no_grad():
            embedding = self.token_embedding(tokenized)
            for idx, rep in enumerate(embedding):
                eos_idx = tokenized[idx].argmax()  # find the index of the EOS token
                frozen_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)  # average embeddings over the tokens for each class
        return frozen_embedding


    def set_vocabulary(self, text_data, embeddings=None):
        # update vocabulary
        self.text_data = text_data
        # update the class embedding
        if embeddings is not None:
            self.vocab_token_embeddings = embeddings
        else:
            new_classes = list(set(self.text_data.keys()).difference(set(self.vocab_token_embeddings.keys())))
            if len(new_classes) > 0:  # when eval_open=True
                if isinstance(list(self.text_data.values())[0]['caption'], str):  # single caption
                    text_captions = [self.text_data[vocab]['caption'].strip()
                                    for vocab in new_classes]
                    token_embeds = self.get_token_embeddings(text_captions)
                    # store info
                    self.vocab_token_embeddings.update({vocab: {'embed': token_embeds[k]} 
                                                        for k, vocab in enumerate(new_classes)})
                else:
                    for vocab in new_classes:
                        text_captions = self.text_data[vocab]['caption']
                        token_embeds = self.get_token_embeddings(text_captions)
                        # store info
                        self.vocab_token_embeddings.update({vocab: {'embed': token_embeds}})


    def tokenize(self, text, context_length=77):
        return torch.cat([clip.tokenize(tok, context_length=context_length) for tok in text])

    def encode_text(self, text, context_length=77, enable_pos_emb=True):
        token_ids = self.tokenize(text, context_length)
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features
    

    def construct_token_tensors(self):
        """ labels: a list of onehot vectors with size of (N, K),
            where N is the number of boxes, and K is the number of classes
            '[SOS] a photo of [CLS] [EOS]'
        """
        num_classes = len(self.vocab_token_embeddings)
        class_token_ids = self.token_ids.repeat(num_classes, 1)  # (K, L)
        with torch.no_grad() if not self.requires_backprop else nullcontext():
            token_tensor = self.token_embedding(class_token_ids).type(self.dtype)  # (K, L=8, D=768)

        # insert the class embeddings into the [CLS] position
        eos_idx = int(self.token_ids[0].argmax())
        class_embeds = torch.stack([self.vocab_token_embeddings[vocab]['embed'] for vocab in list(self.text_data.keys())])
        token_tensor[:, eos_idx - 1, :] = class_embeds.type(self.dtype)  # (K, D)
    
        if self.soft_prompt is not None:
            # update learnable context embedding
            token_tensor[:, 1:len(self.soft_prompt)+1, :] = self.soft_prompt.type(self.dtype)  # (n_ctx, D)

        return token_tensor


    def forward(self, token_ids, token_tensors=None):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.token_embedding(token_ids)

        text_features = text_features.type(self.dtype)
        x = (
            text_features + self.positional_embedding.type(self.dtype)
            if self.enable_pos_emb
            else text_features
        )
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        tf = (
            x[
                torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
            ]  # POS of <EOS>
            @ self.text_projection
        )
        return tf


    def encode_person(self, prompt_ensemble=False):
        if prompt_ensemble:  # 85 templates from CLIP_Surgery
            prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
        
        input_text = [prompt.format("person") for prompt in prompt_templates] if prompt_ensemble else ["a photo of person"]
        with torch.no_grad():
            text_features = self.encode_text(input_text)  # (85, D)
        
        if prompt_ensemble:
            # aggregation
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_feat_out = text_features.mean(dim=0)
            text_feat_out /= text_feat_out.norm()
        else:
            text_feat_out = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
        
        return text_feat_out



class CLIPModel(nn.Module):
    def __init__(self, cfg, clip_model, dtype=torch.float32):
        super(CLIPModel, self).__init__()
        self.dtype = dtype
        self.device = torch.device('cuda')
        self.num_pathways = 1
    
        self.visual_encoder = CLIPVisualEncoder(cfg, clip_model)
        self.text_encoder = CLIPTextEncoder(cfg, clip_model)

        self.dim_embed = clip_model.visual.transformer.width
        self.tau_inv = clip_model.logit_scale.requires_grad_(False).exp()
    
        self.cam_method = cfg.MODEL.CLIP.CAM_METHOD
        self.use_person_embed = cfg.MODEL.CLIP.USE_PERSON_EMBED
        if self.use_person_embed:
            self.person_embed = self.text_encoder.encode_person(prompt_ensemble=True)

    
    def freeze_modules(self, mismatched_keys):
        # freeze the vision model
        for n, p in self.visual_encoder.named_parameters():
            if 'visual_encoder.' + n not in mismatched_keys:
                p.requires_grad=False
            else:
                # has some learnable parameters that need backprop
                self.visual_encoder.requires_backprop = True

        # freeze the text model
        for n, p in self.text_encoder.named_parameters():
            if 'text_encoder.' + n not in mismatched_keys:
                p.requires_grad=False
            else:
                # has some learnable parameters that need backprop
                self.text_encoder.requires_backprop = True
        # freeze others
        self.tau_inv.requires_grad = False

    
    def forward(self, x_list):
        with torch.no_grad() if not self.visual_encoder.requires_backprop else nullcontext():
            return self.visual_encoder(x_list)
    
    def forward_text(self, device=torch.device('cuda'), cond=None):
        # soft context is added
        token_tensors = self.text_encoder.construct_token_tensors()  # (K, L=8, D)
        # run forzen CLIP text encoder
        with torch.no_grad() if not self.text_encoder.requires_backprop else nullcontext():
            text_features = self.text_encoder(
                self.text_encoder.token_ids,
                token_tensors
            )  # (K, D)
        return text_features.type(self.dtype)
    

    def get_ritsm(self, cls_token_feat, patch_token_feat, text_features, input_sizes):
        """ cls_token_feat: (B, D)
            patch_token_feat: (B, D, T, h, w)
            text_features: (K, D)
            input_sizes: (B, 2), (height, width)
        """
        if self.use_person_embed:
            batch_size = patch_token_feat.size(0)
            cls_feat = self.person_embed.unsqueeze(0).expand(batch_size, -1)  # (B, D)
        else:
            if isinstance(text_features, list):
                # text_features = torch.stack([feat.mean(dim=0) for feat in text_features], dim=0)
                text_features = torch.stack([feat[0] for feat in text_features], dim=0)
            pred_cls = (cls_token_feat @ text_features.t()).argmax(dim=-1)  # (B,)
            cls_feat = text_features[pred_cls] # (B, D)

        T, h, w = patch_token_feat.size()[-3:]
        patch_features = patch_token_feat[:, :, int(T//2)]  # (B, D, h, w) only use the keyframe patches
        patch_features = rearrange(patch_features, 'b d h w -> b (h w) d')  # (B, hw, D)
        
        # layernorm, projection, and normalization
        patch_features = self.visual_encoder.ln_post(patch_features)
        if self.visual_encoder.proj is not None:
            patch_features = patch_features @ self.visual_encoder.proj.type(patch_features.dtype)   # (B, hw, D)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        # image-text similarity
        it_sim = torch.bmm(patch_features, cls_feat.unsqueeze(-1))  # (B, hw, 1)

        # 1: reshape
        featmap_attention = rearrange(it_sim[:, :, 0], 'b (h w) -> b h w', h=h, w=w)
        
        # since the target size could be different in a batch, we have to for-loop each sample here:
        batch_attention_maps = []
        for feat_attn, target_size in zip(featmap_attention, input_sizes):
            # 2: resize
            H, W = int(target_size[0]), int(target_size[1])
            attn_map = F.interpolate(feat_attn.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode='bilinear')  # (1, 1, H, W)
            # 3: normalization
            attn_map = attn_map[0, 0].view(-1)  # (H*W, )
            attn_map -= attn_map.min(dim=-1, keepdim=True)[0]
            attn_map /= attn_map.max(dim=-1, keepdim=True)[0]
            attn_map = attn_map.view(H, W)  # (H, W)
            # 4: reverse attention
            attn_map = 1 - attn_map
            batch_attention_maps.append(attn_map)

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
                return self.get_ritsm(cls_token_feat, patch_token_feat, text_features, input_sizes)
            else:
                raise NotImplementedError


def key_transform(key):
    if key in ['positional_embedding', 'text_projection', 'token_embedding.weight', 'ln_final.weight', 'ln_final.bias']:  # 5 keys
        return 'text_encoder.' + key
    if 'visual.' in key:
        return 'visual_encoder.' + key.split('visual.')[-1]  # 152 keys
    if ('visual.' not in key) and ('transformer' in key):
        return 'text_encoder.' + key  # 144 keys


def model_diff(src_model, target_model):
    
    loaded_state_dict = src_model.state_dict()
    # key transformation
    load_transformed = dict()
    for k, v in loaded_state_dict.items():
        load_transformed[key_transform(k)] = v  # logit_scale missed here

    model_keys = set([k for k in list(target_model.state_dict().keys())])  # state_dict() ignores the tau_inv
    load_keys = set(load_transformed.keys())

    # weights that need to train
    mismatched_keys = list(model_keys.difference(load_keys))

    return mismatched_keys


def build_clip_backbone(cfg):
    # load the pre-trained CLIP model
    clip_model, _ = load(
        cfg.MODEL.CLIP.ARCH, context_length=cfg.MODEL.CLIP.LEN_CONTEXT_PROMPT
    )

    model = CLIPModel(cfg, clip_model)

    # find the new model parameters
    mismatched_keys = model_diff(clip_model, model)

    model.freeze_modules(mismatched_keys)

    return model