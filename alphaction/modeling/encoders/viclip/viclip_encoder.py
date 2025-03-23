import logging

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .viclip import ViCLIP
from alphaction.modeling.backbone.vit_utils import interpolate_pos_embed_online



class ViCLIPVisualEncoder(nn.Module):
    def __init__(self, cfg, clip_model, dtype=torch.float32):
        super(ViCLIPVisualEncoder, self).__init__()
        self.dtype = dtype
        self.device = torch.device('cuda')

        self.encode_vision = clip_model.vision_encoder
        self.ln_post = clip_model.vision_encoder.ln_post
        self.proj = clip_model.vision_encoder.proj

        self.use_cls_feat = cfg.MODEL.STM.USE_CLS_FEAT
        self.interpolator = interpolate_pos_embed_online
    
    
    def project_patch_features(self, patch_features):
        """ patch_features: (B, D, T, h, w)
        """
        x = patch_features.permute(0, 2, 3, 4, 1).contiguous()
        x = self.ln_post(x)  # layer_norm
        x = x @ self.proj    # projection: 1024 --> 768
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x_list):
        dtype = x_list[0].dtype
        x = x_list[0].type(self.dtype)  # slow video (b, 3, 16, 256, ?)
        B, C, T, H, W = x.size()
        # get CLS token features
        yc, yp, ws = self.encode_vision(x, interpolator=self.interpolator, return_patch=True)  # if train the encoder, masking_prob needs to be set
        yp = yp.permute(1, 0, 2)  # (B, Thw, D)
        yp = rearrange(yp, 'b (t h w) d -> b d t h w', t=T, h=ws[0], w=ws[1])  # ()
        
        if self.use_cls_feat:
            return [yp, yp, yp, yp], yc

        return [yp, yp, yp, yp]


class ViCLIPTextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device('cuda')

        self.context_prompt = cfg.MODEL.ViCLIP.CONTEXT_INIT

        # clip model components
        self.text_encoder = clip_model.text_encoder
        self.max_txt_l = clip_model.max_txt_l
        self.vocab_text_features = {}  # dynamically updated in self.set_vocabulary()
    

    def get_token_embeddings(self, batch_text):
        with torch.no_grad():
            text = self.text_encoder.tokenize(
                batch_text, context_length=self.max_txt_l
            ).to(self.device)
            text_features = self.text_encoder(text).float()
        return text_features


    def set_vocabulary(self, text_data, embeddings=None):
        # update vocabulary
        self.text_data = text_data
        # update the class embedding
        if embeddings is not None:
            self.vocab_text_features = embeddings
        else:
            new_classes = list(set(self.text_data.keys()).difference(set(self.vocab_text_features.keys())))
            if len(new_classes) > 0:  # when eval_open=True
                if isinstance(list(self.text_data.values())[0]['caption'], str):  # single caption
                    text_captions = [(self.context_prompt.strip() + ' ' + self.text_data[vocab]['caption']).strip()
                                    for vocab in new_classes]
                    text_features = self.get_token_embeddings(text_captions)
                    # store info
                    self.vocab_text_features.update({vocab: {'feat': text_features[k]} 
                                                        for k, vocab in enumerate(new_classes)})
                else:
                    for vocab in new_classes:
                        text_captions = self.text_data[vocab]['caption']
                        text_features = self.get_token_embeddings(text_captions)
                        # store info
                        self.vocab_text_features.update({vocab: {'feat': text_features}})


class ViCLIPModel(nn.Module):
    """docstring for ViCLIP"""

    def __init__(self, cfg, clip_model, dtype=torch.float32):
        # initialize parent class
        super(ViCLIPModel, self).__init__()
        
        # customize the modules
        self.dtype = dtype
        self.device = torch.device('cuda')
        self.num_pathways = 1
    
        self.visual_encoder = ViCLIPVisualEncoder(cfg, clip_model, dtype=dtype)
        self.text_encoder = ViCLIPTextEncoder(cfg, clip_model, dtype=dtype)
        self.tau_inv = 1.0 / clip_model.temp

        self.dim_embed = clip_model.vision_width

        self.cam_method = cfg.MODEL.ViCLIP.CAM_METHOD
        self.use_attn = getattr(cfg.MODEL.ViCLIP, 'USE_ATTN', False)
        self.use_grad = getattr(cfg.MODEL.ViCLIP, 'USE_GRAD', False)
        if self.cam_method == 'HilaCAM': assert self.use_attn

    
    def forward(self, x_list):
        with torch.no_grad():
            return self.visual_encoder(x_list)

    def forward_text(self, device=torch.device('cuda'), cond=None):
        text_features = []
        for vocab, text_data in self.text_encoder.text_data.items():
            feat = self.text_encoder.vocab_text_features[vocab]['feat']
            text_features.append(feat)
        if len(feat.size()) == 1:
            text_features = torch.stack(text_features, dim=0)  # (K, D=768)
        return text_features
    

    def get_ritsm(self, cls_token_feat, patch_token_feat, text_features, input_sizes):
        """ cls_token_feat: (B, D)
            patch_token_feat: (B, D, T, h, w)
            text_features: (K, D)
            input_sizes: (B, 2), (height, width)
        """

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
        with torch.no_grad():
            if self.cam_method == 'RITSM':
                return self.get_ritsm(cls_token_feat, patch_token_feat, text_features, input_sizes)
            else:
                raise NotImplementedError



def load_viclip(cfg, freeze_all=False):
    model = ViCLIP(pretrain=getattr(cfg.MODEL.ViCLIP, 'WEIGHT_FILE', None))
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False
    model = model.eval()
    return model


def build_viclip_backbone(cfg):
    # load the pre-trained ViCLIP model
    viclip_model = load_viclip(cfg, freeze_all=True)

    # construct the customized model
    model = ViCLIPModel(cfg, viclip_model)

    return model

