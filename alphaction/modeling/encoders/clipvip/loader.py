import torch
import torch.nn as nn
from transformers.models.clip.configuration_clip import CLIPConfig
from .CLIP_ViP import CLIPModel
from .clipvip_encoder import Dict2Class
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from typing import Union
from PIL import Image



class VidCLIP(nn.Module):
    def __init__(self, args):
        super(VidCLIP, self).__init__()
        clipconfig = CLIPConfig.from_pretrained(args.clip_config)
        setattr(clipconfig, "vision_additional_config", args.clip_vision_additional_config)
        
        if getattr(args, 'attn_prob', False):
            setattr(clipconfig.vision_config, 'attn_prob', True)
        
        if getattr(args, 'attn_grad', False):
            setattr(clipconfig.vision_config, 'attn_grad', True)
        
        if getattr(args, 'attn_last_only', False):
            setattr(clipconfig.vision_config, 'attn_last_only', True)
        
        self.vision_additional_config = args.clip_vision_additional_config
        if args.clip_weights:
            self.clipmodel = CLIPModel.from_pretrained(args.clip_weights, config=clipconfig)
        else:
            self.clipmodel = CLIPModel(clipconfig)
        
        # init logit scale from 
        logit_scale_value = self.vision_additional_config.logit_scale_init_value
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)
    
    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)

    def forward(self, video, text_input_ids, text_input_mask, \
                image=None, caption_ids=None, caption_masks=None):
        """
        video [B, n_clips*num_frms, C, H, W]
        text_input_ids [B, L]
        text_input_mask [B, L]
        image [B, img_num, C, H, W]
        caption_ids [B, img_num, L]
        caption_masks [B, img_num, L]
        """
        B, N, C, H, W = video.shape

        if self.vision_additional_config.type == "ViP":
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video,
                    "return_loss": False}
            outputs = self.clipmodel(**inputs)
            results = {}
            results["text_features"] = outputs["text_embeds"]
            results["vis_features"] = outputs["image_embeds"]
            # results["loss"] = outputs["loss"]
        else:
            video = video.reshape(-1, C, H, W)
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video}
            outputs = self.clipmodel(**inputs)
            vis_features = outputs["vision_model_output"][1]

            vis_features = self.clipmodel.visual_projection(vis_features)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            vis_features = vis_features.reshape(B, N, -1).mean(1)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            
            results = {}
            results["text_features"] = outputs["text_embeds"]
            results["vis_features"] = vis_features
        if image is not None:
            B, img_num, C, H, W = image.shape
            L = caption_ids.shape[-1]
            inputs = {"input_ids": caption_ids.reshape(-1, L),
                    "attention_mask": caption_masks.reshape(-1, L),
                    "pixel_values": image.reshape(-1, 1, C, H, W),
                    "return_loss": False}
            outputs = self.clipmodel(**inputs)
            results["img_features"] = outputs["image_embeds"]
            results["cap_features"] = outputs["text_embeds"]
        
        return results
    
    def forward_video(self, video):
        inputs = {"pixel_values": video,
                "if_norm": True}
        video_features = self.clipmodel.get_image_features(**inputs)
        return video_features
    
    def forward_text(self, text_input_ids, text_input_mask):
        inputs = {"input_ids": text_input_ids,
                "attention_mask": text_input_mask,
                "if_norm": True}
        text_features = self.clipmodel.get_text_features(**inputs)
        return text_features

    def freeze_text_encoder(self, freeze_text_proj):
        freeze_list = [self.clipmodel.text_model]
        if freeze_text_proj:
            freeze_list.append(self.clipmodel.text_projection)
        for m in freeze_list:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False



def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
    """operated in-place, no need to return `model`"""

    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(
            loaded_state_dict_or_path, map_location="cpu")
    else:
        loaded_state_dict = loaded_state_dict_or_path
    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]
    model.load_state_dict(toload, strict=False)


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def load(name: str, 
         attn_grad: bool = True, 
         attn_prob: bool = True, 
         attn_last_only: bool = True, 
         device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", model_weight=None):
    
    weights = {'clip-vip-b16': "pretrained/pretrain_clipvip_base_16.pt",
               'clip-vip-b32': "pretrained/pretrain_clipvip_base_32.pt"}
    class args:
        clip_config = "openai/clip-vit-base-patch16"
        clip_vision_additional_config = Dict2Class({
            "temporal_size": 12,
            "if_use_temporal_embed": "1",
            "logit_scale_init_value": 4.6,
            "add_cls_num": 3})
        clip_weights = "openai/clip-vit-base-patch16"
        e2e_weights_path = weights[name] if model_weight is None else model_weight
    
    setattr(args, 'attn_grad', attn_grad)
    setattr(args, 'attn_prob', attn_prob)
    setattr(args, 'attn_last_only', attn_last_only)
    
    model = VidCLIP(args)

    load_state_dict_with_mismatch(model, args.e2e_weights_path)

    model.to(device)

    return model.clipmodel,  _transform(224)