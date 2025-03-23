from torch import nn

from ..backbone import build_backbone
from ..stm_decoder.stm_decoder import build_stm_decoder
import fvcore.nn.weight_init as weight_init
import torch
from einops import rearrange
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, t, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class STMDetector(nn.Module):
    def __init__(self, cfg):
        super(STMDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.residual_lateral = cfg.MODEL.BACKBONE.RESIDUAL_LATERAL
        self.scales = [4, 2, 1, 0.5]

        self._construct_space(cfg)
        self.stm_head = build_stm_decoder(cfg)

        self.cam_method = getattr(self.backbone, 'cam_method', "")
        self.prior_boxes_init = cfg.MODEL.PRIOR_BOXES_INIT

        self.cond_modality = cfg.MODEL.STM.COND_MODALITY
        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY
        self.pre_extract_feat = cfg.MODEL.PRE_EXTRACT_FEAT

        self.dtype = torch.float32


    def _construct_space(self, cfg):
        out_channel = cfg.MODEL.STM.HIDDEN_DIM
        backbone_arch = cfg.MODEL.BACKBONE.CONV_BODY.lower()
        if 'vit' in backbone_arch or 'vip' in backbone_arch or 'viclip' in backbone_arch:
            in_channels = [self.backbone.dim_embed] * 4
            self.lateral_convs = nn.ModuleList()

            for idx, scale in enumerate(self.scales):
                dim = in_channels[idx]
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                        LayerNorm(dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                    ]
                    out_dim = dim // 4
                elif scale == 2.0:
                    layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                    out_dim = dim // 2
                elif scale == 1.0:
                    layers = []
                    out_dim = dim
                elif scale == 0.5:
                    layers = [nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                    out_dim = dim
                else:
                    raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
                layers.extend(
                    [
                        nn.Conv3d(
                            out_dim,
                            out_channel,
                            kernel_size=1,
                            bias=False,
                        ),
                        LayerNorm(out_channel),
                        nn.Conv3d(
                            out_channel,
                            out_channel,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ]
                )
                layers = nn.Sequential(*layers)

                self.lateral_convs.append(layers)
        else:
            if self.backbone.num_pathways == 1:
                in_channels = [256, 512, 1024, 2048]
            else:
                in_channels = [256+32, 512+64, 1024+128, 2048+256]
            self.lateral_convs = nn.ModuleList()
            for idx, in_channel in enumerate(in_channels):
                lateral_conv = nn.Conv3d(in_channel, out_channel, kernel_size=1)
                weight_init.c2_xavier_fill(lateral_conv)
                self.lateral_convs.append(lateral_conv)


    def space_forward(self, features, residual=False):
        mapped_features = []
        for i, feature in enumerate(features):
            feat_out = self.lateral_convs[i](feature)
            if residual:
                # projection & upsampling: (B, D, T, h, w) --> (B, d, T, H, W)
                feat_idty = self.backbone.visual_encoder.project_patch_features(feature)
                B, d, T, h, w = feat_idty.size()
                feat_idty = rearrange(feat_idty, 'b d t h w -> (b t) d h w')
                feat_idty = F.interpolate(feat_idty, size=(int(self.scales[i] * h), int(self.scales[i] * w)), mode='bilinear')
                feat_idty = rearrange(feat_idty, '(b t) d h w -> b d t h w', b=B, t=T)
                feat_out = feat_out + feat_idty
            mapped_features.append(feat_out)
        return mapped_features

    def forward(self, slow_video, fast_video, whwh, boxes=None, labels=None, extras={}, part_forward=-1):
        
        if self.backbone.num_pathways == 1:
            features = self.backbone([slow_video])
        else:
            features = self.backbone([slow_video, fast_video])
        
        cls_feat_visual = None
        if self.backbone.visual_encoder.use_cls_feat:
            features, cls_feat_visual = features

        mapped_features = self.space_forward(features, residual=self.residual_lateral)

        if self.open_vocabulary:
            # get the current text feature embeddings
            text_features = self.backbone.forward_text(device=slow_video.device, cond=None)
            tau_inv = self.backbone.tau_inv
            num_cls = len(self.backbone.text_encoder.text_data)
        else:
            text_features, tau_inv = None, 100
        
        if (not self.prior_boxes_init) and self.cam_method:
            assert self.backbone.visual_encoder.use_cls_feat, "text condition relies on USE_CLS_FEAT=True for pre-matching"
            cams = self.backbone.get_cam(cls_feat_visual, features[0], text_features, whwh[:, [1, 0]])
            extras.update({'cams': cams})

        return self.stm_head(mapped_features, whwh, gt_boxes=boxes, labels=labels, 
                             text_features=text_features, tau_inv=tau_inv, cls_feat=cls_feat_visual, patch_feat=None, text_token_feats=None, extras=extras)


def build_detection_model(cfg):
    return STMDetector(cfg)