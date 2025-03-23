from alphaction.modeling import registry
from . import slowfast, i3d, video_model_builder

@registry.BACKBONES.register("Slowfast-Resnet50")
@registry.BACKBONES.register("Slowfast-Resnet101")
def build_slowfast_resnet_backbone(cfg):
    model = slowfast.SlowFast(cfg)
    return model

@registry.BACKBONES.register("PySlowonly")
def build_pyslowonly_resnet_backbone(cfg):
    model = video_model_builder.ResNet(cfg)
    return model

@registry.BACKBONES.register("PySlowfast-R50")
@registry.BACKBONES.register("PySlowfast-R101")
def build_pyslowfast_resnet_backbone(cfg):
    model = video_model_builder.SlowFast(cfg)
    return model

@registry.BACKBONES.register("MAE-ViT-B")
@registry.BACKBONES.register("MAE-ViT-L")
def build_mae_vit_backbone(cfg):
    model = video_model_builder.ViT(cfg)
    return model

@registry.BACKBONES.register("I3D-Resnet50")
@registry.BACKBONES.register("I3D-Resnet101")
@registry.BACKBONES.register("I3D-Resnet50-Sparse")
@registry.BACKBONES.register("I3D-Resnet101-Sparse")
def build_i3d_resnet_backbone(cfg):
    model = i3d.I3D(cfg)
    return model

# OpenAI CLIP
@registry.BACKBONES.register("ViT-B/16")
@registry.BACKBONES.register("ViT-B/32")
@registry.BACKBONES.register("ViT-L/14")
def build_clip_vit_backbone(cfg):
    from alphaction.modeling.encoders.openai_clip.clip_encoder import build_clip_backbone
    model = build_clip_backbone(cfg)
    return model

# CLIP-ViP
@registry.BACKBONES.register("ViP-B/16")
@registry.BACKBONES.register("ViP-B/32")
def build_clipvip_backbone(cfg):
    from alphaction.modeling.encoders.clipvip.clipvip_encoder import build_clipvip_backbone
    model = build_clipvip_backbone(cfg)
    return model

# ViCLIP from InternVideo
@registry.BACKBONES.register("ViCLIP-L/14")
def build_viclip_backbone(cfg):
    from alphaction.modeling.encoders.viclip.viclip_encoder import build_viclip_backbone
    model = build_viclip_backbone(cfg)
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
