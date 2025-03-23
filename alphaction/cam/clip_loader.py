import torch

#* For CLIP ViT
def reshape_transform(tensor, height=None, width=None):
    if height or width is None:
        grid_square = len(tensor) - 1
        if grid_square ** 0.5 % 1 == 0:
            height = width = int(grid_square**0.5)
        else:
            raise ValueError("Heatmap is not square, please set height and width.")
    result = tensor[1:, :, :].reshape(
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(2, 0, 1)
    return result.unsqueeze(0)

def load_clip(clip_version, attn_prob=True, attn_grad=True, attn_last_only=True, resize='adapt', custom=False, model_weight=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'vit' in clip_version.lower() and not custom: #* This is no necessary, for experimental usage, hila CLIP will hook all attentions.
        from hila_clip import clip
        clip_model, preprocess = clip.load(clip_version, device=device, jit=False)
    
    elif 'clip-vip' in clip_version.lower():
        import sys, clip
        sys.path.append("../../")
        from alphaction.modeling.encoders.clipvip import loader
        clip_model, preprocess = loader.load(clip_version, 
                                             attn_prob=attn_prob,
                                             attn_grad=attn_grad, 
                                             attn_last_only=attn_last_only,
                                             device=device, model_weight=model_weight)

    elif custom:
        from hila_clip import clip
        clip_model, preprocess = clip.load(clip_version, device=device, jit=False)

    else:
        import clip
        clip_model, preprocess = clip.load(clip_version, device=device)

    if clip_version.startswith("RN"):
        target_layer = clip_model.visual.layer4[-1]
        cam_trans = None
    elif 'clip-vip' in clip_version.lower():
        target_layer = clip_model.vision_model.encoder.layers[-1]
        cam_trans = reshape_transform
    else:
        target_layer = clip_model.visual.transformer.resblocks[-1]
        cam_trans = reshape_transform

    if resize == 'raw': # remove clip resizing
        if not custom:
            raise Exception("Raw input needs to use custom clip.") 
        preprocess.transforms.pop(0)
        preprocess.transforms.pop(0)
    elif resize == 'adapt': # adapt to clip size
        from torchvision import transforms
        crop_size = preprocess.transforms[1].size # resize to crop size so that no information will be cropped
        preprocess.transforms.insert(0, transforms.Resize(crop_size))
    # clip_model = torch.nn.DataParallel(clip_model)
    return clip_model, preprocess, target_layer, cam_trans, clip

def load_clip_from_checkpoint(checkpoint, model):
    checkpoint = torch.load(checkpoint, map_location='cpu')

    # # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
    # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
    # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
    # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

    model.load_state_dict(checkpoint['model_state_dict'])
    return model