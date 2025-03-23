import torch
import numpy as np
import cv2


def clip_forward(model, image, text):
    # get patch token features
    image_features, encoder_out = model.encode_image(image, transformer_output=True)  # (N, L, D)
    text_features = model.encode_text(text)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    return logits_per_image, encoder_out, text_features


@torch.no_grad()
def ritsm_clip(image, text, model, device, index=None, cam_size=None, return_logits=False, attn_grad=False):
    # forward pass
    logits_per_image, encoder_out, text_features = clip_forward(model, image, text)
    probs = logits_per_image.softmax(dim=-1)
    if index is None:
        # locate the largest score of img-text pair
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    
    input_size = model.visual.input_resolution  # 224
    patch_features = encoder_out[:, 1:, :]  # (B, 7*7, 768)
    heatmap_size = int(patch_features.size(1)**0.5)  # 7

    # projection
    patch_features = model.visual.ln_post(patch_features)
    if model.visual.proj is not None:
        patch_features = patch_features @ model.visual.proj   # (B, 7*7, 512)

    # normalize
    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (K=1, 512)

    # image-text similarity
    it_sim = patch_features @ text_features.t()  # (B, 7*7, K=1)

    # reshape & resize
    image_relevance_all = it_sim[:, :, index].view(-1, 1, heatmap_size, heatmap_size)  # (B, 1, 7, 7)
    image_relevance_all = torch.nn.functional.interpolate(image_relevance_all.float(), size=input_size, mode='bilinear')  # (B, 1, H, W)
    image_relevance = image_relevance_all[0]  # assume batch_size = 1
    image_relevance = image_relevance.reshape(input_size, input_size).detach().cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())    
    # reverse
    image_relevance = np.fabs(1 - image_relevance)

    out = cv2.resize(image_relevance, cam_size) if cam_size is not None else image_relevance
    if return_logits:
        return out, logits_per_image
    return out


@torch.no_grad()
def ritsm_clipvip(video, text, model, device, index=None, cam_size=None, return_logits=False, attn_grad=False, use_mask=False):
    """ video: (B, T, C, H, W)
        text: (K, L)
    """
    num_proxy = model.config.vision_additional_config.add_cls_num + 1
    eos_idx = text.argmax(dim=-1)
    num_frames = video.size(1)
    
    input_size = model.config.vision_config.image_size  # 224
    patch_size = model.config.vision_config.patch_size  # 16
    num_patches = int(input_size // patch_size)  # 14

    # run forward pass
    out_dict = model(text, video)
    logits_per_image = out_dict['logits_per_image']

    if index is None:
        # locate the largest score of img-text pair
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)

    # get patch features from the last vision encoder block
    patch_features = out_dict['vision_model_output']['last_hidden_state'][:, num_proxy:, :]  # (B, T*14*14, 768)
    assert num_frames * (num_patches ** 2) == patch_features.size(1)
    
    # layernorm, projection, and normalization
    patch_features = model.vision_model.post_layernorm(patch_features)
    patch_features = model.visual_projection(patch_features)  # 768 --> 512
    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)  # (B, T*14*14, 512)

    # get the text features
    text_features = out_dict['text_embeds']  # after layernorm, projection, and normalization

    # image-text similarity
    it_sim = patch_features @ text_features.t()  # (B, T*14*14, K=1)
    
    if use_mask:
        th_attn = get_attn_mask(it_sim[0, :, index].view(num_frames, -1), threshold=0.6)
        it_sim = it_sim * th_attn.view(-1).unsqueeze(0).unsqueeze(-1) 
    
    # reshape & resize
    image_relevance_all = it_sim[:, :, index].view(-1, num_frames, num_patches, num_patches)  # (B, T, 14, 14)
    image_relevance_all = torch.nn.functional.interpolate(image_relevance_all.float(), size=input_size, mode='bilinear')  # (B, T, H, W)

    # assume batch_size = 1
    image_relevance_all = image_relevance_all[0]

    all_maps = []
    for image_relevance in image_relevance_all:
        image_relevance = image_relevance.reshape(input_size, input_size).detach().cpu().numpy()
        # normalize and reverse
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        # reverse
        image_relevance = np.fabs(1 - image_relevance)
        atten_map = cv2.resize(image_relevance, cam_size) if cam_size is not None else image_relevance    
        all_maps.append(atten_map)

    out = np.stack(all_maps, axis=0)
    
    if return_logits:
        return out, logits_per_image
    return out


def get_attn_mask(attentions, threshold=0.6):
    """ attentions: (T, L)
    """
    nh = attentions.size(0)
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)  # dim=-1 by default
    th_attn = th_attn.view(nh, -1)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head].view(-1)]
    return th_attn