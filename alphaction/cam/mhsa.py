import torch


def get_multi_head_mask(attentions, threshold=0.6):
    nh, np = attentions.size(0), attentions.size(-1)
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)  # dim=-1 by default
    th_attn = th_attn.view(nh, -1)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head].view(-1)]
    if len(attentions.size()) == 3:
        th_attn = th_attn.view(nh, -1, np)
    return th_attn


def get_masked_attention_map(attentions, nh, heatmap_size, cam_size, mask=None):
    if mask is not None:
        # apply mask on attention map
        attentions = attentions * mask.float()  # (num_heads, N, L)
    # normalize within each frame
    attentions -= attentions.min(dim=-1, keepdim=True)[0]
    attentions /= attentions.max(dim=-1, keepdim=True)[0]
    
    num_frames = attentions.size(1) if len(attentions.size()) == 3 else 1
    # average over multi-heads as the final attention
    attentions = attentions.reshape(nh, num_frames, heatmap_size[0], heatmap_size[1]).mean(dim=0, keepdim=True)
    
    if cam_size is not None:
        # interpolate
        attentions = torch.nn.functional.interpolate(attentions, size=(cam_size[1], cam_size[0]), mode="bilinear")[0]
    return attentions.cpu().numpy()


@torch.no_grad()
def mhsa_clip(image, model, cam_size=None, threshold=0.6):
    # get patch token features
    _, attn_last = model.encode_image(image, last_attn_output=True)  # (B, num_heads, L, D)
    nh = attn_last.shape[1] # number of head
    
    # we keep only the output patch attention
    # assume batch_size = 1
    attentions = attn_last[0, :, 0, 1:].reshape(nh, -1)  # (num_heads, 7*7)
    heatmap_size = [int(attentions.size(-1)**0.5), int(attentions.size(-1)**0.5)]  # 7
    
    th_attn = get_multi_head_mask(attentions, threshold)
    
    attn_map = get_masked_attention_map(attentions, nh, heatmap_size, cam_size, mask=th_attn)  # (1, H, W)

    return attn_map[0]


@torch.no_grad()
def mhsa_clipvip(video, model, cam_size=None, threshold=0.6):
    """ video: (B, T, C, H, W)
        text: (K, L)
        cam_size: (W, H)
    """
    num_proxy = model.config.vision_additional_config.add_cls_num + 1
    num_heads = model.config.vision_config.num_attention_heads
    num_frames = video.size(1)

    # run forward pass to get the last block attentions
    _, heatmap_size = model.get_image_features(video, return_ws=True)  # (h,w)
    last_block = list(dict(model.vision_model.encoder.layers.named_children()).values())[-1]
    attn_inter = last_block.attn_probs['inter']  # [B*num_heads, M, M+N*L] where M=4
    attn_intra = last_block.attn_probs['intra']  # [B*num_heads*N, L, M+L] where L=196 if input_size=224
    
    num_patches = attn_intra.shape[-2] # L
    attentions_inter = attn_inter[:, 0, num_proxy:].reshape(-1, num_heads, num_frames, num_patches)[0]  # [B*num_heads, N*L] --> [num_heads, N, L]
    # attentions_intra = attn_intra[:, 0, num_proxy:].reshape(-1, num_heads, num_frames, num_patches)[0]  # [B*num_heads*N, L] --> [num_heads, N, L]
    
    th_attn = get_multi_head_mask(attentions_inter, threshold)
    attn_map = get_masked_attention_map(attentions_inter, num_heads, heatmap_size, cam_size, mask=th_attn)  # (T, H, W)
    
    # temporal weights
    temporal_weights = attn_inter[:, 0, num_proxy:].reshape(-1, num_frames, num_patches).sum(dim=-1)  # [B*num_heads, N]
    temporal_weights = temporal_weights.reshape(-1, num_heads, num_frames)[0].sum(dim=0)  # [N]
    temporal_weights -= temporal_weights.min(dim=-1, keepdim=True)[0]
    temporal_weights /= temporal_weights.max(dim=-1, keepdim=True)[0]
    temporal_weights = temporal_weights.cpu().numpy()
    attn_map = temporal_weights[:, None, None] * attn_map
    
    # # visualize the weights
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.bar(np.arange(num_frames) + 1, temporal_weights, 0.4)
    # plt.xlabel("video frames")
    # plt.ylabel("normalized attentions")
    # plt.xticks(np.arange(num_frames) + 1)
    # plt.tight_layout()
    # plt.savefig("../../_temp./temporal_weights.png")
    
    return attn_map