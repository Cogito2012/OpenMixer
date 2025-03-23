import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone import build_backbone
from alphaction.modeling.stm_decoder.util.box_ops import clip_boxes_tensor
from torchvision.ops import roi_align
from einops import rearrange
import numpy as np



class NaiveBaseline(nn.Module):
    def __init__(self, cfg):
        super(NaiveBaseline, self).__init__()
        self.backbone = build_backbone(cfg)
        assert self.backbone.visual_encoder.use_cls_feat
        assert cfg.DATA.OPEN_VOCABULARY

        self.use_roi_feat = cfg.MODEL.USE_ROI_FEAT
        self.multi_label_action = cfg.MODEL.MULTI_LABEL_ACTION
    
    
    def roi_align_pool(self, patch_feats, batch_boxes, raw_sizes, out_size=(7, 7), spatial_scale=1.0/16):
        """ patch_feats: (B, D, T, h, w)
            boxes: list of boxes, not normalized
            raw_sizes: (B, 2) in (width, height)
        """
        B, D, T, h, w = patch_feats.size()
        device = patch_feats.device
        feat_maps = patch_feats.mean(dim=2)  # (B, D, h, w)  temporally mean pooling
        boxes_list = [np.hstack([np.ones((boxes.shape[0], 1)) * i, boxes]) for i, boxes in enumerate(batch_boxes)]
        boxes_tensor = torch.from_numpy(np.vstack(boxes_list)).type(patch_feats.dtype).to(device)
        roi_feat = roi_align(feat_maps, boxes_tensor, out_size, spatial_scale)  # (BN, D, 7, 7)
        roi_feat = rearrange(roi_feat, 'm d h w -> m (h w) d')
        
        # get meanpooled roi features
        roi_align_features = []
        batch_indices = boxes_tensor[:, 0].long()
        for i in range(B):
            rois = roi_feat[batch_indices == i].mean(dim=1)  # (n, d)
            roi_align_features.append(rois)
        
        return roi_align_features

    
    def forward(self, slow_video, fast_video, whwh, boxes=None, labels=None, extras={}, part_forward=-1):

        assert not self.training, "NaiveBaseline does not need training!"
        assert 'prior_boxes' in extras, "NaiveBaseline use loaded boxes for testing!"
        device = slow_video.device

        prior_boxes = extras['prior_boxes']
        box_list = []
        for i in range(len(prior_boxes)):
            box = torch.tensor(prior_boxes[i], dtype=torch.float32, device=device)
            cur_whwh = whwh[i]
            box = clip_boxes_tensor(box, cur_whwh[1], cur_whwh[0])
            box[:, 0::2] /= cur_whwh[0]
            box[:, 1::2] /= cur_whwh[1]
            box_list.append(box)

        if self.backbone.num_pathways == 1:
            features = self.backbone([slow_video])
        else:
            features = self.backbone([slow_video, fast_video])
        
        patch_feats, cls_feat_visual = features  # (B, 512)
        B = cls_feat_visual.size(0)

        if self.use_roi_feat:
            # feature projection & RoIAlign Pooling
            patch_feats = self.backbone.visual_encoder.project_patch_features(patch_feats[0])
            roi_features = self.roi_align_pool(patch_feats, prior_boxes, whwh[:, :2])

        # get the current text feature embeddings
        text_features = self.backbone.forward_text(device=slow_video.device)  # (K, 512)
        tau_inv = self.backbone.tau_inv

        if isinstance(text_features, list):
            text_features = torch.stack(text_features).mean(1)
        text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)  # (K, D)

        action_score_list = []
        if self.use_roi_feat:
            # return self.forward_roi_cls(roi_features, text_features, tau_inv, whwh)
            for roi_feat in roi_features:
                # action recognition
                vis_features_normed = roi_feat / roi_feat.norm(dim=-1, keepdim=True)  # (N, D)
                action_score = tau_inv * vis_features_normed @ text_features_normed.t()  # (N, K)
                action_score_list.append(action_score)
        else:
            vis_features_normed = cls_feat_visual / cls_feat_visual.norm(dim=-1, keepdim=True)  # (B, D)
            action_score = tau_inv * vis_features_normed @ text_features_normed.t()  # (B, K)
            for i in range(B):
                # with full frame input, we only have one score vector, which need to be repeated.
                scores = action_score[[i]].repeat(box_list[i].size(0), 1)  # (1, K)
                action_score_list.append(scores)
        
        return action_score_list, box_list


def build_naive_baseline(cfg):
    return NaiveBaseline(cfg)