import torch
import torch.nn as nn
import torch.nn.functional as F
from .util.box_ops import bbox_overlaps, box_cxcywh_to_xyxy, box_xyxy_to_xyzr, clip_boxes_tensor, box_sampling_from_heatmap, box_sampling_from_prior
from .util.msaq import SAMPLE4D
from .util.adaptive_mixing_operator import AdaptiveMixing
from .util.head_utils import _get_activation_layer, bias_init_with_prob, decode_box, position_embedding, make_sample_points, refine_xyzr
from .util.head_utils import FFN, MultiheadAttention
from .util.loss import SetCriterion, HungarianMatcher



class AdaptiveSTSamplingMixing(nn.Module):

    def __init__(self, spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 query_dim=256,
                 feat_channels=None,
                 pretrained_action=False):
        super(AdaptiveSTSamplingMixing, self).__init__()
        self.spatial_points =  spatial_points
        self.temporal_points = temporal_points
        self.out_multiplier = out_multiplier
        self.n_groups = n_groups
        self.query_dim = query_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.query_dim
        self.offset_generator = nn.Sequential(nn.Linear(query_dim, spatial_points * n_groups * 3))
        self.pretrained_action = pretrained_action

        self.norm_s = nn.LayerNorm(query_dim)
        self.adaptive_mixing_s = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.query_dim,
            in_points=self.spatial_points,
            out_points=self.spatial_points*self.out_multiplier,
            n_groups=self.n_groups,
        )

        if not self.pretrained_action:
            self.norm_t = nn.LayerNorm(query_dim)
            self.adaptive_mixing_t = AdaptiveMixing(
                self.feat_channels,
                query_dim=self.query_dim,
                in_points=self.temporal_points,
                out_points=self.temporal_points*self.out_multiplier,
                n_groups=self.n_groups,
            )

        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.offset_generator[-1].weight.data)
        nn.init.zeros_(self.offset_generator[-1].bias.data)

        bias = self.offset_generator[-1].bias.data.view(
            self.n_groups, self.spatial_points, 3)

        if int(self.spatial_points ** 0.5) ** 2 == self.spatial_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
            # 格子采样
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)
        bias[:, :, 2:3].mul_(0.0)

        self.adaptive_mixing_s._init_weights()
        if not self.pretrained_action:
            self.adaptive_mixing_t._init_weights()

    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides):

        offset = self.offset_generator(spatial_queries)
        sample_points_xy = make_sample_points(offset, self.n_groups * self.spatial_points, proposal_boxes)
        sampled_feature, _ = SAMPLE4D(sample_points_xy, features, featmap_strides=featmap_strides, n_points=self.spatial_points)

        # B, C, n_groups, temporal_points, spatial_points, n_query, _ = sampled_feature.size()
        sampled_feature = sampled_feature.flatten(5, 6)                   # B, n_channels, n_groups, temporal_points, spatial_points, n_query
        sampled_feature = sampled_feature.permute(0, 5, 2, 3, 4, 1)       # B, n_query, n_groups, temporal_points, spatial_points, n_channels

        spatial_feats = torch.mean(sampled_feature, dim=3)                            # out_s has shape [B, n_query, n_groups, spatial_points, n_channels]
        spatial_queries = self.adaptive_mixing_s(spatial_feats, spatial_queries)
        spatial_queries = self.norm_s(spatial_queries)

        if temporal_queries is not None:
            temporal_feats = torch.mean(sampled_feature, dim=4)                        # out_t has shape [B, n_query, n_groups, temporal_points, n_channels]
            temporal_queries = self.adaptive_mixing_t(temporal_feats, temporal_queries)
            temporal_queries = self.norm_t(temporal_queries)

        return spatial_queries, temporal_queries


class AMStage(nn.Module):

    def __init__(self, query_dim=256,
                 feat_channels=256,
                 num_heads=8,
                 feedforward_channels=2048,
                 dropout=0.0,
                 num_ffn_fcs=2,
                 ffn_act='RelU',
                 spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 num_action_fcs=1,
                 num_classes_object=1,
                 num_classes_action=80,
                 open_vocabulary=False,
                 text_dim=768,
                 cond_cls=False,
                 fuse_cls=False,
                 fuse_factor=-1,
                 fuse_method='logit_fusion',
                 pretrained_action=False,
                 num_queries=100,
                 dest=False,
                 ):


        super(AMStage, self).__init__()

        # MHSA-S
        ffn_act_cfg = dict(type=ffn_act, inplace=True)
        self.attention_s = MultiheadAttention(query_dim, num_heads, dropout)
        self.attention_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.ffn_s = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.iof_tau = nn.Parameter(torch.ones(self.attention_s.num_heads,))
        self.cond_cls = cond_cls
        self.fuse_cls = fuse_cls
        self.fuse_method = fuse_method
        self.dest = dest

        if not pretrained_action:
            # MHSA-T
            self.attention_t = MultiheadAttention(query_dim, num_heads, dropout)
            self.attention_norm_t = nn.LayerNorm(query_dim, eps=1e-5)
            self.ffn_t = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
            self.ffn_norm_t = nn.LayerNorm(query_dim, eps=1e-5)

        self.samplingmixing = AdaptiveSTSamplingMixing(
            spatial_points=spatial_points,
            temporal_points=temporal_points,
            out_multiplier=out_multiplier,
            n_groups=n_groups,
            query_dim=query_dim,
            feat_channels=feat_channels,
            pretrained_action=pretrained_action,
        )

        cls_feature_dim = query_dim
        reg_feature_dim = query_dim
        action_feat_dim = query_dim * 2 if not self.dest else query_dim

        # human classifier
        self.human_cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.human_cls_fcs.append(
                nn.Linear(cls_feature_dim, cls_feature_dim, bias=True))
            self.human_cls_fcs.append(
                nn.LayerNorm(cls_feature_dim, eps=1e-5))
            self.human_cls_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.human_fc_cls = nn.Linear(cls_feature_dim, num_classes_object + 1)

        # human bbox regressor
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(reg_feature_dim, reg_feature_dim, bias=True))
            self.reg_fcs.append(
                nn.LayerNorm(reg_feature_dim, eps=1e-5))
            self.reg_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.fc_reg = nn.Linear(reg_feature_dim, 4)

        # action classifier
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.open_vocabulary = open_vocabulary
        if not open_vocabulary:
            self.action_cls_fcs = nn.ModuleList()
            for _ in range(num_action_fcs):
                self.action_cls_fcs.append(
                    nn.Linear(action_feat_dim, action_feat_dim, bias=True))
                self.action_cls_fcs.append(
                    nn.LayerNorm(action_feat_dim, eps=1e-5))
                self.action_cls_fcs.append(
                    _get_activation_layer(ffn_act_cfg))
            self.fc_action = nn.Linear(action_feat_dim, num_classes_action)
        else:
            if not pretrained_action:
                self.linear_proj = nn.Linear(action_feat_dim, text_dim, bias=False) if action_feat_dim != text_dim else nn.Identity()
        if self.fuse_cls:
            if self.fuse_method == 'logit_fusion':
                self.logit_alpha = nn.Parameter(torch.ones(num_queries,) * fuse_factor)
        
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                pass
        bias_init = bias_init_with_prob(0.01)
        if hasattr(self, 'fc_action'):
            nn.init.constant_(self.fc_action.bias, bias_init)
        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        self.samplingmixing.init_weights()


    def compute_fused_logits(self, vis_feat, text_feat, cls_feat, labels=None, scale=None, factor=0.1):
        """ vis_feat: (B, N, D)
            text_feat: (K, D)
            cls_feat: (B, D)
        """
        # cosine angle of learned features
        vis_normed = vis_feat / vis_feat.norm(dim=-1, keepdim=True)
        text_mean = torch.stack(text_feat, dim=0).mean(dim=1) if isinstance(text_feat, list) else text_feat

        text_normed = text_mean / text_mean.norm(dim=-1, keepdim=True)
        cos_learned = vis_normed @ text_normed.transpose(-1, -2)  # (B, N, K)

        # cosine angle of pretrained features
        vis_pretrained = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
        cos_fixed = vis_pretrained @ text_normed.transpose(-1, -2)  # (B, K)
        # logits fusion with learnable coefficients
        alpha = self.logit_alpha.view(1, -1, 1)  # (1, N, 1)
        if len(cos_fixed.size()) == 2:
            cos_fixed = cos_fixed.unsqueeze(1)
        logits = alpha * cos_fixed + (1 - alpha) * cos_learned
        return logits


    def learned_action_head(self, action_feat, text_features=None, tau_inv=100, pretrained_cls=None, labels=None):
        N, n_query = action_feat.shape[:2]
        # linearly project visual feature into text embedding space
        action_feat_proj = self.linear_proj(action_feat)  # 512 --> 768
        if self.fuse_cls:
            if self.fuse_method == 'logit_fusion':
                logits = self.compute_fused_logits(action_feat_proj, text_features, pretrained_cls, labels=labels, scale=tau_inv)
                action_score = logits * tau_inv
                return action_score
            else:
                raise NotImplementedError    

        # compute cosine similarity, value in [-1, 1]
        text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)  # (K, D=768)
        vis_features_normed = action_feat_proj / action_feat_proj.norm(dim=-1, keepdim=True)  # (N, n_query, D)
        action_score = (
            tau_inv * vis_features_normed @ text_features_normed.transpose(-1, -2)
        ).view(N, n_query, -1)  # (N, n_query, K)
        return action_score


    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries=None, 
                featmap_strides=[4, 8, 16, 32], text_features=None, tau_inv=100, vis_cls_feat=None, patch_feat=None, text_token_feats=None, labels=None, cond=None):

        N, n_query = spatial_queries.shape[:2]

        with torch.no_grad():
            rois = decode_box(proposal_boxes)
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(proposal_boxes, spatial_queries.size(-1) // 4)

        # IoF
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)             # N*num_heads, n_query, n_query
        pe = pe.permute(1, 0, 2)                                                     # n_query, N, content_dim

        # sinusoidal positional embedding
        spatial_queries = spatial_queries.permute(1, 0, 2)  # n_query, N, content_dim
        spatial_queries_attn = spatial_queries + pe
        spatial_queries = self.attention_s(spatial_queries_attn, attn_mask=attn_bias,)
        spatial_queries = self.attention_norm_s(spatial_queries)
        spatial_queries = spatial_queries.permute(1, 0, 2)
        # N, n_query, content_dim

        if temporal_queries is not None:
            temporal_queries = temporal_queries.permute(1, 0, 2)
            temporal_queries_attn = temporal_queries + pe
            temporal_queries = self.attention_t(temporal_queries_attn, attn_mask=attn_bias,)
            temporal_queries = self.attention_norm_t(temporal_queries)
            temporal_queries = temporal_queries.permute(1, 0, 2)
        # N, n_query, content_dim

        if self.cond_cls:
            if temporal_queries is not None:
                temporal_queries = temporal_queries + cond.unsqueeze(1)

        spatial_queries, temporal_queries = \
            self.samplingmixing(features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides)

        spatial_queries = self.ffn_s(spatial_queries)
        if temporal_queries is not None:
            temporal_queries = self.ffn_t(temporal_queries)

        # layer normalization before heads
        spatial_queries = self.ffn_norm_s(spatial_queries)
        if temporal_queries is not None:
            temporal_queries = self.ffn_norm_t(temporal_queries)

        ################################### heads ###################################
        # objectness head
        cls_feat = spatial_queries
        for cls_layer in self.human_cls_fcs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.human_fc_cls(cls_feat).view(N, n_query, -1)

        # regression head
        reg_feat = spatial_queries
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        xyzr_delta = self.fc_reg(reg_feat).view(N, n_query, -1)

        # action head
        if self.open_vocabulary:
            assert text_features is not None
            if temporal_queries is not None:
                action_feat = torch.cat([spatial_queries, temporal_queries], dim=-1) if not self.dest else temporal_queries
                pretrained_cls = vis_cls_feat if self.fuse_cls else None
                action_score = self.learned_action_head(action_feat, text_features, tau_inv, pretrained_cls=pretrained_cls, labels=labels)  # (N, n_query, K)
            else:
                # compute cosine similarity, value in [-1, 1]
                text_features = torch.stack(text_features, dim=0).mean(dim=1) if isinstance(text_features, list) else text_features  # (K, D)
                text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)  # (K, D=768)
                vis_features_normed = vis_cls_feat / vis_cls_feat.norm(dim=-1, keepdim=True)  # (N, D)
                action_score = (
                    tau_inv * vis_features_normed @ text_features_normed.transpose(-1, -2)
                )
                action_score = action_score.unsqueeze(1).repeat(1, n_query, 1)
        else:
            action_feat = torch.cat([spatial_queries, temporal_queries], dim=-1)
            for act_layer in self.action_cls_fcs:
                action_feat = act_layer(action_feat)
            action_score = self.fc_action(action_feat).view(N, n_query, -1)

        spatial_queries = spatial_queries.view(N, n_query, -1)
        if temporal_queries is not None:
            temporal_queries = temporal_queries.view(N, n_query, -1)
        
        return cls_score, action_score, xyzr_delta, spatial_queries, temporal_queries



class STMDecoder(nn.Module):

    def __init__(self, cfg):

        super(STMDecoder, self).__init__()
        self.device = torch.device('cuda')

        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY
        self.multi_label_action = cfg.MODEL.MULTI_LABEL_ACTION
        self.use_pretrained_action = cfg.MODEL.STM.PRETRAIN_ACTION
        if self.use_pretrained_action:
            assert self.open_vocabulary
        
        self._generate_queries(cfg)

        self.cam_sampling = eval('cfg.MODEL.{}.CAM_SAMPLING'.format(cfg.MODEL.TEXT_ENCODER))
        self.num_heads = cfg.MODEL.STM.NUM_HEADS

        self.num_stages = cfg.MODEL.STM.NUM_STAGES
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            decoder_stage = AMStage(
                query_dim=cfg.MODEL.STM.HIDDEN_DIM,
                feat_channels=cfg.MODEL.STM.HIDDEN_DIM,
                num_heads=cfg.MODEL.STM.NUM_HEADS,
                feedforward_channels=cfg.MODEL.STM.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.STM.DROPOUT,
                num_ffn_fcs=cfg.MODEL.STM.NUM_FCS,
                ffn_act=cfg.MODEL.STM.ACTIVATION,
                spatial_points=cfg.MODEL.STM.SPATIAL_POINTS,
                temporal_points=cfg.MODEL.STM.TEMPORAL_POINTS,
                out_multiplier=cfg.MODEL.STM.OUT_MULTIPLIER,
                n_groups=cfg.MODEL.STM.N_GROUPS,
                num_cls_fcs=cfg.MODEL.STM.NUM_CLS,
                num_reg_fcs=cfg.MODEL.STM.NUM_REG,
                num_action_fcs=cfg.MODEL.STM.NUM_ACT,
                num_classes_object=cfg.MODEL.STM.OBJECT_CLASSES,
                num_classes_action=cfg.MODEL.STM.ACTION_CLASSES,
                open_vocabulary=cfg.DATA.OPEN_VOCABULARY,
                text_dim=eval('cfg.MODEL.{}.EMBED_DIM'.format(cfg.MODEL.TEXT_ENCODER)),
                cond_cls=cfg.MODEL.STM.COND_CLS,
                fuse_cls=cfg.MODEL.STM.FUSE_CLS,
                fuse_factor=cfg.MODEL.STM.FUSE_FACTOR,
                fuse_method=cfg.MODEL.STM.FUSE_METHOD,
                pretrained_action=self.use_pretrained_action,
                num_queries=self.num_queries,
                dest=cfg.MODEL.STM.DeST,
                )
            self.decoder_stages.append(decoder_stage)

        object_weight = cfg.MODEL.STM.OBJECT_WEIGHT
        giou_weight = cfg.MODEL.STM.GIOU_WEIGHT
        l1_weight = cfg.MODEL.STM.L1_WEIGHT
        action_weight = cfg.MODEL.STM.ACTION_WEIGHT
        background_weight = cfg.MODEL.STM.BACKGROUND_WEIGHT
        action_focal_weight = cfg.MODEL.STM.FOCAL_WEIGHT
        use_focal = action_focal_weight > 0
        self.weight_dict = {"loss_ce": object_weight,
                            "loss_bbox": l1_weight,
                            "loss_giou": giou_weight}
        if use_focal:
            self.weight_dict.update({"loss_action_focal": action_focal_weight})
        if not self.use_pretrained_action:
            self.weight_dict.update({"loss_action": action_weight})
        
        self.person_threshold = cfg.MODEL.STM.PERSON_THRESHOLD
        self.cond_cls = cfg.MODEL.STM.COND_CLS
        self.fuse_cls = cfg.MODEL.STM.FUSE_CLS
        self.cond_type = cfg.MODEL.STM.COND_MODALITY

        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=object_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight)

        self.intermediate_supervision = cfg.MODEL.STM.INTERMEDIATE_SUPERVISION
        if self.intermediate_supervision:
            for i in range(self.num_stages - 1):
                inter_weight_dict = {k + f"_{i}": v for k, v in self.weight_dict.items()}
                self.weight_dict.update(inter_weight_dict)


        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=cfg.MODEL.STM.OBJECT_CLASSES,
                                      matcher=matcher,
                                      eos_coef=background_weight,
                                      losses=losses,
                                      use_focal=use_focal)

    def _generate_queries(self, cfg):
        self.num_queries = cfg.MODEL.STM.NUM_QUERIES
        self.hidden_dim = cfg.MODEL.STM.HIDDEN_DIM

        # Build Proposals
        self.init_spatial_queries = nn.Embedding(self.num_queries, self.hidden_dim)
        if not self.use_pretrained_action:
            self.init_temporal_queries = nn.Embedding(self.num_queries, self.hidden_dim)

    def _box_init(self, whwh, extras={}):
        num_queries = self.num_queries
        batch_size = len(whwh)

        if 'prior_boxes' in extras:
            # initialize the proposals by ground-truth (for showing the upper-bound performance)
            proposals = [box_sampling_from_prior(extras['prior_boxes'][b], num_boxes=num_queries, device=whwh.device)
                         for b in range(batch_size)]
            proposals = torch.stack(proposals, dim=0)  # B, N, 4
        elif 'cams' in extras:
            # proposals are initialized by sampling on cams
            if self.cam_sampling == 'topk':
                prior_map = extras.get('prior_map', None)
                proposals = [box_sampling_from_heatmap(extras['cams'][b], prior_map=prior_map, num_boxes=num_queries) 
                            for b in range(batch_size)]
            else:
                raise NotImplementedError
            proposals = torch.stack(proposals, dim=0)  # B, N, 4
        else:
            # proposals are initialize from [0, 0, W, H]
            proposals = torch.ones(num_queries, 4, dtype=torch.float, device=self.device, requires_grad=False)
            proposals[:, :2] = 0.5
            proposals = box_cxcywh_to_xyxy(proposals)  # (0, 0, 1, 1)

            whwh = whwh[:, None, :] # B, 1, 4
            proposals = proposals[None] * whwh # B, N, 4

        xyzr = box_xyxy_to_xyzr(proposals)
        xyzr = xyzr.detach()

        return xyzr


    def _decode_init_queries(self, whwh, cond=None, extras={}):
        
        batch_size = len(whwh)
        # initialize the box queries
        xyzr = self._box_init(whwh, extras)

        init_spatial_queries = self.init_spatial_queries.weight.clone()
        init_spatial_queries = init_spatial_queries[None].expand(batch_size, *init_spatial_queries.size())

        init_temporal_queries = None
        if not self.use_pretrained_action:
            init_temporal_queries = self.init_temporal_queries.weight.clone()
            init_temporal_queries = init_temporal_queries[None].expand(batch_size, *init_temporal_queries.size())
            
        if self.cond_cls:
            assert self.hidden_dim == cond.size(-1), \
                "cfg.MODEL.STM.HIDDEN_DIM should be set to {} when conditioning on pretrained CLS visual feature".format(cond.size(-1))
            if not self.use_pretrained_action:
                init_temporal_queries = cond.unsqueeze(1) + init_temporal_queries

        # normalization over feature dimension
        init_spatial_queries = torch.layer_norm(init_spatial_queries,
                                                normalized_shape=[init_spatial_queries.size(-1)])
        if not self.use_pretrained_action:
            init_temporal_queries = torch.layer_norm(init_temporal_queries,
                                                        normalized_shape=[init_temporal_queries.size(-1)])
    
        return xyzr, init_spatial_queries, init_temporal_queries



    def person_detector_loss(self, outputs_class, outputs_coord, criterion, targets, outputs_actions, output_entropy=[], output_chn=[]):
        if self.intermediate_supervision:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 
                      'aux_outputs': [{'pred_logits': a, 'pred_boxes': b}
                                      for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]}

            if not self.use_pretrained_action:
                output['pred_actions'] = outputs_actions[-1]
                for i, c in enumerate(outputs_actions[:-1]):
                    output['aux_outputs'][i]['pred_actions'] = c
            
            if len(output_entropy) > 0:
                output['pred_entropy'] = output_entropy[-1]
                for i, c in enumerate(output_entropy[:-1]):
                    output['aux_outputs'][i]['pred_entropy'] = c
            
            if len(output_chn) > 0:
                output['pred_chn'] = output_chn[-1]
                for i, c in enumerate(output_chn[:-1]):
                    output['aux_outputs'][i]['pred_chn'] = c

        else:
            raise NotImplementedError

        loss_dict = criterion(output, targets)
        return loss_dict


    def make_targets(self, gt_boxes, whwh, labels):
        targets = []
        for box_in_clip, frame_size, label in zip(gt_boxes, whwh, labels):
            target = {}
            if not self.use_pretrained_action:
                target['action_labels'] = torch.tensor(label, dtype=torch.float32, device=self.device)
            target['boxes_xyxy'] = torch.tensor(box_in_clip, dtype=torch.float32, device=self.device)
            # num_box, 4 (x1,y1,x2,y2) w.r.t augmented images unnormed
            target['labels'] = torch.zeros(len(target['boxes_xyxy']), dtype=torch.int64, device=self.device)
            target["image_size_xyxy"] = frame_size.to(self.device)
            # (4,) whwh
            image_size_xyxy_tgt = frame_size.unsqueeze(0).repeat(len(target['boxes_xyxy']), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)

            # (num_box, 4) whwh
            targets.append(target)

        return targets


    def get_prematched_text(self, vis_cls_feat, text_features, labels=None):
        text_feat = torch.stack(text_features, dim=0).mean(dim=1) if isinstance(text_features, list) else text_features  # (K, D)
        if labels is not None:
            classes = torch.tensor([int(torch.tensor(onehots).argmax(dim=1)[0]) for onehots in labels], device=vis_cls_feat.device).long()  # (B,)
            text_prematched = text_feat[classes]  # (B, D)
        else:
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            vis_feat = vis_cls_feat / vis_cls_feat.norm(dim=-1, keepdim=True)
            pred_cls = (vis_feat @ text_feat.t()).argmax(dim=-1)  # (B,)
            text_prematched = text_feat[pred_cls] # (B, D)
        return text_prematched


    def get_prematched_token_text(self, text_token_feats, text_features, pretrained_cls, labels=None):
        """ text_token_feats: (K, L, D)
            text_features: (K, D)
            pretrained_cls: (B, D)
        """
        text_feat = torch.stack(text_features, dim=0).mean(dim=1) if isinstance(text_features, list) else text_features  # (K, D)
        if labels is not None:
            classes = torch.tensor([int(torch.tensor(onehots).argmax(dim=1)[0]) for onehots in labels], device=pretrained_cls.device).long()  # (B,)
        else:
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            vis_feat = pretrained_cls / pretrained_cls.norm(dim=-1, keepdim=True)
            classes = (vis_feat @ text_feat.t()).argmax(dim=-1)  # (B,)
        
        text_prematched = text_token_feats['feat'][classes]  # (B, L, D)
        token_mask = text_token_feats['mask'][classes]  # (B, L)

        return text_prematched, token_mask
    

    def _expand_mask(self, token_mask, num_heads, tgt_len, dtype=torch.float32):
        """ token_mask: (B, L) where 1 indicates attend, and 0 indicates NOT attend
            return: (B * Nh, N, L) where N is the number of queries
        """
        bsz, src_len = token_mask.size()  # (B, L)
        expanded_mask = token_mask[:, None, None, :].expand(bsz, num_heads, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        return inverted_mask.view(-1, tgt_len, src_len)


    def forward(self, features, whwh, gt_boxes=None, labels=None, extras={}, part_forward=-1, text_features=None, tau_inv=100, cls_feat=None, patch_feat=None, text_token_feats=None):

        cond = None
        if self.cond_cls:
            cond = cls_feat if self.cond_type == 'visual' else self.get_prematched_text(cls_feat, text_features, labels)

        proposal_boxes, spatial_queries, temporal_queries = self._decode_init_queries(whwh, cond=cond, extras=extras)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_action_logits = []
        inter_entropy_outs = []
        inter_chn_loss = []
        B, N, _ = spatial_queries.size()

        for decoder_stage in self.decoder_stages:
            objectness_score, action_score, delta_xyzr, spatial_queries, temporal_queries = \
                decoder_stage(features, proposal_boxes, spatial_queries, temporal_queries, text_features=text_features, tau_inv=tau_inv, vis_cls_feat=cls_feat, patch_feat=patch_feat, text_token_feats=text_token_feats, labels=labels, cond=cond)
            proposal_boxes, pred_boxes = refine_xyzr(proposal_boxes, delta_xyzr)

            inter_class_logits.append(objectness_score)
            inter_pred_bboxes.append(pred_boxes)
            inter_action_logits.append(action_score)

        if not self.training:
            action_scores = inter_action_logits[-1]  # leave the logits_to_prob transform into evaluation
            scores = F.softmax(inter_class_logits[-1], dim=-1)[:, :, 0]
            # scores: B*100
            action_score_list = []
            box_list = []
            for i in range(B):
                selected_idx = scores[i] >= self.person_threshold
                if not any(selected_idx):
                    _,selected_idx = torch.topk(scores[i],k=3,dim=-1)

                action_score = action_scores[i][selected_idx]
                box = inter_pred_bboxes[-1][i][selected_idx]
                cur_whwh = whwh[i]
                box = clip_boxes_tensor(box, cur_whwh[1], cur_whwh[0])
                box[:, 0::2] /= cur_whwh[0]
                box[:, 1::2] /= cur_whwh[1]
                action_score_list.append(action_score)
                box_list.append(box)
            return action_score_list, box_list

        targets = self.make_targets(gt_boxes, whwh, labels)
        losses = self.person_detector_loss(inter_class_logits, inter_pred_bboxes, self.criterion, targets, inter_action_logits, output_entropy=inter_entropy_outs, output_chn=inter_chn_loss)
        weight_dict = self.weight_dict
        for k in losses.keys():
            if k in weight_dict:
                losses[k] *= weight_dict[k]
        return losses

def build_stm_decoder(cfg):
    return STMDecoder(cfg)
