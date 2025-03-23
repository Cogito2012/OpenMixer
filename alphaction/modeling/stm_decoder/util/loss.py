# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
"""
import torch
import torch.nn.functional as F
from torch import nn
# from fvcore.nn import sigmoid_focal_loss_jit
from torchvision.ops import sigmoid_focal_loss

from . import box_ops
from .misc import (accuracy, get_world_size,is_dist_avail_and_initialized)
from .box_ops import generalized_box_iou

from scipy.optimize import linear_sum_assignment



class SetCriterion(nn.Module):
    """ This class computes the loss.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, eos_coef, losses, use_focal=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.FOCAL_LOSS.ALPHA
            self.focal_loss_gamma = cfg.MODEL.FOCAL_LOSS.GAMMA
        
        self.multi_label_action = cfg.MODEL.MULTI_LABEL_ACTION

        self.focal_scaling_gamma = cfg.MODEL.STM.FS_GAMMA

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # if self.use_focal:
        #     src_logits = src_logits.flatten(0, 1)
        #     # prepare one_hot target.
        #     target_classes = target_classes.flatten(0, 1)
        #     pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        #     labels = torch.zeros_like(src_logits)
        #     labels[pos_inds, target_classes[pos_inds]] = 1
        #     # comp focal loss.
        #     class_loss = sigmoid_focal_loss_jit(
        #         src_logits,
        #         labels,
        #         alpha=self.focal_loss_alpha,
        #         gamma=self.focal_loss_gamma,
        #         reduction="sum",
        #     ) / num_boxes
        #     losses = {'loss_ce': class_loss}
        # else:
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def action_cls_loss(self, output_action, targets, indices, idx):
        target_actions_o = torch.cat([t["action_labels"][J] for t, (_, J) in zip(targets, indices)])
        action = output_action[idx]
        if self.multi_label_action:
            act_loss = F.binary_cross_entropy_with_logits(action, target_actions_o)
        else:
            if self.focal_scaling_gamma > 0:
                ce_loss = F.cross_entropy(action, target_actions_o, reduction="none")
                p = F.softmax(action, dim=-1)
                y = torch.argmax(target_actions_o, dim=1)
                p_t = p[torch.arange(p.size(0), dtype=torch.long, device=p.device), y]
                act_loss = ce_loss * ((1 - p_t) ** self.focal_scaling_gamma)
                act_loss = act_loss.mean()
            else:
                act_loss = F.cross_entropy(action, target_actions_o)
        return act_loss


    def action_focal_cls_loss(self, output_action, targets, indices, idx):
        target_actions_o = torch.cat([t["action_labels"][J] for t, (_, J) in zip(targets, indices)])
        action = output_action[idx]
        # filter out the targets that do no have any samples in the batch
        pos_nz = (target_actions_o != 0).any(dim=0)
        act_focal_loss = sigmoid_focal_loss(action[:, pos_nz], target_actions_o[:, pos_nz], 
                                            alpha=self.focal_loss_alpha,
                                            gamma=self.focal_loss_gamma, reduction='mean')
        return act_focal_loss
    

    def entropy_loss(self, pred_entropy, idx):
        # minimize the average entropy
        return pred_entropy[idx].sum(dim=-1).mean()


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        idx = self._get_src_permutation_idx(indices)
        losses = {}
        if 'pred_actions' in outputs_without_aux:
            losses['loss_action'] = self.action_cls_loss(outputs_without_aux['pred_actions'], targets, indices, idx)
        # additionally use focal loss over batch dimension
        if self.use_focal:
            losses['loss_action_focal'] = self.action_focal_cls_loss(outputs_without_aux['pred_actions'], targets, indices, idx)
        
        if 'pred_entropy' in outputs:
            losses['loss_entropy'] = self.entropy_loss(outputs_without_aux['pred_entropy'], idx)

        if 'pred_chn' in outputs:
            losses['loss_chn'] = outputs_without_aux['pred_chn'][idx].sum(dim=-1).mean()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        # losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                idx = self._get_src_permutation_idx(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                
                if 'pred_actions' in aux_outputs:
                    losses['loss_action'+f'_{i}'] = self.action_cls_loss(aux_outputs['pred_actions'], targets, indices, idx)
                if self.use_focal:
                    losses['loss_action_focal'+f'_{i}'] = self.action_focal_cls_loss(aux_outputs['pred_actions'], targets, indices, idx)
                
                if 'pred_entropy' in aux_outputs:
                    losses['loss_entropy'+f'_{i}'] = self.entropy_loss(aux_outputs['pred_entropy'], idx)
                
                if 'pred_chn' in aux_outputs:
                    losses['loss_chn'+f'_{i}'] = aux_outputs['pred_chn'][idx].sum(dim=-1).mean()

        return losses



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes_xyxy"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]