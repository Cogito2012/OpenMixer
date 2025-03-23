import os
from pprint import pformat
from ..pascal_wrapper import frame_mAP_pascal
from ..evaluate_map import frameAP, videoAP, prediction_to_tubes
import numpy as np
import tempfile, csv
from collections import defaultdict
import copy
from ..pascal_evaluation.np_box_ops import iou

import torch
import torch.nn.functional as F



def prepare_for_frame_ap(predictions, dataset, num_classes, score_thresh=0.0):
    _results, _targets = {}, {}
    gt_info = defaultdict(list)
    # get ground truth
    for vid, label, i, frame_id, box in dataset.samples_list:
        image_key = make_image_key(vid, frame_id)
        gt_boxes = np.array(box[None])  # (1, 4), box coordinates in raw resolution
        
        # coordinate normalization
        raw_resolution = dataset.data['resolution'][vid]  # (H, W)
        gt_boxes[:, [0, 2]] /= float(raw_resolution[1])  # Width
        gt_boxes[:, [1, 3]] /= float(raw_resolution[0])  # Height

        # label transformation
        gt_labels = dataset._labels_transform(label, num_classes, 1, onehot=False, training=False)  # (1,), onehot
        gt_labels = [l + 1 for l in gt_labels]  # evaluator accept labels starting from 1
        _targets[image_key] = {
            "bbox": gt_boxes,
            "labels": gt_labels,
            'resolution': raw_resolution
        }
        gt_info[vid].append(int(frame_id))
    gt_info = {vid: sorted(frame_list) for vid, frame_list in gt_info.items()}

    # get detections
    for sample_id, prediction in enumerate(predictions):
        # dict(vid=vid, frame_id=frame_id, box=box, label=label, resolution=raw_resolution)
        info = dataset.get_video_info(sample_id)
        if len(prediction) == 0:
            continue
        
        # get the prediction
        boxes = prediction[0].numpy()  # (N, 4), normalized coordinates w.r.t. input resolution
        scores = torch.sigmoid(prediction[1]) if dataset.multilabel_action else F.softmax(prediction[1], dim=-1)
        scores = scores.numpy()  # (N, K)  # No background class.
        # filtering with score threshold
        box_ids, action_ids = np.where(scores >= score_thresh)
        boxes = boxes[box_ids, :]
        scores = scores[box_ids, action_ids]

        # the only case when this is True, is that open_vocabulary=True & EAL_OPEN=False
        if dataset.open_vocabulary and (not dataset.eval_open):
            action_ids = np.array([dataset.closed_to_open[id] for id in action_ids])

        image_key = make_image_key(info['vid'], info['frame_id'])
        _results[image_key] = {
            "boxes": boxes,
            "scores": scores,
            "action_ids": action_ids
        }
    return _results, _targets, gt_info


def make_image_key(video_id, frame_id):
    """Returns a unique identifier for a video id & frame_id."""
    return "%s,%05d" % (video_id, float(frame_id))


def prepare_for_video_ap(predictions, dataset, score_thresh=0.0):
    # get detections
    _results = {}
    for sample_id, prediction in enumerate(predictions):
        # dict(vid=vid, frame_id=frame_id, box=box, label=label, resolution=raw_resolution)
        info = dataset.get_video_info(sample_id)
        if len(prediction) == 0:
            continue
        
        # get the prediction
        boxes = prediction[0].numpy()  # (N, 4), normalized coordinates w.r.t. input resolution
        # boxes to raw resolution
        boxes[:, [0, 2]] *= info['resolution'][1]  # x1, x2
        boxes[:, [1, 3]] *= info['resolution'][0]  # y1, y2
        # scores
        scores = torch.sigmoid(prediction[1]) if dataset.multilabel_action else F.softmax(prediction[1], dim=-1)
        scores = scores.numpy()  # (N, K) 
        # filtering with score threshold
        box_ids, action_ids = np.where(scores >= score_thresh)
        boxes = boxes[box_ids, :]
        scores = scores[box_ids, action_ids]

        # the only case when this is True, is that open_vocabulary=True & EAL_OPEN=False
        if dataset.open_vocabulary and (not dataset.eval_open):
            action_ids = np.array([dataset.closed_to_open[id] for id in action_ids])

        image_key = make_image_key(info['vid'], info['frame_id'])
        _results[image_key] = {
            "boxes": boxes,
            "scores": scores,
            "action_ids": action_ids
        }
    return _results


def logits_to_scores(logits, multilabel=False):
    """ logits: (B, K)
    """    
    if multilabel:
        return torch.sigmoid(logits) 
    else:
        return F.softmax(logits, dim=-1)


def prepare_preds_ind(predictions, dataset, score_thresh=0.0, cls_offset=0, normalize=False):
    # get detections
    ids_known = list(dataset.open_to_closed.keys())
    ids_unknown = list(dataset.open_to_unseen.keys())
    
    _results_known, _results_unknown = {}, {}
    for sample_id, prediction in enumerate(predictions):
        # dict(vid=vid, frame_id=frame_id, box=box, label=label, resolution=raw_resolution)
        info = dataset.get_video_info(sample_id)
        if len(prediction) == 0:
            continue
        image_key = make_image_key(info['vid'], info['frame_id'])
        
        # get the prediction
        boxes = prediction[0].numpy()  # (N, 4), normalized coordinates w.r.t. input resolution
        scores = prediction[1]

        if dataset.prior_boxes_test:
            ### method 3 #####
            boxes_prior, scores_prior = dataset.dets_data[sample_id]  # (M, 4), (M, Ck+Cu)
            boxes_prior = boxes_prior.numpy()
            iou_mat = iou(boxes, boxes_prior)  # (N, M)
            maxids = np.argmax(iou_mat, axis=1)  # (N,)
            pred_cls = scores.argmax(dim=-1)  # (N,)
            for i, c in enumerate(pred_cls):
                bid = maxids[i]
                pri_c = scores_prior[bid].argmax(dim=-1)
                ## replacing the box and scores if predictions of ours and gdino/maskrcnn are not consistent!
                if (c in ids_known and pri_c in ids_unknown) or \
                   (c in ids_unknown and pri_c in ids_known):
                    boxes[i] = boxes_prior[bid]
                    scores[i] = scores_prior[bid]

        if not normalize:
            # boxes to raw resolution
            boxes[:, [0, 2]] *= info['resolution'][1]  # x1, x2
            boxes[:, [1, 3]] *= info['resolution'][0]  # y1, y2
        
        if info['vid'] in dataset.data_known['videos']:
            scores_known = scores[:, ids_known]  # logits, (N, K) classes defined in open-world K=21
            # logits to scores
            scores_known = logits_to_scores(scores_known, multilabel=dataset.multilabel_action)
            scores_known = scores_known.numpy()  # (N, K) 
            
            # filtering with score threshold (known)
            box_ids, action_ids = np.where(scores_known >= score_thresh)
            _results_known[image_key] = {
                "boxes": boxes[box_ids, :],
                "scores": scores_known[box_ids, action_ids],
                "action_ids": action_ids + cls_offset
            }
        
        if info['vid'] in dataset.data_unknown['videos']:
            scores_unknown = scores[:, ids_unknown]
            scores_unknown = logits_to_scores(scores_unknown, multilabel=dataset.multilabel_action)
            scores_unknown = scores_unknown.numpy()  # (N, K) 
            
            # filtering with score threshold (unknown)
            box_ids, action_ids = np.where(scores_unknown >= score_thresh)
            _results_unknown[image_key] = {
                "boxes": boxes[box_ids, :],
                "scores": scores_unknown[box_ids, action_ids],
                "action_ids": action_ids + cls_offset
            }

    return _results_known, _results_unknown



def _prepare_gts_ind(dataset):
    _targets_known, _targets_unknown = {}, {}
    # get ground truth
    for vid, label, i, frame_id, box in dataset.samples_list:
        image_key = make_image_key(vid, frame_id)
        gt_boxes = np.array(box[None])  # (1, 4), box coordinates in raw resolution
        
        # coordinate normalization
        raw_resolution = dataset.data['resolution'][vid]  # (H, W)
        gt_boxes[:, [0, 2]] /= float(raw_resolution[1])  # Width
        gt_boxes[:, [1, 3]] /= float(raw_resolution[0])  # Height

        # label transformation
        if vid in dataset.data_known['videos']:
            gt_labels = [dataset.open_to_closed[label] + 1]  # single box, evaluator accept labels starting from 1
            _targets_known[image_key] = {
                "bbox": gt_boxes,
                "labels": gt_labels,
                'resolution': raw_resolution
            }
        
        if vid in dataset.data_unknown['videos']:
            gt_labels = [dataset.open_to_unseen[label] + 1]  # single box, evaluator accept labels starting from 1
            _targets_unknown[image_key] = {
                "bbox": gt_boxes,
                "labels": gt_labels,
                'resolution': raw_resolution
            }

    return _targets_known, _targets_unknown


def write_to_csv(detections, save_file, logger):
    # save detections into csv file
    with open(save_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for image_key, dets in detections.items():
            video_name, frame_id = image_key.split(",")[0], image_key.split(",")[1]
            boxes, scores, action_ids = dets['boxes'], dets['scores'], dets['action_ids']
            assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]
            for box, score, action_id in zip(boxes, scores, action_ids):
                box_str = ['{:.5f}'.format(cord) for cord in box]
                score_str = '{:.5f}'.format(score)
                writer.writerow([video_name, frame_id] + box_str + [action_id, score_str])
    logger.info("Detection results are saved into {}".format(save_file))


def update_base_novel_mAP(eval_res, vocabulary):
    # base and novel class mAP
    base_map, novel_map = 0, 0
    num_base, num_novel = 0, 0
    key = None
    base_classes = vocabulary['closed']
    novel_classes = list(set(vocabulary['open']).difference(set(vocabulary['closed'])))
    for k, v in eval_res.items():
        class_name = k.split('AP@0.5IOU/')[-1]
        if class_name in base_classes:
            base_map += v
            num_base += 1
        elif class_name in novel_classes:
            novel_map += v
            num_novel += 1
        if 'mAP' in k:
            key = k

    if num_base > 0:
        base_map /= num_base
    if num_novel > 0:
        novel_map /= num_novel
    eval_res.update({'{}(base)'.format(key): base_map,
                     '{}(novel)'.format(key): novel_map})


def do_jhmdb_evaluation(dataset, predictions, output_folder, logger, metric='frame_ap', save_csv=False):
    # determine the evaluation vocabularies
    if dataset.open_vocabulary:
        vocab = dataset.vocabulary['open'] if dataset.eval_open else dataset.vocabulary['closed']
    else:
        vocab = dataset.closed_set_classes
    
    spatial_only = False
    temporal_only = False

    if metric == 'frame_ap':
        logger.info("Preparing JHMDB results for frame_ap evaluation.")

        if dataset.eval_open and dataset.independent_eval:
            # get prediction results of known and unknown subsets
            _results_known, _results_unknown = prepare_preds_ind(predictions, dataset, cls_offset=1, normalize=True)
            _results = {**_results_known, **_results_unknown}

            # prepare targets of known and unknown subsets
            _targets_known, _targets_unknown = _prepare_gts_ind(dataset)

            # evaluate
            eval_known, eval_unknown = {}, {}
            if len(_results_known) > 0 or len(_targets_known) > 0:
                logger.info("Eval frame-mAP on base classes:")
                eval_known = frame_mAP_pascal(_results_known, _targets_known, dataset.vocabulary['closed'], logger)
            
            if len(_results_unknown) > 0 or len(_targets_unknown) > 0:
                logger.info("Eval frame-mAP on novel classes:")
                vocab_novel = [dataset.vocabulary['open'][k] for k, v in dataset.open_to_unseen.items()]
                eval_unknown = frame_mAP_pascal(_results_unknown, _targets_unknown, vocab_novel, logger)
            
            # merge evaluation
            eval_res = {**eval_known, **eval_unknown}
            all_aps = list(dict(filter(lambda elem: 'PerformanceByCategory/AP' in elem[0], eval_res.items())).values())
            eval_res.update({'PascalBoxes_Precision/mAP@{}IOU'.format(0.5): np.mean(all_aps)})

        else:
            _results, _targets, _ = prepare_for_frame_ap(predictions, dataset, len(vocab))
            eval_res = frame_mAP_pascal(_results, _targets, vocab, logger)
    
    elif metric == 'video_ap':
        logger.info("Preparing JHMDB results for video_ap evaluation.")
        
        if dataset.eval_open and dataset.independent_eval:
            # get prediction results of known and unknown subsets
            _results_known, _results_unknown = prepare_preds_ind(predictions, dataset)
            _results = {**_results_known, **_results_unknown}
            # formatting
            _results_known = prediction_to_tubes(_results_known)
            _results_unknown = prediction_to_tubes(_results_unknown)

            # evaluate
            eval_known = videoAP(dataset.data_known, _results_known, thr=0.5, print_info=False, spatial_only=spatial_only, temporal_only=temporal_only)
            eval_unknown = videoAP(dataset.data_unknown, _results_unknown, thr=0.5, print_info=False, spatial_only=spatial_only, temporal_only=temporal_only)
            
            # merge evaluation
            eval_res = {**eval_known, **eval_unknown}
            all_aps = list(dict(filter(lambda elem: 'PerformanceByCategory/AP' in elem[0], eval_res.items())).values())
            eval_res.update({'PascalBoxes_Precision/mAP@{}IOU'.format(0.5): np.mean(all_aps)})

        else:
            _results = prepare_for_video_ap(predictions, dataset)

            if save_csv:
                # write to CSV file
                with tempfile.NamedTemporaryFile() as f:
                    file_path = f.name
                    if output_folder:
                        file_path = os.path.join(output_folder, "result.csv")
                    write_to_csv(_results, file_path, logger)
            
            # transform into pickle format
            _results_pkl = prediction_to_tubes(_results)
            _targets_pkl = copy.deepcopy(dataset.data)
            _targets_pkl.update({'labels': vocab})
            # evaluate
            eval_res = videoAP(_targets_pkl, _results_pkl, thr=0.5, print_info=False)

    if dataset.open_vocabulary and dataset.eval_open:
        update_base_novel_mAP(eval_res, dataset.vocabulary)

    logger.info('Evaluation results ({}):\n'.format(metric) + pformat(eval_res, indent=2))
    if output_folder:
        log_name = "result_ind.log" if dataset.eval_open and dataset.independent_eval else "result.log"
        if temporal_only: log_name = 'result_ind_temponly.log'
        if dataset.prior_boxes_test: log_name = log_name[:-4] + '_priorbox3.log'
        log_file_path = os.path.join(output_folder, log_name)
        with open(log_file_path, "w") as logf:
            logf.write("Evaluation results (metric: {})\n".format(metric))
            logf.write(pformat(eval_res))
    return eval_res, _results