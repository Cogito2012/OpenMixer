import os
from pprint import pformat
import torch
import torch.nn.functional as F
import numpy as np
import copy
import time
from collections import defaultdict
from ..evaluate_map import videoAP, prediction_to_tubes, gt_to_tubes
from ..pascal_wrapper import frame_mAP_pascal
from ..pascal_evaluation.np_box_ops import iou



def update_base_novel_mAP(eval_res, vocabulary, thr=0.5):
    # base and novel class mAP
    base_map, novel_map = 0, 0
    num_base, num_novel = 0, 0
    key = None
    base_classes = vocabulary['closed']
    novel_classes = list(set(vocabulary['open']).difference(set(vocabulary['closed'])))
    for k, v in eval_res.items():
        class_name = k.split('AP@{}IOU/'.format(thr))[-1]
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

        image_key = "%s,%05d" % (info['vid'], float(info['frame_id']))
        _results[image_key] = {
            "boxes": boxes,
            "scores": scores,
            "action_ids": action_ids
        }
    return _results


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
        image_key = "%s,%05d" % (info['vid'], float(info['frame_id']))
        
        # get the prediction
        boxes = prediction[0].numpy()  # (N, 4), normalized coordinates w.r.t. input resolution
        scores = prediction[1]
         # get the prior boxes (if any)
        if dataset.prior_boxes_test and image_key in dataset.dets_data:
            ## method 1 ####
            boxes_prior = dataset.dets_data[image_key][:, :4]  # (M, 4)
            iou_mat = iou(boxes_prior, boxes)  # (M, N)
            maxids = np.argmax(iou_mat, axis=1)  # (M,)
            scores_prior = prediction[1][maxids]
            # update scores and boxes (override)
            scores = scores_prior.clone()
            boxes = np.copy(boxes_prior)

        if not normalize:
            # boxes to raw resolution
            boxes[:, [0, 2]] *= info['resolution'][1]  # x1, x2
            boxes[:, [1, 3]] *= info['resolution'][0]  # y1, y2
        
        if info['vid'] in dataset.data_known['videos']:
            scores_known = scores[:, ids_known]  # logits, (N, K) classes defined in open-world K=21
            scores_known = torch.sigmoid(scores_known) if dataset.multilabel_action else F.softmax(scores_known, dim=-1)
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
            scores_unknown = torch.sigmoid(scores_unknown) if dataset.multilabel_action else F.softmax(scores_unknown, dim=-1)
            scores_unknown = scores_unknown.numpy()  # (N, K) 
            
            # filtering with score threshold (unknown)
            box_ids, action_ids = np.where(scores_unknown >= score_thresh)
            _results_unknown[image_key] = {
                "boxes": boxes[box_ids, :],
                "scores": scores_unknown[box_ids, action_ids],
                "action_ids": action_ids + cls_offset
            }

    return _results_known, _results_unknown


def print_time(logger, message, start):
    logger.info("==> %g seconds to %s", time.time() - start, message)


def prepare_gt(data, logger):
    """ Convert frame-level action boxes into video-level action tubes
    """
    gt_boxes = defaultdict(list)
    gt_labels = defaultdict(list)
    video_ids = []

    for vid in data['videos']:
        if vid not in data['boxes']:
            continue
        video_ids.append(vid)
        for fid, annos in data['boxes'][vid].items():
            clip_key = "%s,%04d" % (vid, float(fid))
            gt_boxes[clip_key] = annos[:, 1:5].tolist()
            gt_labels[clip_key] = annos[:, 0].astype(int).tolist()

    start = time.time()
    gt_tubes = gt_to_tubes(gt_boxes, gt_labels, {}, logger, normalized=False)
    print_time(logger, "convert to action tubes", start)

    targets = {'videos': video_ids, 
               'labels': data['labels'],
               'gttubes': gt_tubes}

    return targets


def _prepare_gts_ind(dataset):
    _targets_known, _targets_unknown = {}, {}
    # get ground truth  vid, frame_id, labels, boxes_raw
    for vid, frame_id, all_labels, gt_boxes in dataset.samples_list:
        image_key = "%s,%04d" % (vid, float(frame_id))
        
        # coordinate normalization
        raw_resolution = dataset.data['resolution'][vid]  # (H, W)
        gt_boxes[:, [0, 2]] /= float(raw_resolution[1])  # Width
        gt_boxes[:, [1, 3]] /= float(raw_resolution[0])  # Height

        # label transformation
        if vid in dataset.data_known['videos']:
            gt_labels = [dataset.open_to_closed[label] + 1 for label in all_labels]  # single box, evaluator accept labels starting from 1
            _targets_known[image_key] = {
                "bbox": gt_boxes,
                "labels": gt_labels,
                'resolution': raw_resolution
            }
        
        if vid in dataset.data_unknown['videos']:
            gt_labels = [dataset.open_to_unseen[label] + 1 for label in all_labels]  # single box, evaluator accept labels starting from 1
            _targets_unknown[image_key] = {
                "bbox": gt_boxes,
                "labels": gt_labels,
                'resolution': raw_resolution
            }

    return _targets_known, _targets_unknown


def do_ucf24_evaluation(dataset, predictions, output_folder, logger, metric='frame_ap', save_csv=False):
    
    if dataset.open_vocabulary:
        vocab = dataset.vocabulary['open'] if dataset.eval_open else dataset.vocabulary['closed']
    else:
        vocab = dataset.closed_set_classes
    
    if metric == 'frame_ap':
        assert dataset.eval_open and dataset.independent_eval

        # get prediction results of known and unknown subsets
        _results_known, _results_unknown = prepare_preds_ind(predictions, dataset, cls_offset=1, normalize=True)
        _results = {**_results_known, **_results_unknown}

        # prepare targets of known and unknown subsets
        _targets_known, _targets_unknown = _prepare_gts_ind(dataset)

        # evaluate
        eval_known, eval_unknown = {}, {}
        if len(_results_known) > 0 or len(_targets_known) > 0:
            logger.info("Eval frame-mAP on base classes:")
            eval_known = frame_mAP_pascal(_results_known, _targets_known, dataset.vocabulary['closed'], logger, iou_list=[dataset.test_iou_thresh])
        
        if len(_results_unknown) > 0 or len(_targets_unknown) > 0:
            logger.info("Eval frame-mAP on novel classes:")
            vocab_novel = [dataset.vocabulary['open'][k] for k, v in dataset.open_to_unseen.items()]
            eval_unknown = frame_mAP_pascal(_results_unknown, _targets_unknown, vocab_novel, logger, iou_list=[dataset.test_iou_thresh])
        
        # merge evaluation
        eval_res = {**eval_known, **eval_unknown}
        all_aps = list(dict(filter(lambda elem: 'PerformanceByCategory/AP' in elem[0], eval_res.items())).values())
        eval_res.update({'PascalBoxes_Precision/mAP@{}IOU'.format(dataset.test_iou_thresh): np.mean(all_aps)})

    elif metric == 'video_ap':
        logger.info("Preparing UCF24 results for video_ap evaluation.")

        if dataset.eval_open and dataset.independent_eval:
            # get prediction results of known and unknown subsets
            _results_known, _results_unknown = prepare_preds_ind(predictions, dataset)
            _results = {**_results_known, **_results_unknown}
            
            # formatting prediction to action tubes
            _results_known = prediction_to_tubes(_results_known, anno_rate=dataset._anno_sample_rate)
            _results_unknown = prediction_to_tubes(_results_unknown, anno_rate=dataset._anno_sample_rate)

            # formatting GT to action tubes
            targets_known = prepare_gt(dataset.data_known, logger)
            targets_unknown = prepare_gt(dataset.data_unknown, logger)

            # evaluate
            eval_known = videoAP(targets_known, _results_known, thr=dataset.test_iou_thresh, anno_rate=dataset._anno_sample_rate, print_info=False)
            eval_unknown = videoAP(targets_unknown, _results_unknown, thr=dataset.test_iou_thresh, anno_rate=dataset._anno_sample_rate, print_info=False)
            
            # merge evaluation
            eval_res = {**eval_known, **eval_unknown}
            all_aps = list(dict(filter(lambda elem: 'PerformanceByCategory/AP' in elem[0], eval_res.items())).values())
            eval_res.update({'PascalBoxes_Precision/mAP@{}IOU'.format(dataset.test_iou_thresh): np.mean(all_aps)})
        
        else:
            _results = prepare_for_video_ap(predictions, dataset)

            # transform into pickle format
            _results_pkl = prediction_to_tubes(_results, anno_rate=dataset._anno_sample_rate)
            _targets_pkl = copy.deepcopy(dataset.data)
            _targets_pkl.update({'labels': vocab})
            _targets_pkl = prepare_gt(_targets_pkl, logger)
            # evaluate
            eval_res = videoAP(_targets_pkl, _results_pkl, thr=dataset.test_iou_thresh, anno_rate=dataset._anno_sample_rate, print_info=False)

    if dataset.open_vocabulary and dataset.eval_open:
        update_base_novel_mAP(eval_res, dataset.vocabulary, thr=dataset.test_iou_thresh)

    logger.info('Evaluation results ({}):\n'.format(metric) + pformat(eval_res, indent=2))
    if output_folder:
        log_name = "result_ind.log" if dataset.eval_open and dataset.independent_eval else "result.log"
        log_name = log_name[:-4] + '_iou{}.log'.format(dataset.test_iou_thresh * 100) if dataset.independent_eval != 0.5 else log_name
        if dataset.prior_boxes_test: log_name = log_name[:-4] + '_priorbox.log'
        log_file_path = os.path.join(output_folder, log_name)
        with open(log_file_path, "w") as logf:
            logf.write(pformat(eval_res))
    return eval_res, _results
