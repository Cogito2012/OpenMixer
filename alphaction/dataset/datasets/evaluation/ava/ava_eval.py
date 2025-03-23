import numpy as np
import tempfile
import os
from pprint import pformat
import csv
import time
from collections import defaultdict, OrderedDict
from ..pascal_evaluation import object_detection_evaluation, standard_fields
from ..evaluate_map import videoAP, frameAP, prediction_to_tubes, gt_to_tubes
import torch


def do_ava_evaluation(dataset, predictions, output_folder, logger, metric='frame_ap', save_csv=False):
    logger.info("Preparing results for AVA format")
    ava_results = prepare_for_ava_detection(predictions, dataset)
    logger.info("Evaluating predictions")
    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, "result.csv")
        if len(dataset.eval_file_paths) == 0 and save_csv:
            write_csv(ava_results, file_path, logger)
            return
        eval_res = evaluate_predictions_on_ava(
            dataset, ava_results, file_path, logger, metric=metric, save_csv=save_csv
        )
    
    if dataset.open_vocabulary:
        update_base_novel_mAP(eval_res, dataset.vocabulary)
    
    logger.info(pformat(eval_res, indent=2))
    if output_folder:
        log_file_path = os.path.join(output_folder, "result.log")
        with open(log_file_path, "w") as logf:
            logf.write("Evaluation results (metric: {})\n".format(metric))
            logf.write(pformat(eval_res))
    return eval_res, ava_results


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, float(timestamp))


def decode_image_key(image_key):
    return image_key[:-5], image_key[-4:]


def prepare_for_ava_detection(predictions, dataset):
    ava_results = {}
    score_thresh = 0.0
    for video_id, prediction in enumerate(predictions):
        video_info = dataset.get_video_info(video_id)
        if len(prediction) == 0:
            continue
        #  normalized xyxy boxes, (N, 4). scores: (N, 60)
        boxes, scores = prediction[0], torch.sigmoid(prediction[1])

        box_ids, action_ids = torch.where(scores >= score_thresh)  # (N*60,)
        boxes = boxes[box_ids, :].numpy()
        scores = scores[box_ids, action_ids].numpy()

        if dataset.open_vocabulary and (not dataset.eval_open):
            action_ids = np.array([dataset.closed_to_open[id] for id in action_ids])

        video_name = video_info['movie']
        timestamp = video_info['timestamp']
        # clip_key = make_image_key(video_name, timestamp)
        clip_key = "%s,%04d" % (video_name, float(timestamp))

        ava_results[clip_key] = {
            "boxes": boxes,
            "scores": scores,
            "action_ids": action_ids
        }
    return ava_results


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
      exclusions_file: Path of file containing a csv of video-id,timestamp.

    Returns:
      A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
      or an empty set if exclusions file is None.
    """
    excluded = set()
    exclusions_file = open(exclusions_file, 'r')
    if exclusions_file:
        reader = csv.reader(exclusions_file)
        for row in reader:
            assert len(row) == 2, "Expected only 2 columns, got: {}".format(row)
            excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
      labelmap_file: Path of file containing a label map protocol buffer.

    Returns:
      labelmap: The label map in the form used by the object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
      class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    labelmap_file = open(labelmap_file, 'r')
    for line in labelmap_file:
        if line.startswith("  name:"):
            name = line.split('"')[1]
        elif line.startswith("  id:") or line.startswith("  label_id:"):
            class_id = int(line.strip().split(" ")[-1])
            labelmap.append({"id": class_id, "name": name})
            class_ids.add(class_id)
    return labelmap, class_ids


def read_csv(csv_file, logger, class_whitelist=None):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
      csv_file: Path of csv file.
      class_whitelist: If provided, boxes corresponding to (integer) class labels
        not in this set are skipped.

    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      labels: A dictionary mapping each unique image key (string) to a list of
        integer class lables, matching the corresponding box in `boxes`.
      scores: A dictionary mapping each unique image key (string) to a list of
        score values lables, matching the corresponding label in `labels`. If
        scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    csv_file = open(csv_file, 'r')
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], "Wrong number of columns: " + row
        # image_key = make_image_key(row[0], row[1])
        image_key = "%s,%04d" % (row[0], float(row[1]))
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue
        score = 1.0
        if len(row) == 8:
            score = float(row[7])
        boxes[image_key].append([y1, x1, y2, x2])
        labels[image_key].append(action_id)
        scores[image_key].append(score)
    print_time(logger, "read file " + csv_file.name, start)
    return boxes, labels, scores


def write_csv(ava_results, csv_result_file, logger):
    start = time.time()
    with open(csv_result_file, 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=',')
        for clip_key in ava_results:
            movie_name, timestamp = decode_image_key(clip_key)
            cur_result = ava_results[clip_key]
            boxes = cur_result["boxes"]
            scores = cur_result["scores"]
            action_ids = cur_result["action_ids"]
            assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]
            for box, score, action_id in zip(boxes, scores, action_ids):
                box_str = ['{:.5f}'.format(cord) for cord in box]
                score_str = '{:.5f}'.format(score)
                spamwriter.writerow([movie_name, timestamp, ] + box_str + [action_id, score_str])
    print_time(logger, "write file " + csv_result_file, start)


def print_time(logger, message, start):
    logger.info("==> %g seconds to %s", time.time() - start, message)


def read_all_gts(gt_files, logger, class_whitelist):
    all_boxes, all_labels = defaultdict(list), defaultdict(list)
    for gt_file in gt_files:
        # Reads the ground truth dataset.
        boxes, labels, _ = read_csv(gt_file, logger, class_whitelist)
        for k, v in boxes.items():
            all_boxes[k].extend(v)
            all_labels[k].extend(labels[k])
    return all_boxes, all_labels


def transform_format(ava_results, logger, box_fmt="yxyx"):
    start = time.time()
    result_boxes = defaultdict(list)
    result_labels = defaultdict(list)
    result_scores = defaultdict(list)
    for clip_key in ava_results:
        # movie_name, timestamp = decode_image_key(clip_key)
        cur_result = ava_results[clip_key]
        boxes = cur_result["boxes"]
        scores = cur_result["scores"]
        action_ids = cur_result["action_ids"]
        assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]
        for box, score, action_id in zip(boxes, scores, action_ids):
            action_id = int(action_id) + 1  # from [0, 59] to [1, 60]

            if box_fmt == "yxyx":  # pascal_evaluator format
                box = box[[1, 0, 3, 2]]  # y1, x1, y2, x2

            result_boxes[clip_key].append(box.tolist())
            result_labels[clip_key].append(action_id)
            result_scores[clip_key].append(score)
    
    print_time(logger, "transform prediction format ", start)
    return result_boxes, result_labels, result_scores


def transform_gt_format(dataset, logger, class_whitelist=None, box_fmt="xyxy"):
    start = time.time()
    gt_boxes = defaultdict(list)
    gt_labels = defaultdict(list)

    # loop over key frames
    for video_idx, sec_idx, sec, center_idx in dataset._keyframe_indices:
        # get the clip_key
        # clip_key = make_image_key(dataset._video_idx_to_name[video_idx], sec)
        clip_key = "%s,%04d" % (dataset._video_idx_to_name[video_idx], float(sec))
        # get the annos of this keyframe
        clip_label_list = dataset._keyframe_boxes_and_labels[video_idx][sec_idx]

        # loop over boxes
        for box_labels in clip_label_list:
            box = box_labels[0]  # normalized (x1,y1,x2,y2)
            if box_fmt == "yxyx":  # pascal_evaluator format
                box = [box[1], box[0], box[3], box[2]]
            # loop over action ID [multiple action IDs for each box]
            for action_id in box_labels[1]:
                if class_whitelist and action_id not in class_whitelist:
                    continue
                if dataset.open_vocabulary:
                    # for eval_open, we do recognition over 60 classes, so that id in [1, 80] needs to be transformed into [1,60]
                    # otherwise, we do recognition over 47/30 classes, so that id in [1, 80] needs to be transformed into [1,47/30]
                    action_id = dataset.id_to_indices['open'][action_id] + 1 if dataset.eval_open else dataset.id_to_indices['closed'][action_id] + 1

                gt_boxes[clip_key].append(box)
                gt_labels[clip_key].append(action_id)
    
    print_time(logger, "transform groundtruth format ", start)
    return gt_boxes, gt_labels


def evaluate_frame_AP_pascal(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, categories, excluded_keys, logger):

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

    start = time.time()
    for image_key in boxes_gt:
        if image_key in excluded_keys:
            logger.info(("Found excluded timestamp in ground truth: %s. "
                        "It will be ignored."), image_key)
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key, {
                standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(boxes_gt[image_key], dtype=float),
                standard_fields.InputDataFields.groundtruth_classes:
                    np.array(labels_gt[image_key], dtype=int),
                standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(boxes_gt[image_key]), dtype=bool)
            })
    print_time(logger, "convert groundtruth", start)

    start = time.time()
    for image_key in boxes_pred:
        if image_key in excluded_keys:
            logger.info(("Found excluded timestamp in detections: %s. "
                         "It will be ignored."), image_key)
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key, {
                standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes_pred[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels_pred[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores_pred[image_key], dtype=float)
            })
    print_time(logger, "convert detections", start)

    start = time.time()
    metrics = pascal_evaluator.evaluate()
    print_time(logger, "run_evaluator", start)

    return metrics



def evaluate_frame_AP(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, categories, excluded_keys, logger):
    start = time.time()
    # filtering out frames in excluded_keys
    detections = {}
    for image_key in boxes_pred:
        if image_key in excluded_keys:
            logger.info(("Found excluded timestamp in detections: %s. "
                         "It will be ignored."), image_key)
            continue
        detections[image_key] = {
            "boxes": np.array(boxes_pred[image_key], dtype=float),
            "scores": np.array(scores_pred[image_key], dtype=float),
            "action_ids": np.array(labels_pred[image_key], dtype=float)
        }
    detections = prediction_to_tubes(detections, normalized=True)
    print_time(logger, "convert detections", start)
    
    # format AVA ground truth as the that of the JHMDB
    start = time.time()
    video_ids = list(set([img_key.split(",")[0] for img_key in boxes_gt.keys()]))
    cls_names = [(elem['id'], elem['name']) for elem in categories]
    ordered_names = [elem[1] for elem in sorted(cls_names)]
    gt_tubes = gt_to_tubes(boxes_gt, labels_gt, excluded_keys, logger, normalized=True)
    ground_truths = {'videos': video_ids, 
                     'labels': ordered_names,
                     'gttubes': gt_tubes}
    print_time(logger, "convert groundtruth", start)

    start = time.time()
    eval_results = frameAP(ground_truths, detections, thr=0.5, start_cls=1, print_info=False)
    print_time(logger, "run_evaluator", start)

    return eval_results


def evaluate_video_AP(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, categories, excluded_keys, logger):
    start = time.time()
    # filtering out frames in excluded_keys
    detections = {}
    for image_key in boxes_pred:
        if image_key in excluded_keys:
            logger.info(("Found excluded timestamp in detections: %s. "
                         "It will be ignored."), image_key)
            continue
        detections[image_key] = {
            "boxes": np.array(boxes_pred[image_key], dtype=float),
            "scores": np.array(scores_pred[image_key], dtype=float),
            "action_ids": np.array(labels_pred[image_key], dtype=float)
        }
    detections = prediction_to_tubes(detections, normalized=True)
    print_time(logger, "convert detections", start)
    
    # format AVA ground truth as the that of the JHMDB
    start = time.time()
    video_ids = list(set([img_key.split(",")[0] for img_key in boxes_gt.keys()]))
    cls_names = [(elem['id'], elem['name']) for elem in categories]
    ordered_names = [elem[1] for elem in sorted(cls_names)]
    gt_tubes = gt_to_tubes(boxes_gt, labels_gt, excluded_keys, logger, normalized=True)
    ground_truths = {'videos': video_ids, 
                     'labels': ordered_names,
                     'gttubes': gt_tubes}
    print_time(logger, "convert groundtruth", start)

    start = time.time()
    eval_results = videoAP(ground_truths, detections, thr=0.5, print_info=False)
    print_time(logger, "run_evaluator", start)

    return eval_results


def update_categories(categories, dataset):
    mapping = dataset.id_to_indices['open'] if dataset.eval_open else dataset.id_to_indices['closed']
    categories_new = {mapping[elem['id']] + 1: elem['name'] for elem in categories}
    # sort by id
    categories_new = OrderedDict(sorted(categories_new.items()))
    # dict to list
    categories_new = [{'id': id, 'name': name} for id, name in categories_new.items()]
    return categories_new


def evaluate_predictions_on_ava(dataset, ava_results, csv_result_file, logger, metric='frame_ap', save_csv=False):
    if save_csv:
        write_csv(ava_results, csv_result_file, logger)

    labelmap = dataset.eval_file_paths["labelmap_file"]
    exclusions = dataset.eval_file_paths["exclusion_file"]

    categories, class_whitelist = read_labelmap(labelmap)
    # re-assign the label id as the evaluator uses maximum label id to determine number of classes
    categories = update_categories(categories, dataset)

    logger.info("CATEGORIES (%d):\n%s", len(categories),
                pformat(categories, indent=2))
    excluded_keys = read_exclusions(exclusions)

    # transform GT and prediction boxes
    if save_csv:
        gt_files = dataset.eval_file_paths["csv_gt_files"]
        boxes_gt, labels_gt = read_all_gts(gt_files, logger, class_whitelist)  # 234155
        # Reads detections dataset.
        boxes_pred, labels_pred, scores_pred = read_csv(csv_result_file, logger, class_whitelist)
    else:
        boxes_gt, labels_gt = transform_gt_format(dataset, logger, class_whitelist, box_fmt="yxyx")  # 234056
        boxes_pred, labels_pred, scores_pred = transform_format(ava_results, logger, box_fmt="yxyx")

    # evaluate
    if metric == 'frame_ap':
        # Frame-level AP is the default choice of STMixer
        logger.info("AVA evaluation by PASCAL frame_ap.")
        metrics = evaluate_frame_AP_pascal(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, categories, excluded_keys, logger)
    
    elif metric == 'frame_ap_hit':
        logger.info("AVA evaluation by non-PASCAL frame_ap.")
        metrics = evaluate_frame_AP(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, categories, excluded_keys, logger)
        
    elif metric == 'video_ap':
        logger.info("AVA evaluation by video_ap.")
        metrics = evaluate_video_AP(boxes_gt, labels_gt, boxes_pred, labels_pred, scores_pred, categories, excluded_keys, logger)
    
    return metrics


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
    base_map /= num_base
    novel_map /= num_novel
    eval_res.update({'{}(base)'.format(key): base_map,
                     '{}(novel)'.format(key): novel_map})