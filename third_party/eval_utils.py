import os, sys
LIB_PATH=['../alphaction/dataset/datasets/evaluation']
sys.path.extend(LIB_PATH)

from pascal_evaluation.object_detection_evaluation import PascalDetectionEvaluator
from pascal_evaluation.standard_fields import InputDataFields, DetectionResultFields

import pickle
import numpy as np



def load_gt_data(anno_file, split=0):
    assert os.path.exists(anno_file), "Annotation file does not exist: {}".format(anno_file)
    with open(anno_file, 'rb') as fid:
        data = pickle.load(fid, encoding='iso-8859-1')
    return data


def eval_person_boxes(results, gt_data):
    class_id = 1

    pascal_evaluator = PascalDetectionEvaluator([{'id': class_id, 'name': 'person'}],
                                                matching_iou_threshold=0.5)

    # prepare ground truth
    for vid, annos in gt_data['gttubes'].items():
        # each video contains only one action type
        act_id = list(annos.keys())[0]
        act_annos = annos[act_id][0]
        height, width = gt_data['resolution'][vid]
        # each action type contains only one action box on a frame
        for fid_box in act_annos:
            img_key = "%s,%04d" % (vid, float(fid_box[0]))
            box_normed = fid_box[1:5] / np.array([width, height, width, height], dtype=np.float32)  # (xyxy)
            box_normed = box_normed[[1, 0, 3, 2]]  # (yxyx)
            pascal_evaluator.add_single_ground_truth_image_info(
                img_key, {
                    InputDataFields.groundtruth_boxes: box_normed[None],
                    InputDataFields.groundtruth_classes: np.array([class_id], dtype=int),
                    InputDataFields.groundtruth_difficult: np.zeros(1, dtype=bool)
                })

    # prepare detection results
    for vid, dets in results.items():
        boxes, scores = dets['boxes'], dets['scores']
        frame_ids = list(boxes.keys())
        for fid in frame_ids:
            img_key = "%s,%04d" % (vid, float(fid))
            boxes_pred = boxes[fid].copy()
            boxes_pred = boxes_pred[:, [1, 0, 3, 2]]
            pascal_evaluator.add_single_detected_image_info(
                img_key, {
                    DetectionResultFields.detection_boxes: boxes_pred,
                    DetectionResultFields.detection_classes: np.array([class_id]*len(boxes[fid]), dtype=int),
                    DetectionResultFields.detection_scores:  scores[fid].copy()
                })
    
    eval_res = pascal_evaluator.evaluate()

    precisions = pascal_evaluator._evaluation.precisions_per_class
    recalls = pascal_evaluator._evaluation.recalls_per_class

    return eval_res, precisions, recalls