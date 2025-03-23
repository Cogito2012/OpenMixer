from .pascal_evaluation import object_detection_evaluation, standard_fields
import numpy as np



def parse_id(activity_list=None, class_num=24):
    if activity_list is None:  # use the class ID instead
        activity_list = ['Class{}'.format(i) for i in range(class_num)]
        # activity_list = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
    categories = []
    for i, act_name in enumerate(activity_list):
        categories.append({'id': i + 1, 'name': act_name})
    return categories


class STDetectionEvaluaterUCF(object):
    '''
    evaluater class designed for multi-iou thresholds
        based on https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py
    parameters:
        dataset that provide GT annos, in the format of AWSCVMotionDataset
        tiou_thresholds: a list of iou thresholds
    attributes:
        clear(): clear detection results, GT is kept
        load_detection_from_path(), load anno from a list of path, in the format of [confi x1 y1 x2 y2 scoresx15]
        evaluate(): run evaluation code
    '''

    def __init__(self, tiou_thresholds=[0.5], load_from_dataset=False, activity_list=None, class_num=24):
        categories = parse_id(activity_list=activity_list, class_num=class_num)
        self.class_num = class_num
        self.categories = categories
        self.tiou_thresholds = tiou_thresholds
        self.lst_pascal_evaluator = []
        self.load_from_dataset = load_from_dataset
        self.exclude_key = []
        for iou in self.tiou_thresholds:
            self.lst_pascal_evaluator.append(
                object_detection_evaluation.PascalDetectionEvaluator(categories, matching_iou_threshold=iou))

    def clear(self):
        for evaluator in self.lst_pascal_evaluator:
            evaluator.clear()

    def load_ground_truth(self, ground_truth):
        # write into evaluator
        for image_key, info in ground_truth.items():
            boxes = info['bbox'].copy()  # normalized coordinates
            resolution = info['resolution']
            boxes_eval = []
            for box in boxes:
                area = (box[3] - box[1]) * resolution[0] * (box[2] - box[0]) * resolution[1]
                if area < 10: continue  # ignore too small boxes
                boxes_eval.append(box)
            if len(boxes_eval) == 0:  # no boxes
                self.exclude_key.append(image_key)  # mark the excluded frames to filter the detections later
                continue
            
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_ground_truth_image_info(
                    image_key, {
                        standard_fields.InputDataFields.groundtruth_boxes:
                            np.vstack(boxes_eval),
                        standard_fields.InputDataFields.groundtruth_classes:
                            np.array(info['labels'], dtype=int),
                        standard_fields.InputDataFields.groundtruth_difficult:
                            np.zeros(len(boxes_eval), dtype=bool)
                    })
    

    def load_detection(self, detections):
        """ Load detection results from dict memory
        """
        for image_key, info in detections.items():
            # filtering out results that are in the excluded frames
            if image_key in self.exclude_key or len(info['boxes']) == 0:
                continue

            # sorted by confidence:
            boxes, labels, scores = info['boxes'], info['action_ids'], info['scores']
            index = np.argsort(-scores)
            boxes, labels, scores = boxes[index], labels[index], scores[index]

            # add info into evaluator
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_detected_image_info(
                    image_key, {
                        standard_fields.DetectionResultFields.detection_boxes: boxes,
                        standard_fields.DetectionResultFields.detection_classes: labels,
                        standard_fields.DetectionResultFields.detection_scores: scores
                    })

    def evaluate(self):
        result = {}
        for x, iou in enumerate(self.tiou_thresholds):
            evaluator = self.lst_pascal_evaluator[x]
            metrics = evaluator.evaluate()
            result.update(metrics)
        return result


def frame_mAP_pascal(_results, _targets, vocab, logger, iou_list=[0.5]):
    evaluater = STDetectionEvaluaterUCF(tiou_thresholds=iou_list, activity_list=vocab, class_num=len(vocab))

    logger.info("Adding ground truth into evaluator")
    evaluater.load_ground_truth(_targets)

    logger.info("Adding predictions into evaluator")
    evaluater.load_detection(_results)

    eval_res = evaluater.evaluate()
    
    return eval_res