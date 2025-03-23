#!/usr/bin/env python3

import logging
import numpy as np
import torch
import os
import pickle
import json
import copy

# import sys, argparse
# DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')  # the project root path
# sys.path.append(DIR_PATH)
import alphaction.dataset.datasets.utils as utils
from alphaction.dataset.datasets.cv2_transform import PreprocessWithBoxes

logger = logging.getLogger(__name__)


class Jhmdb(torch.utils.data.Dataset):
    """
    JHMDB Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        # get some cfg
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._samples_split = cfg.JHMDB.SAMPLES_SPLIT  # [0, 1, 2]

        # data preprocessor
        self.preprocess_with_box = PreprocessWithBoxes(split, cfg.DATA, cfg.JHMDB)

        # train & eval settings
        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY
        self.eval_open = cfg.TEST.EVAL_OPEN  # if eval_open, both known and unknown classes are evaluated
        if not self.open_vocabulary: self.eval_open = False # for closed vocabulary setting, no need to eval_open
        self.open_world_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.JHMDB.OPEN_WORLD_DIR)
        self.refine_vocab = cfg.DATA.REFINE_VOCAB
        
        # evaluation setting
        self.multilabel_action = cfg.MODEL.MULTI_LABEL_ACTION
        self.independent_eval = cfg.TEST.INDEPENDENT_EVAL  # indepedent mAP evaluation on seen and unseen classes.
        
        # annotation data
        self.data, known_classes = self._load_anno_data(cfg)
        self._num_classes = len(known_classes)
        # video path
        self.video_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.JHMDB.FRAME_DIR)  # data/JHMDB/Frames

        if self.open_vocabulary:
            self.vocabulary = self._read_vocabulary(os.path.join(self.open_world_path, 'vocab_open.txt'))
            self.vocabulary.update({'closed': known_classes})
            if self.refine_vocab:
                with open(os.path.join(self.open_world_path, cfg.JHMDB.VOCAB_REFINE), 'r') as f:
                    refine_maps = json.load(f)
                self.text_input = {k: {vocab: {'caption': refine_maps[vocab]} for vocab in vocab_list} 
                                         for k, vocab_list in self.vocabulary.items()}
            else:
                self.text_input = {k: {vocab: {'caption': vocab} for vocab in vocab_list} for k, vocab_list in self.vocabulary.items()}

            if cfg.JHMDB.VOCAB_HN and self._split == 'train':  # read hard negatives
                with open(os.path.join(self.open_world_path, cfg.JHMDB.VOCAB_HN), 'r') as f:
                    vocab_hn = json.load(f)
                for vocab in self.vocabulary['closed']:
                    self.text_input['closed'][vocab].update({'hardneg': vocab_hn[vocab]})
            
            self.open_to_closed, self.closed_to_open = self._get_mappings()
            
            if self._split == 'test' and self.eval_open and self.independent_eval:
                self._update_local_targets()
        else:
            self.closed_set_classes = known_classes
        
        self.use_prior_map = cfg.MODEL.USE_PRIOR_MAP
        if self.use_prior_map:
            split_dir = os.path.dirname(os.path.join(self.open_world_path, cfg.JHMDB.CW_SPLIT_FILE))
            maps_dir = os.path.join(split_dir, 'prior_maps_{}'.format(self._samples_split))
            self.prior_map = self.load_priors(maps_dir)
        
        self.prior_boxes_init = cfg.MODEL.PRIOR_BOXES_INIT
        self.prior_boxes_test = cfg.TEST.PRIOR_BOX_TEST
        if self.prior_boxes_init or (self._split == 'test' and self.prior_boxes_test):
            if self.prior_boxes_init == 'rand':
                boxes_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.JHMDB.PRIOR_BOX_FILE)
                print("Loading random boxes from: {}".format(boxes_file))
                self.dets_data = utils.load_dets_data(boxes_file)
            else:
                if cfg.JHMDB.PRIOR_BOX_FILE.endswith('pkl'):
                    dets_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.JHMDB.PRIOR_BOX_FILE)
                    print("Loading prior boxes from: {}".format(dets_file))
                    self.dets_data = utils.load_dets_data(dets_file, topk=1)
                elif cfg.JHMDB.PRIOR_BOX_FILE.endswith('pth'):
                    print("Loading prior boxes from: {}".format(cfg.JHMDB.PRIOR_BOX_FILE))
                    self.dets_data = torch.load(cfg.JHMDB.PRIOR_BOX_FILE)

        self.test_iou_thresh = cfg.TEST.IOU_THRESH  # 0.5 by default
        
        # key frames (annotated frames)
        self.samples_list = []
        for vid in self.data['videos']:  # loop the videos
            for action, tubes_list in self.data['gttubes'][vid].items():  # loop the action classes (only one action class in practice)
                for i, tube in enumerate(tubes_list):  # loop the action tubes for each class
                    self.samples_list += [(vid, int(action), i, int(frame_box[0]), frame_box[1:5]) for frame_box in tube]  # loop the annotated frames

        print(self.samples_list.__len__(), "actions annotated.")
    

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.samples_list)
    

    def _load_anno_data(self, cfg):
        if not self.open_vocabulary:  # data/JHMDB/JHMDB-GT.pkl
            anno_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.JHMDB.GROUND_TRUTH_FILE)
            # load dataset
            anno_data, known_classes = self._load_data(anno_file)
        
        else:  # data/JHMDB/openworld/train50%/closed_world_0.pkl
            # load the closed world annos for train/test
            anno_file = os.path.join(self.open_world_path, cfg.JHMDB.CW_SPLIT_FILE)  # train annos, or test known part
            anno_data, known_classes = self._load_data(anno_file)
            # in testing, if eval_open, needs to include the unknown
            if self._split == 'test' and self.eval_open:
                anno_file = os.path.join(self.open_world_path, cfg.JHMDB.OW_SPLIT_FILE)  # unknown part
                data_unknown, _ = self._load_data(anno_file)
                # for indipendent evaluation, needs to keep known and unknown separately.
                if self.independent_eval:
                    self.data_known = copy.deepcopy(anno_data) # used in eval only
                    self.data_unknown = copy.deepcopy(data_unknown) 
                # merge
                for k, v in anno_data.items():
                    if isinstance(v, list):
                        v.extend(data_unknown[k])
                    elif isinstance(v, dict):
                        v.update(data_unknown[k])
        return anno_data, known_classes


    def _load_data(self, anno_file):
        assert os.path.exists(anno_file), "Annotation file does not exist: {}".format(anno_file)
        with open(anno_file, 'rb') as fid:
            data = pickle.load(fid, encoding='iso-8859-1')
        classes = data['labels']
        if self._split == 'train':
            return data['train'][self._samples_split], classes
        else:
            return data['test'][self._samples_split], classes  # closed_world / open_world test set

    
    def _update_local_targets(self):
        # update known class labels (open_world to seen)
        updated_known = copy.deepcopy(self.data_known)
        for vid in self.data_known['videos']:  # loop the videos
            new_annos = {}
            for action, tubes_list in self.data_known['gttubes'][vid].items():
                new_action = self.open_to_closed[action]  # from [0-20] to [0-9], 10 seens
                new_annos[new_action] = copy.deepcopy(tubes_list)
            updated_known['gttubes'][vid] = new_annos  # replace video-level annos
        self.data_known = updated_known  # change pointer
        self.data_known.update({'labels': self.vocabulary['closed']})
        
        # class index mapping from open_world to unseen_world
        open_ids = range(len(self.vocabulary['open']))
        seen_ids = list(self.open_to_closed.keys())
        unseen_ids = list(set(open_ids).difference(set(seen_ids)))
        self.open_to_unseen = {id: i for i, id in enumerate(unseen_ids)}
        
        # update unknown class labels (open_world to unseen)
        updated_unknown = copy.deepcopy(self.data_unknown)
        for vid in self.data_unknown['videos']:  # loop the videos
            new_annos = {}
            for action, tubes_list in self.data_unknown['gttubes'][vid].items():
                new_action = self.open_to_unseen[action]  # from [0-20] to [0-10], 11 unseens
                new_annos[new_action] = copy.deepcopy(tubes_list)
            updated_unknown['gttubes'][vid] = new_annos  # replace video-level annos
        self.data_unknown = updated_unknown  # change pointer
        unseen_classes = [self.vocabulary['open'][cls_id] for cls_id in unseen_ids]
        self.data_unknown.update({'labels': unseen_classes})


    def _read_vocabulary(self, vocab_file):
        vocab = {'open': []}
        assert os.path.exists(vocab_file), "Vocabulary file does not exist: {}".format(vocab_file)
        with open(vocab_file, 'r') as f:
            for line in f.readlines():
                vocab['open'].append(line.strip())
        return vocab
    

    def get_sequence(self, frame_id, num_frames):
        half_len = self._seq_len // 2
        start_id = max(frame_id - half_len, 0)
        end_id = min(frame_id + half_len, num_frames-1)
        seq_fids = [s for s in range(start_id, end_id)]
        # padding the frame sequence of the video clip
        if len(seq_fids) < self._seq_len:  # needs to pad the sequence
            pad_size = (self._seq_len - len(seq_fids)) // 2
            pad_left = [0 for _ in range(pad_size)]
            pad_right = [end_id for _ in range(self._seq_len - len(seq_fids) - pad_size)]
            seq_fids = pad_left + seq_fids + pad_right
        assert len(seq_fids) == self._seq_len
        # temporal sampling
        seq_fids = [seq_fids[i] for i in range(0, self._seq_len, self._sample_rate)]
        return seq_fids
    

    def _get_mappings(self):
        # open-to-closed mapping of class IDs
        open_to_closed = {self.vocabulary['open'].index(name): id for id, name in enumerate(self.vocabulary['closed'])} # from values in [0,20] to values in [0,9]
        # closed-to-open mapping of class IDs
        closed_to_open = {v: k for k, v in open_to_closed.items()}  # from values in [0,9] to values in [0,20]
        return open_to_closed, closed_to_open
    
    
    def load_priors(self, map_dir):
        # prior height
        prior_file = os.path.join(map_dir, 'prior_height.png')
        prior_height = utils.read_greyscale_image(prior_file)
        # prior width
        prior_file = os.path.join(map_dir, 'prior_width.png')
        prior_width = utils.read_greyscale_image(prior_file)
        # form a single tensor
        prior_map = torch.stack((prior_width, prior_height), dim=0)  # (2, H, W)
        return prior_map
    

    def _labels_transform(self, label, num_classes=10, num_boxes=1, onehot=True, training=True):
        # from open to closed set
        if self.open_vocabulary:
            # only for open-vocabulary setting, 
            if training or (not self.eval_open):
                # labels can be outside of the range of (train/test) vocabulary when:
                # 1) training, 2) testing only closed-set: labels can be outside of closed set range when evaluating the closed test set
                label = self.open_to_closed[label]
        assert label >= 0 and label <= 20
        if onehot:
            # Construct label arrays.
            label_arrs = np.zeros((num_boxes, num_classes), dtype=np.int32)
            label_arrs[:, label] = 1  # onehot for all boxes
            return label_arrs
        return [label] * num_boxes


    def get_video_info(self, index):
        vid, label, i, frame_id, box = self.samples_list[index]
        raw_resolution = self.data['resolution'][vid]  # (H, W)
        return dict(vid=vid, frame_id=frame_id, box=box, label=label, resolution=raw_resolution)


    def __getitem__(self, index):
        vid, label, i, frame_id, box = self.samples_list[index]
        raw_resolution = self.data['resolution'][vid]  # (H, W)

        # Get the frame idxs for current clip.
        seq = self.get_sequence(frame_id, self.data['nframes'][vid])

        # read video frames
        image_paths = [os.path.join(self.video_path, vid, "{:0>5}.png".format(frame + 1)) for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend='cv2'
        )

        # load the annotations for training dataset
        boxes = None
        if self._split == 'train' or self.prior_boxes_init == 'gt':
            boxes = box[None].copy()
            boxes[:, [0, 2]] /= float(raw_resolution[1])  # normalized box coordinates
            boxes[:, [1, 3]] /= float(raw_resolution[0])
        
        extra_boxes = None
        if self.prior_boxes_init in ['det', 'rand']:
            img_key = "%s,%05d" % (vid, float(frame_id))
            if img_key in self.dets_data:
                box_scores = self.dets_data[img_key]  # (n, 5), normalized xyxy
            else:
                box_scores = np.array([[0, 0, imgs[0].shape[-1]-1, imgs[0].shape[-2]-1, 0]])
            extra_boxes = box_scores[:, :4].astype(np.float32)
            # temporarily, we append extra boxes to the GT boxes
            if self._split == 'train':
                num_gt_boxes = len(boxes)
                boxes = np.vstack((boxes, extra_boxes))
            else:
                boxes = extra_boxes.copy()
        
        # preprocess images with boxes
        imgs, boxes = self.preprocess_with_box.process(
            imgs, boxes=boxes
        )  # (3, T, 256, 341)

        if extra_boxes is not None:
            if self._split == 'train':
                extra_boxes = boxes[num_gt_boxes:].copy()
                boxes = boxes[:num_gt_boxes].copy() 
            else:
                extra_boxes = boxes.copy()
                boxes = None

        label_arrs = None
        if self._split == 'train':
            label_arrs = self._labels_transform(label, self._num_classes, len(boxes))
        
        pathways = self.cfg.MODEL.BACKBONE.PATHWAYS
        imgs = utils.pack_pathway_output(self.cfg, imgs, pathways=pathways)
        if pathways == 1:
            slow, fast = imgs[0], None
        else:
            slow, fast = imgs[0], imgs[1]

        h, w = slow.shape[-2:]
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32)
        extras = {'extra_boxes': extra_boxes}

        return slow, fast, whwh, boxes, label_arrs, extras, index


if __name__ == '__main__':
    
    import sys, argparse
    DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')  # the project root path
    sys.path.append(DIR_PATH)
    from alphaction.config import cfg
    from alphaction.dataset import make_data_loader
    from tqdm import tqdm
    import cv2

    parser = argparse.ArgumentParser(description="Dataset analysis")
    parser.add_argument(
        "--config-file",
        default="config_files/jhmdb_temp.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    # Merge config.
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.USE_PRIOR_MAP = False
    cfg.freeze()

    # make dataloader.
    data_loader, vocabulary_train, iter_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )

    all_boxes = []
    prior_height, prior_width = 240, 320
    size_prior_map = np.zeros((prior_height, prior_width, 2), dtype=np.float32)  # quantization bin 1/240 x 1/320
    count_map = np.zeros((prior_height, prior_width), dtype=np.float32) + 1e-9
    for slow_video, fast_video, whwh, boxes, labels, metadata, idx in tqdm(data_loader, total=len(data_loader)):
        boxes_normed = np.concatenate(boxes) / whwh.numpy()
        all_boxes.append(boxes_normed)
        
        ctr_x = 0.5 * (boxes_normed[0, 0] + boxes_normed[0, 2])
        ctr_y = 0.5 * (boxes_normed[0, 1] + boxes_normed[0, 3])
        loc_r = int(ctr_y * prior_height)
        loc_c = int(ctr_x * prior_width)
        
        size_prior_map[loc_r, loc_c, 0] += (boxes_normed[0, 2] - boxes_normed[0, 0]) # width
        size_prior_map[loc_r, loc_c, 1] += (boxes_normed[0, 3] - boxes_normed[0, 1]) # height
        count_map[loc_r, loc_c] += 1
    
    all_boxes = np.concatenate(all_boxes, axis=0)
    # compute mean
    size_prior_map /= count_map[:, :, None]
    
    aspect_ratios = (all_boxes[:, 2] - all_boxes[:, 0]) / (all_boxes[:, 3] - all_boxes[:, 1]) # (x2-x1) / (y2-y1)
    print(np.mean(aspect_ratios))
    
    split_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.JHMDB.OPEN_WORLD_DIR, cfg.JHMDB.CW_SPLIT_FILE)
    save_path = os.path.join(os.path.dirname(split_file), 'prior_maps_{}'.format(cfg.JHMDB.SAMPLES_SPLIT))
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, "prior_width.png"), size_prior_map[:, :, 0]*255)
    cv2.imwrite(os.path.join(save_path, "prior_height.png"), size_prior_map[:, :, 1]*255)