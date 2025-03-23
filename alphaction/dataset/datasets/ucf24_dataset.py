#!/usr/bin/env python3

import logging
import numpy as np
import torch
import os
import pickle
import json
import copy

import alphaction.dataset.datasets.utils as utils
from alphaction.dataset.datasets.cv2_transform import PreprocessWithBoxes

logger = logging.getLogger(__name__)


class UCF24(torch.utils.data.Dataset):
    """
    UCF24 Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        # get some cfg
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._anno_sample_rate = 5
        self._seq_len = self._video_length * self._sample_rate

        # data preprocessor
        self.preprocess_with_box = PreprocessWithBoxes(split, cfg.DATA, cfg.UCF24)

        # train & eval settings
        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY  # True
        self.eval_open = cfg.TEST.EVAL_OPEN  # True
        self.open_world_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.UCF24.OPEN_WORLD_DIR)
        self.refine_vocab = cfg.DATA.REFINE_VOCAB

        # evaluation setting
        self.multilabel_action = cfg.MODEL.MULTI_LABEL_ACTION  # False
        self.independent_eval = cfg.TEST.INDEPENDENT_EVAL  # True
        
        # annotation data
        self.data, known_classes = self._load_anno_data(cfg)
        self._num_classes = len(known_classes)
        # video path
        self.video_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.UCF24.FRAME_DIR)  # data/UCF24/rgb-images

        if self.open_vocabulary:
            self.vocabulary = self._read_vocabulary(os.path.join(self.open_world_path, 'vocab_open.txt'))
            self.vocabulary.update({'closed': known_classes})
            if self.refine_vocab:
                with open(os.path.join(self.open_world_path, cfg.UCF24.VOCAB_REFINE), 'r') as f:
                    refine_maps = json.load(f)
                self.text_input = {k: {vocab: {'caption': refine_maps[vocab][0] if cfg.UCF24.SINGLE_PROMPT else refine_maps[vocab]} for vocab in vocab_list} 
                                         for k, vocab_list in self.vocabulary.items()}
            else:
                self.text_input = {k: {vocab: {'caption': vocab} for vocab in vocab_list} for k, vocab_list in self.vocabulary.items()}
            
            self.open_to_closed, self.closed_to_open = self._get_mappings()
            
            if self._split == 'test' and self.eval_open and self.independent_eval:
                self._update_local_targets()
        else:
            self.closed_set_classes = known_classes
        
        self.use_prior_map = cfg.MODEL.USE_PRIOR_MAP  # False
        self.test_iou_thresh = cfg.TEST.IOU_THRESH  # 0.5 by default

        self.prior_boxes_init = cfg.MODEL.PRIOR_BOXES_INIT
        self.prior_boxes_test = cfg.TEST.PRIOR_BOX_TEST
        if self.prior_boxes_init == 'det' or (self._split == 'test' and self.prior_boxes_test):
            if cfg.UCF24.PRIOR_BOX_FILE.endswith('pkl'):
                dets_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.UCF24.PRIOR_BOX_FILE)
                print("Loading prior boxes from: {}".format(dets_file))
                self.dets_data = utils.load_dets_data(dets_file, topk=1)
            elif cfg.UCF24.PRIOR_BOX_FILE.endswith('pth'):
                print("Loading prior boxes from: {}".format(cfg.UCF24.PRIOR_BOX_FILE))
                self.dets_data = torch.load(cfg.UCF24.PRIOR_BOX_FILE)

        self.pre_extract_feat = cfg.MODEL.PRE_EXTRACT_FEAT  # if true, we will not load images from scratch
        if self.pre_extract_feat:
            setting = cfg.UCF24.CW_SPLIT_FILE[:-4].replace("/","-")  #  train50%-closed_world_0
            model_str = cfg.MODEL.BACKBONE.CONV_BODY.replace("/","")  # ViP-B16
            #  data/ucf24/ViP-B16_train50%-closed_world_0_16x1/
            self.feature_dir = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'features', f'{model_str}_{setting}_{self._video_length}x{self._sample_rate}_{split}')

        # key frames (annotated frames)
        self.samples_list = []
        for vid in self.data['videos']:  # loop the videos
            if vid not in self.data['boxes']:
                continue  # some videos may not have boxes
            for i, (fid, annos) in enumerate(self.data['boxes'][vid].items()):  # loop the frames
                if i % self._anno_sample_rate != 0:
                    continue
                actions = annos[:, 0].astype(int)  # action id within [0, 23]
                self.samples_list += [(vid, fid, actions, annos[:, 1:5])]

        logger.info("{} {} samples annotated.".format(self.samples_list.__len__(), self._split))
    

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
        if not self.open_vocabulary:
            pass        
        else: 
            # load the closed world annos for train/test, e.g., "data/UCF24/openworld/train50%/closed_world_0.pkl"
            anno_file = os.path.join(self.open_world_path, cfg.UCF24.CW_SPLIT_FILE)  # train annos, or test known part
            anno_data, known_classes = self._load_data(anno_file)
            # in testing, if eval_open, needs to include the unknown
            if self._split == 'test' and self.eval_open:
                anno_file = os.path.join(self.open_world_path, cfg.UCF24.OW_SPLIT_FILE)  # unknown part
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
            return data['train'], classes
        else:
            return data['test'], classes  # closed_world / open_world test set
    

    def _read_vocabulary(self, vocab_file):
        vocab = {'open': []}
        assert os.path.exists(vocab_file), "Vocabulary file does not exist: {}".format(vocab_file)
        with open(vocab_file, 'r') as f:
            for line in f.readlines():
                vocab['open'].append(line.strip())
        return vocab

    
    def _get_mappings(self):
        # open-to-closed mapping of class IDs
        open_to_closed = {self.vocabulary['open'].index(name): id for id, name in enumerate(self.vocabulary['closed'])} # from values in [0,20] to values in [0,9]
        # closed-to-open mapping of class IDs
        closed_to_open = {v: k for k, v in open_to_closed.items()}  # from values in [0,9] to values in [0,20]
        return open_to_closed, closed_to_open


    def _update_local_targets(self):
        # update known class labels (open_world to seen)
        updated_known = copy.deepcopy(self.data_known)
        for vid in self.data_known['videos']:  # loop the videos
            if vid not in self.data_known['boxes']:
                continue
            for fid, annos in self.data_known['boxes'][vid].items():
                action = annos[:, 0].astype(int)
                new_action = np.array([self.open_to_closed[act] for act in action])  # from [0-23] to [0-11], 12 seens
                new_annos = np.hstack((new_action[:, None].astype(float), annos[:, 1:]))
                updated_known['boxes'][vid][fid] = new_annos  # replace frame-level annos
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
            if vid not in self.data_unknown['boxes']:
                continue
            for fid, annos in self.data_unknown['boxes'][vid].items():
                action = annos[:, 0].astype(int)
                new_action = np.array([self.open_to_unseen[act] for act in action]) # from [0-23] to [0-11], 12 unseens
                new_annos = np.hstack((new_action[:, None].astype(float), annos[:, 1:]))
                updated_unknown['boxes'][vid][fid] = new_annos  # replace frame-level annos
        self.data_unknown = updated_unknown  # change pointer
        unseen_classes = [self.vocabulary['open'][cls_id] for cls_id in unseen_ids]
        self.data_unknown.update({'labels': unseen_classes})


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
    

    def _labels_transform(self, labels, num_classes, num_boxes=1, onehot=True, training=True):
        label_list = labels.tolist()
        # from open to closed set
        if self.open_vocabulary:
            # only for open-vocabulary setting, 
            if training or (not self.eval_open):
                # labels can be outside of the range of (train/test) vocabulary when:
                # 1) training, 2) testing only closed-set: labels can be outside of closed set range when evaluating the closed test set
                label_list = [self.open_to_closed[act] for act in label_list]
        if onehot:
            # Construct label arrays.
            label_arrs = np.zeros((num_boxes, num_classes), dtype=np.int32)
            label_arrs[np.arange(num_boxes), np.array(label_list)] = 1  # onehot for all boxes
            return label_arrs
        return np.array(label_list)


    def save_features(self, index_batch, patch_feat, cls_feat, text_feat):
        """ save backbone features by keyframe
        """
        for i, index in enumerate(index_batch):
            info = self.get_video_info(index)
            imgkey = "%s,%05d" % (info['vid'], float(info['frame_id']))
            save_file = os.path.join(self.feature_dir, imgkey + '.pt')
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({'patch_feat': patch_feat[i], 
                        'cls_feat': cls_feat[i], 
                        'text_feat': text_feat[i]}, save_file)


    def finished_feat_extraction(self):
        """ check if all samples are processed
        """
        for vid, frame_id, _, _ in self.samples_list:
            imgkey = "%s,%05d" % (vid, float(frame_id))
            feat_file = os.path.join(self.feature_dir, imgkey + '.pt')
            if not os.path.exists(feat_file):
                return False
        return True


    def load_features(self, vid, frame_id):
        """ load pre-extracted features
        """
        imgkey = "%s,%05d" % (vid, float(frame_id))
        save_file = os.path.join(self.feature_dir, imgkey + '.pt')
        if not os.path.exists(save_file):
            return None
        features = torch.load(save_file)
        return features
    

    def get_video_info(self, index):
        vid, frame_id, labels, boxes = self.samples_list[index]
        resolution = self.data['resolution'][vid]  # (H, W)
        return dict(vid=vid, frame_id=frame_id, boxes=boxes, labels=labels, resolution=resolution)
    

    def __getitem__(self, index):
        vid, frame_id, labels, boxes_raw = self.samples_list[index]
        raw_resolution = self.data['resolution'][vid]  # (H, W)
        
        if self.pre_extract_feat:
            features = self.load_features(vid, frame_id)
            assert features is not None
            # create dummy images for box preprocessing
            imgs = np.ones((self._seq_len, raw_resolution[0], raw_resolution[1], 3), dtype=np.uint8)
        else:
            # Get the frame idxs for current clip.
            seq = self.get_sequence(frame_id, self.data['nframes'][vid])
            # read video frames
            image_paths = [os.path.join(self.video_path, vid, "{:0>5}.jpg".format(frame + 1)) for frame in seq]
            imgs = utils.retry_load_images(
                image_paths, backend='cv2'
            )

        # load the annotations for training dataset
        boxes = None
        if self._split == 'train' or self.prior_boxes_init == 'gt':
            boxes = boxes_raw.copy()
            boxes[:, [0, 2]] /= float(raw_resolution[1])  # normalized box coordinates
            boxes[:, [1, 3]] /= float(raw_resolution[0])

        extra_boxes = None
        if self.prior_boxes_init == 'det':
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
            label_arrs = self._labels_transform(labels, self._num_classes, len(boxes))
        
        pathways = self.cfg.MODEL.BACKBONE.PATHWAYS
        imgs = utils.pack_pathway_output(self.cfg, imgs, pathways=pathways)
        if pathways == 1:
            slow, fast = imgs[0], None
        else:
            slow, fast = imgs[0], imgs[1]

        h, w = slow.shape[-2:]
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32)

        extras = {'extra_boxes': extra_boxes}
        if self.pre_extract_feat:
            extras.update(features)

        return slow, fast, whwh, boxes, label_arrs, extras, index