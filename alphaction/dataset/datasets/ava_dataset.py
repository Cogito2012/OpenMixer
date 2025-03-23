#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch
import os

from . import ava_helper as ava_helper
from . import cv2_transform as cv2_transform
from .cv2_transform import PreprocessWithBoxes
from . import utils as utils
from .evaluation.ava.ava_eval import read_labelmap
import json, copy

logger = logging.getLogger(__name__)


class Ava(torch.utils.data.Dataset):
    """
    AVA Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.STM.ACTION_CLASSES

        # data preprocessor
        self.preprocess_with_box = PreprocessWithBoxes(split, cfg.DATA, cfg.AVA)
        
        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY
        self.eval_open = cfg.TEST.EVAL_OPEN
        if not self.open_vocabulary: self.eval_open = False # for closed vocabulary setting, no need to eval_open
        self.refine_vocab = cfg.DATA.REFINE_VOCAB

        anno_dir = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.AVA.ANNOTATION_DIR)  # data/AVA/openworld
        exclusion_file = os.path.join(anno_dir, cfg.AVA.EXCLUSION_FILE)

        if self.open_vocabulary:
            open_labelmap_file = os.path.join(anno_dir, "open_world.pbtxt")
            # openworld/within_group/closed_world_0.pbtxt
            closed_labelmap_file = os.path.join(anno_dir, cfg.AVA.CW_SPLIT_DIR.split('/')[0], cfg.AVA.LABEL_MAP_FILE)
            # openworld/within_group/closed_world_0/ava2.2_val_seen.csv
            csv_gt_files = [os.path.join(anno_dir, cfg.AVA.CW_SPLIT_DIR, filename) for filename in cfg.AVA.TEST_GT_BOX_LISTS]
            labelmap_file = open_labelmap_file if self.eval_open else closed_labelmap_file
        else:
            csv_gt_files = os.path.join(anno_dir, cfg.AVA.TEST_GT_BOX_LISTS)
            labelmap_file = os.path.join(anno_dir, cfg.AVA.LABEL_MAP_FILE)

        # files that used in ava_eval.py
        self.eval_file_paths = {
            "csv_gt_files": csv_gt_files,
            "labelmap_file": labelmap_file,
            "exclusion_file": exclusion_file,
        }

        if self.open_vocabulary:
            self.open_to_closed, self.closed_to_open, self.vocabulary, self.id_to_indices = self.get_mappings(closed_labelmap_file, open_labelmap_file)
            
            if self.refine_vocab:
                with open(os.path.join(anno_dir, cfg.AVA.VOCAB_REFINE), 'r') as f:
                    refine_maps = json.load(f)
                self.text_input = {k: {vocab: {'caption': refine_maps[vocab][0] if cfg.AVA.SINGLE_PROMPT else refine_maps[vocab]} for vocab in vocab_list} 
                                         for k, vocab_list in self.vocabulary.items()}
            else:
                self.text_input =  {k: {vocab: {'caption': vocab} for vocab in vocab_list} for k, vocab_list in self.vocabulary.items()}
        else:
            labelmaps, _ = read_labelmap(labelmap_file)
            self.closed_set_classes = [data['name'] for data in labelmaps]
        
        self.use_prior_map = cfg.MODEL.USE_PRIOR_MAP
        if self.use_prior_map:
            pass  # TODO
        self.prior_boxes_init = cfg.MODEL.PRIOR_BOXES_INIT
        
        self._load_data(cfg)


    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files
        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = ava_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        assert len(boxes_and_labels) == len(self._image_paths)  # number of videos

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== AVA dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))


    def get_mappings(self, mapfile, mapfile_open):
        """ Mapping the 60 open-world classes to 47/30 closed-world class IDs
            Return: dict() where the key is open-world class ID and value is the closed-world class ID
        """
        # a list of {"id": integer, "name": classname } dicts
        labelmaps_open, _ = read_labelmap(mapfile_open)
        labelmaps, _ = read_labelmap(mapfile)

        # new class ID starting from 1
        open_to_closed = {labelmaps_open.index(elem): i for i, elem in enumerate(labelmaps)}  # from indices in [0,59] to indices in [0,46/29]
        closed_to_open = {v: k for k, v in open_to_closed.items()}  # from values in [0,46/29] to values in [0,59]

        # get vocabularies
        vocabulary = dict()
        vocabulary['closed'] = [elem['name'] for elem in labelmaps]  # closed-set vocabularies in raw order
        vocabulary['open'] = [elem['name'] for elem in labelmaps_open]  # open-set vocabularies in raw order

        id_to_indices = dict()
        id_to_indices['closed'] = {elem['id']: i for i, elem in enumerate(labelmaps)}  # id in [1,80] to indices in [0,46]
        id_to_indices['open'] = {elem['id']: i for i, elem in enumerate(labelmaps_open)}   # id in [1,80] to indices in [0,59]

        return open_to_closed, closed_to_open, vocabulary, id_to_indices


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
        return len(self._keyframe_indices)

    def get_video_info(self, index):
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[index]
        # movie_name is the human-readable youtube name.
        movie_name = self._video_idx_to_name[video_idx]
        return dict(movie=movie_name, timestamp=sec)
    

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.
        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            time index (zero): The time index is currently not supported for AVA.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """

        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend='cv2'
        )

        # load the annotations for training dataset
        boxes = None
        if self._split == 'train' or self.prior_boxes_init == 'gt':
            clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
            assert len(clip_label_list) > 0

            # Get boxes and labels for current clip.
            boxes = []
            labels = []
            for box_labels in clip_label_list:
                boxes.append(box_labels[0])
                labels.append(box_labels[1])
            boxes = np.array(boxes)
            # Score is not used.
            boxes = boxes[:, :4].copy()

        imgs, boxes = self.preprocess_with_box.process(
            imgs, boxes=boxes
        )  # (3, T, 256, 341)

        label_arrs = None
        if self._split == 'train':
            # Construct label arrays.
            label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
            for i, box_labels in enumerate(labels):
                # AVA label index starts from 1.
                for label in box_labels:
                    if label == -1:
                        continue
                    # this label is the id, needs to be transformed into closed-set indices
                    if self.open_vocabulary:
                        label = self.id_to_indices['closed'][label]  # label id in [1,80] to indices in [0,46]
                    assert label >= 0 and label <= 80
                    label_arrs[i][label] = 1

        pathways = self.cfg.MODEL.BACKBONE.PATHWAYS
        imgs = utils.pack_pathway_output(self.cfg, imgs, pathways=pathways)
        if pathways == 1:
            slow, fast = imgs[0], None
        else:
            slow, fast = imgs[0], imgs[1]

        h, w = slow.shape[-2:]
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32)

        metadata = [video_idx, sec]

        extra_boxes = None
        extras = {'extra_boxes': extra_boxes,
                  "metadata": metadata,}

        return slow, fast, whwh, boxes, label_arrs, metadata, idx