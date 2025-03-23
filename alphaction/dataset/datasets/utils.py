#!/usr/bin/env python3

import logging
import numpy as np
import time
import cv2
import torch
from iopath.common.file_io import g_pathmgr
import os
import pickle


logger = logging.getLogger(__name__)


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.
    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.
    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with g_pathmgr.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def read_greyscale_image(img_file):
    assert os.path.exists(img_file), "File does not exist!\n{}".format(img_file)
    im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    im = im.astype(np.float32) / 255.0
    im = torch.from_numpy(im)
    return im


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.
    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames
    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq

def pack_pathway_output(cfg, frames, pathways=2):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if pathways==1:
        frame_list = [frames]
    elif pathways==2:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError()
    return frame_list


def load_dets_data(det_file, topk=None):
    assert os.path.exists(det_file), "detection file does not exist: {}".format(det_file)
    with open(det_file, 'rb') as fid:
        data = pickle.load(fid, encoding='iso-8859-1')
    # get list of all frames
    all_dets = dict()
    for vid, dets in data.items():
        for i in list(dets['boxes'].keys()):
            boxes, scores = dets['boxes'][i], dets['scores'][i]
            key = "%s,%05d" % (vid, i)
            if topk is None:
                all_dets[key] = np.hstack((boxes, scores[:, None]))  # (n, 5)
            else:
                indices = np.argsort(scores)[::-1][:topk]  # topK maximum indices
                all_dets[key] = np.hstack((boxes[indices], scores[indices, None]))  # (n, 5)
    return all_dets