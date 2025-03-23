import numpy as np
import os
import cv2
import argparse
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from alphaction.modeling.encoders.viclip import retrieve_text, _frame_from_video
from alphaction.config import cfg
from alphaction.dataset import make_data_loader
from alphaction.dataset.datasets import utils as utils
from alphaction.utils.random_seed import set_seed



def get_cfg():
    parser = argparse.ArgumentParser(description="PyTorch Action Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-final-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--skip-val-in-train",
        dest="skip_val",
        help="Do not validate during training",
        action="store_true",
    )
    parser.add_argument(
        "--transfer",
        dest="transfer_weight",
        help="Transfer weight from a pretrained model",
        action="store_true"
    )
    parser.add_argument(
        "--adjust-lr",
        dest="adjust_lr",
        help="Adjust learning rate scheduler from old checkpoint",
        action="store_true"
    )
    parser.add_argument(
        "--no-head",
        dest="no_head",
        help="Not load the head layer parameters from weight file",
        action="store_true"
    )
    parser.add_argument(
        "--use-tfboard",
        action='store_true',
        dest='tfboard',
        help='Use tensorboard to log stats'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Manual seed at the begining."
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = 1
    args.distributed = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Merge config.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(args.seed, 0, num_gpus)

    return cfg


def get_one_sample(dataset):
    idx = int(np.random.choice(list(range(len(dataset))), 1))
    video_idx, sec_idx, sec, center_idx = dataset._keyframe_indices[idx]
    # Get the frame idxs for current clip.
    seq = utils.get_sequence(
        center_idx,
        dataset._seq_len // 2,
        dataset._sample_rate,
        num_frames=len(dataset._image_paths[video_idx]),
    )

    # Load images of current clip.
    image_paths = [dataset._image_paths[video_idx][frame] for frame in seq]
    imgs = utils.retry_load_images(
        image_paths, backend='cv2'
    )

    clip_label_list = dataset._keyframe_boxes_and_labels[video_idx][sec_idx]
    assert len(clip_label_list) > 0
    labels = []
    for box_labels in clip_label_list:
        for label in box_labels[1]:
            if label == -1:
                continue
            label = dataset.id_to_indices['closed'][label]
            labels.append(label)
    
    return imgs, labels



if __name__ == '__main__':

    cfg = get_cfg()

    data_loader, vocabulary_train, iter_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )
    
    for n in range(10):
        print("Trial {}...".format(n + 1))
        frames, labels = get_one_sample(data_loader.dataset)
        class_texts = [elems['caption'] for clsname, elems in data_loader.dataset.text_input['closed'].items()]
        gt_texts = [class_texts[clsid] for clsid in labels]

        texts, probs = retrieve_text(frames, class_texts, name='viclip', topk=5, weight_file='pretrained/ViClip-InternVid-10M-FLT.pth')

        for t, p in zip(texts, probs):
            print(f'text: {t} ~ prob: {p:.4f}')
        
        print("Ground Truth class texts: ", gt_texts)