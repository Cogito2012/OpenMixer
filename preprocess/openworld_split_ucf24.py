import os
import pickle
import random
import numpy as np
import copy
import cv2


def read_labels(anno_dir):
    """ Anno structure:
            "labels": ['Basketball', 'BasketballDunk', ..., 'WalkingWithDog'],
            "boxes": {"Basketball/v_Basketball_g01_c01": {9: np.array((?, 5)), 10: np.array((?, 5)), ...}},
            ""
    """
    anno_data = {'labels': [], 'boxes': dict()}
    # get class labels
    all_classes = sorted([foldername for foldername in os.listdir(anno_dir) if os.path.isdir(os.path.join(anno_dir, foldername))])
    anno_data['labels'] = all_classes
    # get boxes
    for i, clsname in enumerate(all_classes):
        cls_dir = os.path.join(anno_dir, clsname)
        video_names = sorted([vid for vid in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, vid))])
        for vid in video_names:
            txt_dir = os.path.join(cls_dir, vid)
            anno_frames = sorted([int(fid[:-4]) for fid in os.listdir(txt_dir) if fid.endswith('.txt')])
            anno_data['boxes'][f"{clsname}/{vid}"] = dict()
            for fid in anno_frames:
                anno_file = os.path.join(txt_dir, '%05d.txt'%(fid))
                annos = np.loadtxt(anno_file)
                assert annos is not None, "Empty annotation file: {}".format(anno_file)
                annos = np.reshape(annos, (-1, 5))
                assert all(annos[:, 0] == i + 1), "invalida class labels"  # all boxes in a frame are from the same action
                annos[:, 0] -= 1  # change the class index to starting from zero
                anno_data['boxes'][f"{clsname}/{vid}"][fid] = annos  # (?, 5), [cls, x1, y1, x2, y2] unnormalized in [240, 320]
    return anno_data


def write_list(list_data, result_file):
    with open(result_file, 'w') as f:
        for name in list_data:
            f.writelines('{}\n'.format(name))


def read_list(result_file):
    list_data = []
    with open(result_file, 'r') as f:
        for line in f.readlines():
            list_data.append(line.strip())
    return list_data


def get_annos(dict_data, known_samples):
    return {k: v for k, v in dict_data.items() if k in known_samples}


def split_samples(list_data, known_cls):
    known_set, unknown_set = [], []
    for vid in list_data:
        if vid.split('/')[0] in known_cls:
            known_set.append(vid)
        else:
            unknown_set.append(vid)
    return known_set, unknown_set


def get_video_info(video_dir, list_data):
    info = {'resolution': dict(), 'nframes': dict()}
    for vid in list_data:
        frame_ids = sorted([int(fid.split(".")[0]) for fid in os.listdir(os.path.join(video_dir, vid)) if fid.endswith('.jpg')])
        file_first = os.path.join(video_dir, vid, '%05d.jpg'%(frame_ids[0]))  # starting from 1
        assert os.path.exists(file_first), "file does not exist: {}".format(file_first)
        assert frame_ids[0] == 1, "Not start from 1: {}".format(vid)
        im = cv2.imread(file_first)  # (H, W, 3)
        info['resolution'][vid] = im.shape[:2]
        info['nframes'][vid] = len(frame_ids)
    return info


def weighted_random_selection(classes, weights, num_selections):
    selected_classes = set()
    while len(selected_classes) < num_selections:
        num_prev = len(selected_classes)

        # Select a class based on weights
        selected_class = random.choices(classes, weights=weights)[0]
        selected_classes.add(selected_class)

        num_cur = len(selected_classes)
        if num_cur > num_prev:
            # Decrease the weight of the selected class
            idx = classes.index(selected_class)
            weights[idx] *= 0.5  # Decrease weight by half for each selection

    return selected_classes


def generate_topk_selections(classes, num_select, topk=32):
    weights = [1] * len(classes)
    all_selections = []
    for n in range(64):
        selected = weighted_random_selection(classes, weights, num_select)
        if selected not in all_selections:
            all_selections.append(selected)
    return all_selections[-topk:]


if __name__ == '__main__':

    dataset_root = '../data/UCF24'
    result_dir = os.path.join(dataset_root, 'openworld')
    os.makedirs(result_dir, exist_ok=True)
    random.seed(42)
    num_splits = 32
    train_ratio = 0.50
    set_open_world = True
    small_open_world = True

    # read UCF24 labels
    print("Reading the annotation files...")
    annotations = read_labels(os.path.join(dataset_root, 'labels'))

    # read UCF24 splits
    train_list = read_list(os.path.join(dataset_root, 'splitfiles', 'trainlist01.txt'))
    test_list = read_list(os.path.join(dataset_root, 'splitfiles', 'testlist01.txt'))

    # read resolutions
    print("Reading the video info...")
    video_info = get_video_info(os.path.join(dataset_root, 'rgb-images'), train_list + test_list)
    annotations.update(video_info)

    # prepare open-world vocabulary list
    vocab_open_file = os.path.join(result_dir, "vocab_open.txt")
    if not os.path.exists(vocab_open_file):
        classes_ow = annotations['labels']
        # write the open-world name list
        write_list(classes_ow, vocab_open_file)
    else:
        classes_ow = read_list(vocab_open_file)

    split_setting = 'train{}%'.format(int(train_ratio * 100)) + '_{}splits'.format(num_splits)
    result_setting_dir = os.path.join(result_dir, split_setting)
    os.makedirs(result_setting_dir, exist_ok=True)

    # generate all selections
    all_selections = generate_topk_selections(classes_ow, int(train_ratio * len(classes_ow)), topk=num_splits)

    for n in range(num_splits):
        vocab_closed_file = os.path.join(result_setting_dir, 'vocab_closed_{}.txt'.format(n))
        if not os.path.exists(vocab_closed_file):
            # write the closed-world classes
            # classes_cw = random.sample(classes_ow, int(train_ratio * len(classes_ow)))

            classes_cw = list(all_selections[n])
            
            write_list(classes_cw, vocab_closed_file)
        else:
            classes_cw = read_list(vocab_closed_file)

        # prepare the video list
        closed_world = {'labels': classes_cw, 'train': dict(), 'test': dict()}
        open_world = {'labels': classes_cw, 'test': dict()}

        # train videos (subset of train videos)
        known_videos_train, unknown_videos_train = split_samples(train_list, classes_cw)
        closed_world['train']['videos'] = copy.deepcopy(known_videos_train)
        open_world['test']['videos'] = copy.deepcopy(unknown_videos_train) if not small_open_world else []

        # test videos (remaining subset of train videos & all test videos)
        known_videos_test, unknown_videos_test = split_samples(test_list, classes_cw)
        closed_world['test']['videos'] = copy.deepcopy(known_videos_test)
        open_world['test']['videos'] += copy.deepcopy(unknown_videos_test)

        # fill in the infos (resolutions, nframes)
        for key in ['boxes', 'nframes', 'resolution']:
            # split annotations
            closed_world['train'][key] = copy.deepcopy(get_annos(annotations[key], known_videos_train))
            anno_part1 = copy.deepcopy(get_annos(annotations[key], unknown_videos_train)) if not small_open_world else {}

            closed_world['test'][key] = copy.deepcopy(get_annos(annotations[key], known_videos_test))
            anno_part2 = copy.deepcopy(get_annos(annotations[key], unknown_videos_test))
            open_world['test'][key] = {**anno_part1, **anno_part2}
        
        # save into files
        save_prefix = os.path.join(result_setting_dir, '{}_{}.pkl')
        with open(os.path.join(save_prefix.format('closed_world', n)), 'wb') as fid:
            pickle.dump(closed_world, fid, protocol=pickle.HIGHEST_PROTOCOL)
        
        save_file = os.path.join(save_prefix.format('open_world', n))
        if small_open_world:
             save_file = save_file[:-4] + '_small' + '.pkl'
        with open(save_file, 'wb') as fid:
            pickle.dump(open_world, fid, protocol=pickle.HIGHEST_PROTOCOL)