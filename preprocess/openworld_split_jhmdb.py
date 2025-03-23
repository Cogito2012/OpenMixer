import os
import pickle
import random
import copy


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


def closed_world_only():

    split_setting = 'train_all'
    result_setting_dir = os.path.join(result_dir, split_setting)
    os.makedirs(result_setting_dir, exist_ok=True)
    write_list(classes_ow, os.path.join(result_setting_dir, 'vocab_closed.txt'))

    closed_world = {'labels': classes_ow, 'train': [dict() for _ in range(3)], 'test': [dict() for _ in range(3)]}

    for sample_split in range(3):
        # train videos
        known_videos_train, unknown_videos_train = split_samples(annos['train_videos'][sample_split], classes_ow)
        closed_world['train'][sample_split]['videos'] = copy.deepcopy(known_videos_train)
        assert len(unknown_videos_train) == 0  # should be empty
        
        # test videos (remaining subset of train videos & all test videos)
        known_videos_test, unknown_videos_test = split_samples(annos['test_videos'][sample_split], classes_ow)
        closed_world['test'][sample_split]['videos'] = copy.deepcopy(known_videos_test)
        assert len(unknown_videos_test) == 0  # should be empty 

        for key in ['gttubes', 'nframes', 'resolution']:
            # split annotations
            closed_world['train'][sample_split][key] = copy.deepcopy(get_annos(annos[key], known_videos_train))

            closed_world['test'][sample_split][key] = copy.deepcopy(get_annos(annos[key], known_videos_test))
            anno_part2 = copy.deepcopy(get_annos(annos[key], unknown_videos_test))
    
    save_file = os.path.join(result_setting_dir, 'closed_world.pkl')
    with open(save_file, 'wb') as fid:
        pickle.dump(closed_world, fid, protocol=pickle.HIGHEST_PROTOCOL)
    

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

    dataset_root = '../data/JHMDB'
    result_dir = os.path.join(dataset_root, 'openworld')
    os.makedirs(result_dir, exist_ok=True)
    random.seed(42)
    num_splits = 32
    train_ratio = 0.50
    set_open_world = True
    small_open_world = True
    # closed-world classes used by the ZS-STAD paper
    # classes_cw = ['brush_hair', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick', 'pour', 
    #               'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'stand', 'swing_baseball', 'walk']

    anno_file = os.path.join(dataset_root, 'JHMDB-GT.pkl')
    with open(anno_file, 'rb') as fid:
        annos = pickle.load(fid, encoding='iso-8859-1')
    
    vocab_open_file = os.path.join(result_dir, "vocab_open.txt")
    if not os.path.exists(vocab_open_file):
        classes_ow = copy.deepcopy(annos['labels'])
        # write the open-world name list
        write_list(classes_ow, vocab_open_file)
    else:
        classes_ow = read_list(vocab_open_file)

    if not set_open_world:
        closed_world_only()
        exit(0)

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
        closed_world = {'labels': classes_cw, 'train': [dict() for _ in range(3)], 'test': [dict() for _ in range(3)]}
        open_world = {'labels': classes_cw, 'test': [dict() for _ in range(3)]}

        for sample_split in range(3):
            # train videos (subset of train videos)
            known_videos_train, unknown_videos_train = split_samples(annos['train_videos'][sample_split], classes_cw)
            closed_world['train'][sample_split]['videos'] = copy.deepcopy(known_videos_train)
            open_world['test'][sample_split]['videos'] = copy.deepcopy(unknown_videos_train) if not small_open_world else []

            # test videos (remaining subset of train videos & all test videos)
            known_videos_test, unknown_videos_test = split_samples(annos['test_videos'][sample_split], classes_cw)
            closed_world['test'][sample_split]['videos'] = copy.deepcopy(known_videos_test)
            open_world['test'][sample_split]['videos'] += copy.deepcopy(unknown_videos_test)

            for key in ['gttubes', 'nframes', 'resolution']:
                # split annotations
                closed_world['train'][sample_split][key] = copy.deepcopy(get_annos(annos[key], known_videos_train))
                anno_part1 = copy.deepcopy(get_annos(annos[key], unknown_videos_train)) if not small_open_world else {}

                closed_world['test'][sample_split][key] = copy.deepcopy(get_annos(annos[key], known_videos_test))
                anno_part2 = copy.deepcopy(get_annos(annos[key], unknown_videos_test))
                open_world['test'][sample_split][key] = {**anno_part1, **anno_part2}

        save_prefix = os.path.join(result_setting_dir, '{}_{}.pkl')
        with open(os.path.join(save_prefix.format('closed_world', n)), 'wb') as fid:
            pickle.dump(closed_world, fid, protocol=pickle.HIGHEST_PROTOCOL)
        
        save_file = os.path.join(save_prefix.format('open_world', n))
        if small_open_world:
             save_file = save_file[:-4] + '_small' + '.pkl'
        with open(save_file, 'wb') as fid:
            pickle.dump(open_world, fid, protocol=pickle.HIGHEST_PROTOCOL)