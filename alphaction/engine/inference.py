# modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/engine/inference.py
import logging
import os

import torch
from tqdm import tqdm
import time
import datetime

from alphaction.dataset.datasets.evaluation import evaluate
from alphaction.utils.comm import get_rank, is_main_process, all_gather, gather, synchronize, get_world_size
from alphaction.structures.memory_pool import MemoryPool


def compute_on_dataset_1stage(model, data_loader, device, debug=False):
    # single stage inference, for model without memory features
    # cpu_device = torch.device("cpu")
    results_dict = {}
    if get_world_size() == 1:
        extra_args = {}
    else:
        rank = get_rank()
        extra_args = dict(desc="testing", disable=(not rank==0))
    
    extras = {'prior_map': data_loader.dataset.prior_map.to(device)} if data_loader.dataset.use_prior_map else {}
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), **extra_args):
            if debug and i > 2:
                break
            slow_video, fast_video, whwh, boxes, _, metadata, idx = batch
            slow_video = slow_video.to(device)
            if fast_video is not None:
                fast_video = fast_video.to(device)
            whwh = whwh.to(device)
            
            if data_loader.dataset.prior_boxes_init == 'gt':
                extras.update({'prior_boxes': boxes})
            elif data_loader.dataset.prior_boxes_init in ['det', 'rand']:
                extras.update({'prior_boxes': [info['extra_boxes'] for info in metadata]})
            
            action_score_list, box_list = model(slow_video, fast_video, whwh, extras=extras)
            results_dict.update(
                {video_id: (box.cpu(), action_score.cpu()) for video_id, box, action_score in zip(idx, box_list, action_score_list)}
            )

    return results_dict


def compute_on_dataset_2stage(model, data_loader, device, logger):
    # two stage inference, for model with memory features.
    # first extract features and then do the inference
    cpu_device = torch.device("cpu")
    num_devices = get_world_size()
    dataset = data_loader.dataset
    if num_devices == 1:
        extra_args = {}
    else:
        rank = get_rank()
        extra_args = dict(desc="rank {}".format(rank))

    loader_len = len(data_loader)
    person_feature_pool = MemoryPool()
    batch_info_list = [None]*loader_len
    logger.info("Stage 1: extracting clip features.")
    start_time = time.time()

    for i, batch in enumerate(tqdm(data_loader, **extra_args)):
        slow_clips, fast_clips, boxes, objects, extras, video_ids = batch
        slow_clips = slow_clips.to(device)
        fast_clips = fast_clips.to(device)
        boxes = [box.to(device) for box in boxes]
        objects = [None if (box is None) else box.to(device) for box in objects]
        movie_ids = [e["movie_id"] for e in extras]
        timestamps = [e["timestamp"] for e in extras]
        with torch.no_grad():
            feature = model(slow_clips, fast_clips, boxes, objects, part_forward=0)
            person_feature = [ft.to(cpu_device) for ft in feature[0]]
            object_feature = [ft.to(cpu_device) for ft in feature[1]]
        # store person features into memory pool
        for movie_id, timestamp, p_ft, o_ft in zip(movie_ids, timestamps, person_feature, object_feature):
            person_feature_pool[movie_id, timestamp] = p_ft
        # store other information in list, for further inference
        batch_info_list[i] = (movie_ids, timestamps, video_ids, object_feature)

    # gather feature pools from different ranks
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Stage 1 time: {} ({} s / video per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    feature_pool = all_gather(person_feature_pool)
    all_feature_pool_p = MemoryPool()
    all_feature_pool_p.update_list(feature_pool)
    del feature_pool, person_feature_pool

    # do the inference
    results_dict = {}
    logger.info("Stage 2: predicting with extracted feature.")
    start_time = time.time()
    for movie_ids, timestamps, video_ids, object_feature in tqdm(batch_info_list, **extra_args):
        current_feat_p = [all_feature_pool_p[movie_id, timestamp].to(device)
                          for movie_id, timestamp in zip(movie_ids, timestamps)]
        current_feat_o = [ft_o.to(device) for ft_o in object_feature]
        extras = dict(
            person_pool=all_feature_pool_p,
            movie_ids=movie_ids,
            timestamps=timestamps,
            current_feat_p=current_feat_p,
            current_feat_o=current_feat_o,
        )
        with torch.no_grad():
            output = model(None, None, None, None, extras=extras, part_forward=1)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {video_id: result for video_id, result in zip(video_ids, output)}
        )
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Stage 2 time: {} ({} s / video per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    return results_dict



def compute_on_dataset(model, data_loader, device, logger, debug=False):
    model.eval()
    results_dict = compute_on_dataset_1stage(model, data_loader, device, debug=debug)
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    video_ids = list(sorted(predictions.keys()))
    if len(video_ids) != video_ids[-1] + 1:
        logger = logging.getLogger("alphaction.inference")
        logger.warning(
            "Number of videos that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in video_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        output_folder=None,
        metric='frame_ap',
        use_cache=False
):
    # convert to a torch.device for efficiency
    device = torch.device("cuda")
    num_devices = get_world_size()
    logger = logging.getLogger("alphaction.inference")
    dataset = data_loader.dataset

    pred_file = os.path.join(output_folder, "predictions.pth")
    if use_cache and os.path.exists(pred_file):
        if not is_main_process():
            return
        logger.info("Loading the prediction results on {} dataset({} videos).".format(dataset_name, len(dataset)))
        predictions = torch.load(pred_file)
    
    else:
        logger.info("Start evaluation on {} dataset({} videos).".format(dataset_name, len(dataset)))
        start_time = time.time()
        predictions = compute_on_dataset(model, data_loader, device, logger, debug=('debug' in output_folder))
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / video per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    
        if not is_main_process():
            return

        if output_folder:
            torch.save(predictions, pred_file)

    return evaluate(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        metric=metric
    )