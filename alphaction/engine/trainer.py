import datetime
import logging
import time

import torch

from alphaction.utils.metric_logger import MetricLogger
from alphaction.engine.inference import inference
from alphaction.utils.comm import synchronize, reduce_dict, all_gather
from alphaction.structures.memory_pool import MemoryPool
import torch.nn as nn
import copy


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        tblogger,
        start_val,
        val_period,
        dataset_names_val,
        data_loaders_val,
        vocabularies_val,
        distributed,
        frozen_backbone_bn,
        output_folder,
        metric='frame_ap'
):
    logger = logging.getLogger("alphaction.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    person_pool = arguments["person_pool"]
    model.train()
    if frozen_backbone_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
    start_training_time = time.time()
    end = time.time()
    losses_reduced = torch.tensor(0.0)
    
    extras = {'prior_map': data_loader.dataset.prior_map.to(device)} if data_loader.dataset.use_prior_map else {}

    for iteration, (slow_video, fast_video, whwh, boxes, labels, metadata, idx) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        slow_video = slow_video.to(device)
        if fast_video is not None:
            fast_video = fast_video.to(device)
        whwh = whwh.to(device)
        
        if data_loader.dataset.prior_boxes_init == 'gt':
            extras.update({'prior_boxes': boxes})
        elif data_loader.dataset.prior_boxes_init in ['det', 'rand']:
            extras.update({'prior_boxes': [info['extra_boxes'] for info in metadata]})
        
        loss_dict  = model(slow_video, fast_video, whwh, boxes, labels, extras)
        losses = sum(loss_dict.values()) / len(loss_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict["total_loss"] = losses.detach().clone()
        loss_dict_reduced = reduce_dict(loss_dict)

        meters.update(**loss_dict_reduced)
        losses_reduced = loss_dict_reduced.pop("total_loss")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            meter_info = ''.join([k.strip() + '), ' for k in str(meters).split(") ") 
                                  if not k.split(':')[0].split('_')[-1].isdigit()])  # remove redundent info
            logger.info(
                meters.delimiter.join(["eta: {eta}", "iter: {iter}/{max_iter}", "{meters}", "lr: {lr:.9f}",]).format(
                    eta=eta_string, iter=iteration, max_iter=max_iter, meters=meter_info, lr=optimizer.param_groups[0]["lr"]
                ))
            if tblogger is not None:
                for name, meter in meters.meters.items():
                    tblogger.add_scalar(name, meter.median, iteration)
                tblogger.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)

        scheduler.step()

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if iteration == max_iter:
            arguments.pop("person_pool", None)
            checkpointer.save("model_final", **arguments)

        if dataset_names_val and iteration > start_val and iteration % val_period == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # do validation
            val_in_train(
                model,
                dataset_names_val,
                data_loaders_val,
                vocabularies_val,
                tblogger,
                iteration,
                distributed,
                output_folder,
                metric=metric,
            )
            model.train()
            if frozen_backbone_bn:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
            torch.cuda.empty_cache()
            end = time.time()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def backup_train_vocab(text_encoder):
    # backup vocabulary
    vocabulary_train = copy.deepcopy(text_encoder.text_data)

    # backup embeddings
    class_embeddings_train = None
    if hasattr(text_encoder, 'class_embedding'):  # defined in CLIP backbone.text_encoder
        class_embeddings_train = text_encoder.class_embedding.clone()

    elif hasattr(text_encoder, 'vocab_token_embeddings'):  # defined in CLIPViP backbone.text_encoder
        class_embeddings_train = copy.deepcopy(text_encoder.vocab_token_embeddings)

    return vocabulary_train, class_embeddings_train


def val_in_train(
        model,
        dataset_names_val,
        data_loaders_val,
        text_val,
        tblogger,
        iteration,
        distributed,
        output_folder,
        metric='frame_ap'
):
    if distributed:
        model_val = model.module
    else:
        model_val = model
    for i, (dataset_name, data_loader_val) in enumerate(zip(dataset_names_val, data_loaders_val)):
        # set open vocabulary
        if len(text_val) > 0:
            text_train, class_embeddings_train = backup_train_vocab(model_val.backbone.text_encoder)
            # reset vocabulary
            model_val.backbone.text_encoder.set_vocabulary(text_val[i])

        # inference
        eval_res = inference(
            model_val,
            data_loader_val,
            dataset_name,
            output_folder=output_folder,
            metric=metric
        )
        synchronize()
        if tblogger is not None:
            eval_res, _ = eval_res
            iou_thresh = data_loader_val.dataset.test_iou_thresh
            key = 'PascalBoxes_Precision/mAP@{}IOU'.format(iou_thresh)
            total_mAP = eval_res[key]
            tblogger.add_scalar(dataset_name + '_mAP_{}IOU'.format(iou_thresh), total_mAP, iteration)
            if data_loader_val.dataset.open_vocabulary and data_loader_val.dataset.eval_open:
                tblogger.add_scalar(dataset_name + '_mAP_{}IOU(base)'.format(iou_thresh), eval_res['{}(base)'.format(key)], iteration)
                tblogger.add_scalar(dataset_name + '_mAP_{}IOU(novel)'.format(iou_thresh), eval_res['{}(novel)'.format(key)], iteration)

        # after evaluation, restore the training vocabulary
        if len(text_val) > 0:
            model_val.backbone.text_encoder.set_vocabulary(text_train, embeddings=class_embeddings_train)