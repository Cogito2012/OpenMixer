r"""
Basic training script for PyTorch
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.collect_env import get_pretty_env_info
from alphaction.config import cfg
from alphaction.dataset import make_data_loader
from alphaction.solver import make_lr_scheduler, make_optimizer
from alphaction.engine.inference import inference
from alphaction.engine.trainer import do_train
from alphaction.engine.feature_extraction import do_feature_extraction
from alphaction.modeling.detector import build_detection_model
from alphaction.utils.checkpoint import ActionCheckpointer
from alphaction.utils.comm import synchronize, get_rank, setup_distributed
from alphaction.utils.logger import setup_logger, setup_tblogger
from alphaction.utils.random_seed import set_seed
from alphaction.structures.memory_pool import MemoryPool
# pytorch issuse #973
import resource
import tensorboardX


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


def train(cfg, local_rank, distributed, tblogger=None, transfer_weight=False, adjust_lr=False, skip_val=False,
          no_head=False):
    # build the model.
    model = build_detection_model(cfg)

    device = torch.device("cuda")
    model.to(device)

    # make solver.
    optimizer = make_optimizer(cfg, model)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )

    output_dir = cfg.OUTPUT_DIR
    inf_folder = "inference" if not cfg.TEST.SMALL_OPEN_WORLD else "inference_small"
    output_folder = os.path.join(cfg.OUTPUT_DIR, inf_folder)
    os.makedirs(output_folder, exist_ok=True)

    # load weight.
    save_to_disk = get_rank() == 0

    arguments = {}
    arguments["iteration"] = 0
    arguments["person_pool"] = MemoryPool()

    # make dataloader.
    data_loader, vocabulary_train, iter_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments['iteration'],
    )

    scheduler = make_lr_scheduler(cfg, optimizer, iter_per_epoch)
    checkpointer = ActionCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk, topk=1)
    ckpt_file = os.path.join(output_dir, cfg.MODEL.WEIGHT) if cfg.MODEL.WEIGHT else None
    extra_checkpoint_data = checkpointer.load(ckpt_file, model_weight_only=transfer_weight,
                                              adjust_scheduler=adjust_lr, no_head=no_head)
    arguments.update(extra_checkpoint_data)
    
    if cfg.DATA.OPEN_VOCABULARY:
        model_this = model.module if distributed else model
        model_this.backbone.text_encoder.set_vocabulary(vocabulary_train)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD * iter_per_epoch if 'debug' not in output_dir else 1
    val_after = cfg.SOLVER.EVAL_AFTER * iter_per_epoch if 'debug' not in output_dir else 0
    val_period = cfg.SOLVER.EVAL_PERIOD * iter_per_epoch if 'debug' not in output_dir else 1

    frozen_backbone_bn = ('vit' not in cfg.MODEL.BACKBONE.CONV_BODY.lower()) and cfg.MODEL.BACKBONE.FROZEN_BN

    # make validation dataloader if necessary
    if not skip_val:
        dataset_names_val = cfg.DATA.DATASETS
        data_loaders_val, vocabularies_val, _ = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    else:
        dataset_names_val = []
        data_loaders_val = []
        vocabularies_val = []
    
    trainable_bb_params = [name for name, p in model.named_parameters() if p.requires_grad and 'backbone' in name]
    if cfg.MODEL.PRE_EXTRACT_FEAT and len(trainable_bb_params) == 0:
        do_feature_extraction(model, data_loader, distributed)
        do_feature_extraction(model, data_loaders_val[0], distributed)
    
    # training
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        tblogger,
        val_after,
        val_period,
        dataset_names_val,
        data_loaders_val,
        vocabularies_val,
        distributed,
        frozen_backbone_bn,
        output_folder,
        metric=cfg.TEST.METRIC
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()

    output_folders = [None] * len(cfg.DATA.DATASETS)
    dataset_names = cfg.DATA.DATASETS
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            inf_folder = "inference" if not cfg.TEST.SMALL_OPEN_WORLD else "inference_small"
            output_folder = os.path.join(cfg.OUTPUT_DIR, inf_folder, dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder
    # make test dataloader.
    data_loaders_test, vocabularies_val, _ = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    
    # test for each dataset.
    for i, (output_folder, dataset_name, data_loader_test) in enumerate(zip(output_folders, dataset_names, data_loaders_test)):
        # set open vocabulary
        if len(vocabularies_val) > 0:
            model.backbone.text_encoder.set_vocabulary(vocabularies_val[i])
        
        inference(
            model,
            data_loader_test,
            dataset_name,
            output_folder=output_folder,
            metric=cfg.TEST.METRIC
        )
        synchronize()


def main():
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
        "--set_split",
        type=int,
        default=-1,
        help='Set the dataset split'
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        setup_distributed(args)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    global_rank = get_rank()

    # Merge config.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.set_split >= 0:
        # input
        if cfg.DATA.DATASETS[0] == 'jhmdb':
            cfg.JHMDB.CW_SPLIT_FILE = cfg.JHMDB.CW_SPLIT_FILE.format(args.set_split)
            cfg.JHMDB.OW_SPLIT_FILE = cfg.JHMDB.OW_SPLIT_FILE.format(args.set_split)
        elif cfg.DATA.DATASETS[0] == 'ucf24':
            cfg.UCF24.CW_SPLIT_FILE = cfg.UCF24.CW_SPLIT_FILE.format(args.set_split)
            cfg.UCF24.OW_SPLIT_FILE = cfg.UCF24.OW_SPLIT_FILE.format(args.set_split)
        # output
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.format(args.set_split)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Print experimental infos.
    logger = setup_logger("alphaction", output_dir, global_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + get_pretty_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    tblogger = None
    if args.tfboard:
        tblogger = setup_tblogger(output_dir, global_rank)

    set_seed(args.seed, global_rank, num_gpus)

    # do training.
    model = train(cfg, args.local_rank, args.distributed, tblogger, args.transfer_weight, args.adjust_lr, args.skip_val,
                  args.no_head)

    if tblogger is not None:
        tblogger.close()

    # do final testing.
    if not args.skip_test:
        run_test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
