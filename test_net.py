import argparse
import os

import torch
from alphaction.config import cfg
from alphaction.dataset import make_data_loader
from alphaction.engine.inference import inference
from alphaction.modeling.detector import build_detection_model, build_naive_baseline
from alphaction.utils.checkpoint import ActionCheckpointer
from torch.utils.collect_env import get_pretty_env_info
from alphaction.utils.comm import synchronize, get_rank
from alphaction.utils.logger import setup_logger
#pytorch issuse #973
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    # Merge config file.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    # Print experimental infos.
    save_dir = ""
    logger = setup_logger("alphaction", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + get_pretty_env_info())

    # Build the model.
    if cfg.MODEL.DET == 'STMDetector':
        model = build_detection_model(cfg)
    elif cfg.MODEL.DET == 'NaiveBaseline':
        model = build_naive_baseline(cfg)
    model.to("cuda")

    if cfg.MODEL.DET != 'NaiveBaseline':
        # load weight.
        output_dir = cfg.OUTPUT_DIR
        checkpointer = ActionCheckpointer(cfg, model, save_dir=output_dir)
        ckpt_file = os.path.join(output_dir, cfg.MODEL.WEIGHT) if cfg.MODEL.WEIGHT else None
        checkpointer.load(ckpt_file)

    output_folders = [None] * len(cfg.DATA.DATASETS)
    dataset_names = cfg.DATA.DATASETS
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            inf_folder = "inference" if not cfg.TEST.SMALL_OPEN_WORLD else "inference_small"
            output_folder = os.path.join(cfg.OUTPUT_DIR, inf_folder, dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder

    # Do inference.
    data_loaders_test, vocabularies_test, _ = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for i, (output_folder, dataset_name, data_loader_test) in enumerate(zip(output_folders, dataset_names, data_loaders_test)):
        # set open vocabulary
        if len(vocabularies_test) > 0:
            model.backbone.text_encoder.set_vocabulary(vocabularies_test[i])
        
        inference(
            model,
            data_loader_test,
            dataset_name,
            output_folder=output_folder,
            metric=cfg.TEST.METRIC,
            use_cache=True
        )
        synchronize()


if __name__ == "__main__":
    main()
