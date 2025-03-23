import bisect
import copy
import torch.utils.data
from alphaction.utils.comm import get_world_size
from . import datasets as D
from . import samplers
from .collate_batch import BatchCollator

def build_dataset(cfg, split):
    if cfg.DATA.DATASETS[0] == 'ucf24':
        dataset = D.UCF24(cfg, split)
    elif cfg.DATA.DATASETS[0] == 'jhmdb':
        dataset = D.Jhmdb(cfg, split)
    elif cfg.DATA.DATASETS[0] == 'ava_v2.2':
        dataset = D.Ava(cfg, split)
    else:
        raise NotImplementedError

    return [dataset]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        video_info = dataset.get_video_info(i)
        aspect_ratio = float(video_info["height"]) / float(video_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, videos_per_batch, num_iters=None, start_iter=0, drop_last=False
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, videos_per_batch, drop_uneven=drop_last
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, videos_per_batch, drop_last=drop_last
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        # for training
        videos_per_batch = cfg.SOLVER.VIDEOS_PER_BATCH
        assert (
                videos_per_batch % num_gpus == 0
        ), "SOLVER.VIDEOS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(videos_per_batch, num_gpus)
        videos_per_gpu = videos_per_batch // num_gpus
        shuffle = True
        drop_last = True
        # num_iters = cfg.SOLVER.MAX_EPOCH*cfg.SOLVER.ITER_PER_EPOCH
        split = 'train'
    else:
        # for testing
        videos_per_batch = cfg.TEST.VIDEOS_PER_BATCH
        assert (
                videos_per_batch % num_gpus == 0
        ), "TEST.VIDEOS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(videos_per_batch, num_gpus)
        videos_per_gpu = videos_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        drop_last = False
        # num_iters = None
        start_iter = 0
        split = 'test'

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    # build dataset
    datasets = build_dataset(cfg, split=split)

    # build sampler and dataloader
    data_loaders, vocabularies, iter_per_epoch_all = [], [], []
    for dataset in datasets:
        if is_train:
            # number of iterations for all epochs
            iter_per_epoch = int(len(dataset) // cfg.SOLVER.VIDEOS_PER_BATCH) if cfg.SOLVER.ITER_PER_EPOCH == -1 else cfg.SOLVER.ITER_PER_EPOCH
            iter_per_epoch_all.append(iter_per_epoch)
        num_iters = cfg.SOLVER.MAX_EPOCH * iter_per_epoch if is_train else None
        # sampler
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, videos_per_gpu, num_iters, start_iter, drop_last
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
        if cfg.DATA.OPEN_VOCABULARY:
            vocabularies.append(dataset.text_input)
        else:
            vocabularies.append(None)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0], vocabularies[0]['closed'], iter_per_epoch_all[0]
    
    vocabularies_val = []
    if len(vocabularies) > 0:
        for vocab in vocabularies:
            if cfg.TEST.EVAL_OPEN and vocab is not None:
                vocabularies_val.append(vocab['open'])
            else:
                vocabularies_val.append(vocab['closed'])
    
    return data_loaders, vocabularies_val, iter_per_epoch_all