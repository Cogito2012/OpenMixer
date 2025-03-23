import logging
import torch
from tqdm import tqdm
import time
import datetime
from alphaction.utils.comm import get_rank, synchronize, get_world_size


def do_feature_extraction(model_ddp, data_loader, distributed):
    
    device = torch.device("cuda")
    num_devices = get_world_size()
    dataset = data_loader.dataset

    if dataset.finished_feat_extraction():
        return
    logger = logging.getLogger("alphaction.feature_extraction.{}".format(dataset._split))

    logger.info("Start feature extraction on {} dataset({} videos).".format(dataset.__class__.__name__, len(dataset)))
    start_time = time.time()
    model = model_ddp.module if distributed else model_ddp
    model.eval()

    extra_args = {} if get_world_size() == 1 else dict(desc="feature extracting", disable=(not get_rank()==0))
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), **extra_args):
            video, _, whwh, boxes, _, metadata, idx = batch
            video = video.to(device)

            # extract patch token features and CLS token feature
            features, cls_feat = model.backbone([video])
            # extract text features
            text_features = model.backbone.forward_text(device=device)

            # save torch tensors
            dataset.save_features(idx, features[0].cpu(), cls_feat.cpu(), text_features.cpu())
            
            if dataset.finished_feat_extraction():
                logger.info("Finished feature extraction. ")
                break  # check if all samples are processed
    
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info("Feature extraction time: {} ({} s / video per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices))
