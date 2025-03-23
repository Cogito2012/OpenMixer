import torch
from torchvision.ops import box_convert
from video_io import *
from tqdm import tqdm


def preprocess_clip(clip, device):
    """ clip: (T, H, W, 3) in uint8 format
    """
    # preprocess video
    clip = torch.from_numpy(clip).to(device).float() / 255.0
    clip = clip.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    return clip


def maskrcnn_video(video_path, model, categories, box_thresh=0.35, topk=None, batch_size=16, fmt='%05d.png', device=torch.device('cuda')):
    # load video data
    if isinstance(video_path, list):
        video = read_video_from_list(video_path)
        video_name = os.path.dirname(video_path[0]).split("/")[-1]
    elif os.path.isfile(video_path):
        video = read_video_from_file(video_path)  # (T, H, W, C) in RGB uint8 format
        video_name = video_path.split("/")[-1][:-4]
    else:
        video = read_video_from_folder(video_path, fmt=fmt)
        video_name = video_path.split("/")[-1]
    num_frames = len(video)

    if isinstance(video_path, list):
        frame_ids = [int(imgfile[:-4].split("/")[-1].split("_")[-1])
                     for imgfile in video_path]
    else:
        frame_ids = list(range(num_frames))
    
    if num_frames > batch_size:
        video = np.array_split(video, int(num_frames // batch_size))
        frame_ids = np.array_split(frame_ids, int(num_frames // batch_size))
    else:
        video, frame_ids = [video], [frame_ids]

    results = {'boxes': dict(), 'scores': dict()}
    for fids, clip in tqdm(zip(frame_ids, video), total=len(video), desc="{}".format(video_name), ncols=0):
        # preprocess
        height, width = clip.shape[1:3]
        batch = preprocess_clip(clip, device)  # (T, 3, H, W)
        with torch.no_grad():
            outputs = model(batch)
        # get results
        for i, outs in enumerate(outputs):
            mask = outs['labels'] == categories.index('person')
            if not any(mask):
                continue  # no person at all

            if box_thresh is not None:
                mask = mask & (outs['scores'] > box_thresh)
            if topk is not None:
                inds = torch.topk(outs['scores'], topk)[1]
                topk_mask = torch.zeros_like(outs['scores'], dtype=torch.bool).scatter_(0, inds, True)
                mask = mask & topk_mask
            if not any(mask):  # no valid person
                continue
            
            # mask out and sort boxes and scores
            boxes = outs['boxes'][mask]  # the predicted boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
            scores = outs['scores'][mask]
            idx = torch.argsort(scores, descending=True)
            boxes, scores = boxes[idx], scores[idx]
            # save
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / height
            results['boxes'][fids[i]] = boxes.cpu().numpy()  # normalized (x1, y1, x2, y2)
            results['scores'][fids[i]] = scores.cpu().numpy()

    return results
