from typing import Tuple, List
import torch
from torchvision.ops import box_convert
import numpy as np
import supervision as sv
import cv2
import imageio


def annotate(image_source: np.ndarray, boxes: torch.Tensor, normalized=True, logits=None, phrases=[], is_xyxy=False, color=None, text_padding=10, set_text_color='black') -> np.ndarray:
    h, w, _ = image_source.shape
    if normalized:
        boxes = boxes * torch.Tensor([w, h, w, h])
    if not is_xyxy:
        assert isinstance(boxes, torch.Tensor)
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    elif isinstance(boxes, torch.Tensor):
        xyxy = boxes.numpy()
    else:
        xyxy = boxes
    detections = sv.Detections(xyxy=xyxy)

    if logits is not None and len(phrases) == logits.size(0):
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
    else:
        labels = phrases

    if color is None or (not isinstance(color, tuple)):
        svcolor = sv.ColorPalette.default()
    else:
        svcolor = sv.Color(*color)
    text_color = sv.Color.white() if set_text_color == 'white' else sv.Color.black()
    box_annotator = sv.BoxAnnotator(color=svcolor, text_padding=text_padding, text_color=text_color)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def video_to_gif(video, giffile, fps=5.0, toBGR=False):
    assert giffile.endswith('.gif')
    with imageio.get_writer(giffile, mode='I', duration=1.0/fps, loop=0) as writer:
        for frame in video:
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if toBGR else np.copy(frame)
            writer.append_data(frame_vis)