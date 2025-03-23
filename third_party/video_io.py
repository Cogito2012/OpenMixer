import cv2
import numpy as np
import os
import supervision as sv


def read_video_from_file(video_file, toRGB=True):
    assert os.path.exists(video_file), "File does not exist! {}".format(video_file)
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    video = []
    while success:
        if toRGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        success, frame = cap.read()
    video = np.array(video)
    return video


def read_video_from_folder(video_path, fmt='%05d.png', start_frame=1, toRGB=True):
    frame_files = [name for name in os.listdir(video_path) if name.endswith(fmt[-4:])]
    vid_name = video_path.split("/")[-1]
    video = []
    for i in range(len(frame_files)):
        if len(fmt.split("_")) == 1:
            frame_name = fmt%(i + start_frame)
        elif len(fmt.split("_")) == 2:
            frame_name = fmt%(vid_name, i + start_frame)
        frame_file = os.path.join(video_path, frame_name)  # frame starting from 1
        frame = cv2.imread(frame_file)
        if toRGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
    video = np.array(video)
    return video


def read_video_from_list(img_list, toRGB=True):
    video = []
    for frame_file in img_list:
        frame = cv2.imread(frame_file)
        if toRGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
    video = np.array(video)
    return video


def write_video(mat, video_file, fps=30, write_frames=True):
    """ mat: (T, H, W, C)
    """
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)
    
    if write_frames:
        os.makedirs(video_file[:-4], exist_ok=True)
        for i, frame in enumerate(mat):
            cv2.imwrite(os.path.join(video_file[:-4], '%06d.jpg'%(i)), frame)


def vis_dets(results, vid, video_dir, savefile):
    # read frames
    video = read_video_from_folder(os.path.join(video_dir, vid), toRGB=False)  # (T, H, W, C)
    # parse detections
    boxes = results[vid]['boxes']  # normalized (x1, y1, x2, y2)
    scores = results[vid]['scores']
    video_vis = []
    # visualize
    for i, frame in enumerate(video):
        h, w = frame.shape[:2]
        xyxy = boxes[i] * np.array([[w, h, w, h]])
        detections = sv.Detections(xyxy=xyxy, confidence=scores[i])
        labels = [f"person {s:.2f}" for s in scores[i]]
        # annotate on frame
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
        video_vis.append(annotated_frame)
    video_vis = np.array(video_vis)
    # visualize
    write_video(video_vis, savefile, fps=20, write_frames=False)