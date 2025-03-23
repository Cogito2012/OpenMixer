import os, argparse, pickle
from tqdm import tqdm

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models._meta import _COCO_CATEGORIES
from maskrcnn_utils import maskrcnn_video
from video_io import vis_dets

from eval_utils import eval_person_boxes, load_gt_data
from pprint import pformat
import matplotlib.pyplot as plt


def main(args):

    dataset = args.data.upper()
    if args.data == 'jhmdb':
        video_dir = f'../data/{dataset}/Frames'
        fmt = '%05d.png'
        box_thresh, topk = None, 1
    
    elif args.data == 'ucf24':
        video_dir = f'../data/{dataset}/rgb-images'
        fmt = '%05d.jpg'
        box_thresh, topk = 0.35, None
    
    else:
        raise NotImplemented
    results_save_file = f'../data/{dataset}/{dataset}-MaskRCNN.pkl'

    if not os.path.exists(results_save_file):
        # setup device and model
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
        model.eval()

        all_video_files = []
        for folder in os.listdir(video_dir):
            videos_class_path = os.path.join(video_dir, folder)
            if not os.path.isdir(videos_class_path):
                continue
            vid_files = [folder + '/' + vid for vid in os.listdir(videos_class_path) if os.path.isdir(os.path.join(videos_class_path, vid))]
            all_video_files.extend(vid_files)
        
        results = dict()
        for vid in tqdm(all_video_files, total=len(all_video_files), ncols=0):
            print("\nRuning on the file: {}...".format(vid))
            results[vid] = maskrcnn_video(os.path.join(video_dir, vid), model, _COCO_CATEGORIES,
                                          fmt=fmt, box_thresh=box_thresh, topk=topk, device=device)
    
        with open(results_save_file, 'wb') as fid:
            pickle.dump(results, fid, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        with open(results_save_file, 'rb') as fid:
            results = pickle.load(fid, encoding='iso-8859-1')

    # evaluation
    if args.eval:
        # load the ground truth
        jhmdb_gt_file = '../data/JHMDB/JHMDB-GT.pkl'
        gt_data = load_gt_data(jhmdb_gt_file)
        
        eval_res, precisions, recalls = eval_person_boxes(results, gt_data)

        print(pformat(eval_res, indent=2))

        plt.figure(figsize=(10, 6))
        plt.plot(recalls[0], precisions[0], label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig('../temp/jhmdb/precision_recall_curve_maskrcnn.png', bbox_inches='tight')
        plt.close()
    
    # visualize
    if args.vis:
        test_video = ['kick_ball/FIFA_11_Gamescom-Trailer_kick_ball_f_cm_np1_ba_med_4']
        save_dir = os.path.join(os.path.dirname(results_save_file), 'VisMaskRCNN')
        os.makedirs(save_dir, exist_ok=True)
        for vid in test_video:
            savefile = os.path.join(save_dir, vid.replace('/', '-') + "_pred.mp4")
            vis_dets(results, vid, video_dir, savefile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Mask RCNN (ResNet-50 FPN) Experiments")
    parser.add_argument(
        "--data", type=str, default='jhmdb', choices=['jhmdb', 'ucf24'], help="dataset used for testing",
    )
    parser.add_argument(
        "--vis", action='store_true', help="visualize the detection results",
    )
    parser.add_argument(
        "--eval", action='store_true', help="evaluate the quality"
    )
    args = parser.parse_args()
    
    main(args)