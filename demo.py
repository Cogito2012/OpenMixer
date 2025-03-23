import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2 
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from alphaction.config import cfg
from alphaction.dataset import make_data_loader
import alphaction.dataset.datasets.utils as utils
from alphaction.dataset.datasets.evaluation.jhmdb.jhmdb_eval import prepare_preds_ind as prepare_jhmdb_preds
from alphaction.dataset.datasets.evaluation.ucf24.ucf24_eval import prepare_preds_ind as prepare_ucf24_preds
from alphaction.dataset.datasets.evaluation.ucf24.ucf24_eval import prepare_gt as prepare_ucf24_gt
from alphaction.utils.visualize import annotate, video_to_gif
from alphaction.utils.random_seed import set_seed


def parse_dets(result, vocab):
    det_boxes = result['boxes']
    det_actions = result['action_ids']
    scores = result['scores']
    # get predictions
    pred_cls = det_actions[scores.argmax(axis=-1)]
    pred_action = vocab[pred_cls]
    pred_box = det_boxes[pred_cls]
    pred_score =  scores[pred_cls]
    return pred_box, pred_action, pred_score


def visualize_detections(video_path, gt_data, detections, vocab, outdir, num_per_cls=1, videos_to_save=[], img_fmt="{:0>5}.png", fps_vis=5):
    
    os.makedirs(outdir, exist_ok=True)

    for gt_label in gt_data['labels']:
        # randomly select num_per_cls videos for visualization
        all_vids = [vid for vid in gt_data['videos'] if gt_label in vid]
        if len(videos_to_save) > 0:
            vis_vids = [vid for vid in videos_to_save if vid.split('/')[0] == gt_label]
        else:
            vis_vids = np.random.choice(all_vids, num_per_cls).tolist()

        # get all detections & gts
        all_dets = {img_key: dets for img_key, dets in detections.items() if img_key.split(',')[0] in vis_vids}
        all_gts = [(vid, list(gts.keys())[0], list(gts.values())[0][0]) for vid, gts in gt_data['gttubes'].items() if vid in vis_vids]  # single tube for each video

        # visualize according to GT
        for vid, gt_action, box_tube in all_gts:
            
            vis_video = []
            # read video frames
            image_paths = [os.path.join(video_path, vid, img_fmt.format(int(frame))) for frame in box_tube[:, 0]]
            imgs = utils.retry_load_images(
                image_paths, backend='cv2'
            )  # a list of BGR images [h, w, c]

            frame_ids = []
            for frame, fid_box in zip(imgs, box_tube):
                # visualize GT boxes
                annotated_frame = annotate(image_source=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), boxes=fid_box[None, 1:5], normalized=False, phrases=[vid.split('/')[0]], is_xyxy=True, color=(255,255,0), set_text_color='black')
                # visualize detections
                img_key = "%s,%05d" % (vid, float(fid_box[0]))
                if img_key in all_dets:
                    pred_box, pred_action, pred_score = parse_dets(all_dets[img_key], vocab)
                    annotated_frame = annotate(image_source=cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), boxes=pred_box[None], normalized=False, logits=torch.tensor([pred_score]), phrases=[pred_action], is_xyxy=True, set_text_color='white', color=(0,0,255))

                    vis_video.append(annotated_frame)
                    frame_ids.append(int(fid_box[0]))

            gif_file = os.path.join(outdir, vid.replace("/", '-') + '_vis.gif')
            video_to_gif(vis_video, gif_file, fps=fps_vis, toBGR=True)

            if vid in videos_to_save:
                frame_path = os.path.join(outdir, vid.replace("/", '-'))
                os.makedirs(frame_path, exist_ok=True)
                for fid, frame in zip(frame_ids, vis_video):
                    cv2.imwrite(frame_path + '/{}.png'.format(fid), frame)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection Inference")
    parser.add_argument(
        "--config-file",
        default="config_files/jhmdb/openmixer_e2e.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Manual seed at the begining."
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    # Merge config file.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(args.seed, 0, 1)
    
    # get the test dataset
    data_loaders_test, vocabularies_test, _ = make_data_loader(cfg, is_train=False)
    dataset = data_loaders_test[0].dataset
    open_vocabulary = list(vocabularies_test[0].keys())
    context = cfg.MODEL.CLIPViP.CONTEXT_INIT
    
    # load the predictions
    inf_folder = "inference" if not cfg.TEST.SMALL_OPEN_WORLD else "inference_small"
    output_folder = os.path.join(cfg.OUTPUT_DIR, inf_folder, cfg.DATA.DATASETS[0])
    pred_file = os.path.join(output_folder, 'predictions.pth')
    assert os.path.exists(pred_file)
    predictions = torch.load(pred_file)
    
    vis_dir = os.path.join(output_folder, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # get detections
    ids_known = list(dataset.open_to_closed.keys())
    ids_unknown = list(dataset.open_to_unseen.keys())
    # tau_inv = np.exp(eval('cfg.MODEL.{}.LOGIT_SCALE_INIT'.format(cfg.MODEL.TEXT_ENCODER)))

    vocab_base = [open_vocabulary[open_id] for open_id in dataset.open_to_closed.keys()]
    vocab_novel = [open_vocabulary[open_id] for open_id in dataset.open_to_unseen.keys()]

    if cfg.DATA.DATASETS[0] == 'jhmdb':
        dets_base, dets_novel = prepare_jhmdb_preds(predictions, dataset)
        img_fmt = "{:0>5}.png"
        fps_vis = 5
    
    elif cfg.DATA.DATASETS[0] == 'ucf24':
        dets_base, dets_novel = prepare_ucf24_preds(predictions, dataset)
        # formatting GT to action tubes
        logger = logging.getLogger("alphaction.inference")
        dataset.data_known = prepare_ucf24_gt(dataset.data_known, logger)
        dataset.data_unknown = prepare_ucf24_gt(dataset.data_unknown, logger)
        img_fmt = "{:0>5}.jpg"
        fps_vis = 1

    videos_to_save = []

    # videos_to_save = ['brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0',
    #                   'catch/Torwarttraining_catch_u_cm_np1_ri_med_1',
    #                   'pick/Fishing_For_People_pick_f_cm_np1_ri_med_2',
    #                   'pour/Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_3',
    #                   'push/pushing_box_around_house_push_f_cm_np1_le_bad_3']
    videos_to_save = ['Biking/v_Biking_g04_c01',
                      'FloorGymnastics/v_FloorGymnastics_g01_c03',
                      'HorseRiding/v_HorseRiding_g03_c03',
                      'Surfing/v_Surfing_g01_c07',
                      'VolleyballSpiking/v_VolleyballSpiking_g04_c03']  # ucf24
    # visualize base classes
    visualize_detections(dataset.video_path, dataset.data_known, dets_base, vocab_base, outdir=os.path.join(vis_dir, 'base'), 
                         videos_to_save=videos_to_save, num_per_cls=2, img_fmt=img_fmt, fps_vis=fps_vis)

    # videos_to_save = ['kick_ball/FIFA_11_Gamescom-Trailer_kick_ball_f_cm_np1_ba_med_4',
    #                   'pullup/Super_Training_Pull_Ups_pullup_f_cm_np1_ba_med_1',
    #                   'shoot_bow/Shootingarecurve_shoot_bow_u_nm_np1_fr_med_4',
    #                   'golf-Cobra_Golf_-_Camilo_Villegas_golf_f_cm_np1_ri_goo_1']
    # videos_to_save = ['golf/Cobra_Golf_-_Camilo_Villegas_golf_f_cm_np1_ri_goo_1',
    #                   'shoot_bow/6arrowswithin30seconds_shoot_bow_f_nm_np1_fr_med_4',
    #                   'shoot_gun/MeShootin2_shoot_gun_u_nm_np1_ri_med_2',
    #                   'sit/NoCountryForOldMen_sit_f_nm_np1_ri_med_6',
    #                   'swing_baseball/HowtoswingaBaseballbat_swing_baseball_f_nm_np1_le_bad_0']
    videos_to_save = ['BasketballDunk/v_BasketballDunk_g03_c02',
                      'IceDancing/v_IceDancing_g05_c01',
                      'LongJump/v_LongJump_g01_c06',
                      'Skijet/v_Skijet_g04_c02',
                      'WalkingWithDog/v_WalkingWithDog_g01_c04']
    # visualize novel classes
    visualize_detections(dataset.video_path, dataset.data_unknown, dets_novel, vocab_novel, outdir=os.path.join(vis_dir, 'novel'), 
                         videos_to_save=videos_to_save, num_per_cls=2, img_fmt=img_fmt, fps_vis=fps_vis)
        
        

if __name__ == '__main__':
    
    main()