from pytorch_grad_cam import *

import torch
import numpy as np
import cv2
from hilacam import hilacam_clip, hilacam_clipvip
from ritsm import ritsm_clip, ritsm_clipvip
from mhsa import mhsa_clip, mhsa_clipvip


class CAMWrapper(object):
    CAM_LIST = ['gradcam', 'scorecam', 'gradcam++', 'ablationcam', 'xgradcam', 'eigencam', 'eigengradcam', 
                'layercam', 'hilacam', 'hilacam_clipvip', 'groupcam', 'sscam1', 'sscam2', 'rawcam', 'testhila', 
                'gradientcam', 'gscorecam', 'vitgradcam', 'rise', 'gscorecambeta', 'testcam', 'ritsm', 'ritsm_clipvip', 'mhsa', 'mhsa_clipvip']
    CAM_DICT = {"gradcam": GradCAM,    # consider as baseline
        "scorecam": ScoreCAM,         # Good but slow
        "gradcam++": GradCAMPlusPlus, # focus barely shift from different inputs
        "ablationcam": AblationCAM,   # slightly worse than scorecam but has similar computing complexity
        "xgradcam": XGradCAM,         # bad for CLIP
        "eigencam": EigenCAM,         # does not work for CLIP
        "eigengradcam": EigenGradCAM, # not good for CLIP
        "layercam": LayerCAM,         # sometimes very focus, sometimes very noisy
        "hilacam": hilacam_clip,
        "hilacam_clipvip": hilacam_clipvip,
        "groupcam": GroupCAM,         # slightly noiser than gradcam
        "sscam1": SSCAM1,             # doesn't work for clip, extremly slow
        "sscam2": SSCAM2,             # doesn't work for clip, extremly slow
        "testhila": HilaCAM,
        "gscorecam": GScoreCAM,
        "rise": RiseCAM,
        "gscorecambeta": GScoreCAMBeta,
        "testcam": TestCAM,
        "ritsm": ritsm_clip,
        "ritsm_clipvip": ritsm_clipvip,
        "mhsa": mhsa_clip,
        "mhsa_clipvip": mhsa_clipvip
    }
    def __init__(self, model, target_layers, tokenizer, cam_version, clip_version="ViT-B/32", preprocess=None, target_category=None, is_clip=True,
                 mute=False, cam_trans=None, is_transformer=False, attn_grad=True, **kwargs):
        """[summary]

        Args:
            model (model): [description]
            target_layers (model layer): List[layers]
            drop (bool, optional): [description]. Defaults to False.
            cam_version (str, optional): [description]. Defaults to 'gradcam'.
            target_category (int or tensor, optional): [description]. Defaults to None.
            mute (bool, optional): [description]. Defaults to False.
            channel_frame (csv, optional): [description]. Defaults to None.
            channels (int, optional): [description]. Defaults to None.
            cam_trans (function, optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        self.mute = mute
        self.model = model
        self.version = cam_version
        self.clip_version = clip_version
        self.target_layers = target_layers
        self.target_category = target_category
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.cam_trans = cam_trans
        self.is_transformer = is_transformer
        self.attn_grad = attn_grad
        self.is_clip = is_clip
        self.channels = None
        self.__dict__.update(kwargs)

        if self.version not in self.CAM_LIST:
            raise ValueError(f"CAM version not found. Please choose from: {self.CAM_LIST}")
        # define cam
        self._load_cam()
    
    def _select_channels(self, text):
        if self.channel_dict is not None and text in self.channel_dict.keys():
            return self.channel_dict[text][:self.topk]
        else:
            return None

    # load cam
    def _load_cam(self):
        if self.version in ['hilacam', 'hilacam_clipvip', 'ritsm', 'ritsm_clipvip', 'mhsa', 'mhsa_clipvip']:
            self.cam = self.CAM_DICT[self.version]
        elif self.version == 'testhila':
            target_layer = self.model.visual.attnpool
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, use_cuda=True, clip=self.is_clip , reshape_transform=self.cam_trans, hila=True)

        elif self.version in ['scorecam', 'gscorecam']:
            batch_size = self.batch_size if hasattr(self, "batch_size") else 128
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans, drop=self.drop, 
                                        mute=self.mute, channels=self.channels, topk=self.topk, batch_size=batch_size, is_transformer=self.is_transformer)
        elif self.version == 'groupcam':
            self.cam = self.CAM_DICT[self.version](self.model, self.target_layers[0], cluster_method='k_means', is_clip=self.is_clip)
        elif self.version == 'layercam':
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans)
        elif self.version.startswith('sscam'):
            self.cam = self.CAM_DICT[self.version](model=self.model, is_clip=self.is_clip)
        elif self.version == 'rise':
            img_size = self.dataset_size if hasattr(self, 'dataset_size') else (384, 384)
            mask_path = f'data/rise_mask_{img_size[0]}x{img_size[1]}.npy'
            self.cam = self.CAM_DICT[self.version](model=self.model, image_size=img_size, mask_path=mask_path, batch_size=64)
        else:
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans, is_transformer=self.is_transformer)

    def getCAM(self, input_img, input_text, cam_size, target_category, return_logits=False):
        cam_input = (input_img, input_text) if self.is_clip else input_img
        self.cam.img_size = cam_size
        if self.version in ['hilacam', 'hilacam_clipvip', 'ritsm', 'ritsm_clipvip']:
            grayscale_cam = self.cam(input_img, input_text, self.model, 'cuda', cam_size=cam_size, index=target_category, return_logits=return_logits, attn_grad=self.attn_grad)
        elif self.version in ['mhsa', 'mhsa_clipvip']:
            grayscale_cam = self.cam(input_img, self.model, cam_size=cam_size, threshold=0.6)
        elif self.version == 'groupcam':
            grayscale_cam = self.cam(cam_input, class_idx=target_category)
            grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0)
        elif self.version.startswith('sscam'):
            grayscale_cam = self.cam(input_img, input_text, class_idx=target_category, 
                                     param_n=35, mean=0, sigma=2, mute=self.mute)
        elif self.version == 'layercam':
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        elif self.version == 'rise':
            grayscale_cam = self.cam(inputs=cam_input, targets=target_category, image_size=cam_size)
        # elif self.version == 'lime':
        #     grayscale_cam = self.cam(inputs=cam_input, target=target_category, image_size=(224, 224), image=kwargs['image'])
        else:
            
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def __call__(self, inputs, label, heatmap_size):

        if isinstance(inputs, tuple):
            img, text = inputs[0], inputs[1]
        else:
            img = inputs
            text = None

        if self.preprocess is not None:
            img = self.preprocess(img)
        # tokenize text
        text_token = None if self.tokenizer is None else self.tokenizer(text).cuda()
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        if not img.is_cuda:
            img = img.cuda()
        if hasattr(self, "channel_dict"):
            # self.cam.channels = self.channel_dict[text]
            self.cam.channels = self._select_channels(text)

        return self.getCAM(img, text_token, heatmap_size, label)
    
    def getLogits(self, img, text):
        with torch.no_grad():
            if self.preprocess is not None:
                img = self.preprocess(img)
            img_per_text, text_per_img = self.model(img.unsqueeze(0).cuda(), self.tokenizer(text).cuda())
        return img_per_text, text_per_img


def get_heatmap_from_mask(
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap = np.float32(heatmap) / 255

    # if np.max(img) > 1:
    #     raise Exception("The input image should np.float32 in the range [0, 1]")

    # cam = heatmap + img
    # cam = cam / np.max(cam)
    return heatmap
