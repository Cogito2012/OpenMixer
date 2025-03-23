from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DATA = CN()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 4

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 16

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3]

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

_C.DATA.TRAIN_MIN_SCALES = [256,320] # list
_C.DATA.TRAIN_MAX_SCALE = 1333 # int
_C.DATA.TEST_MIN_SCALES = [256] # list
_C.DATA.TEST_MAX_SCALE = 1333 # int

_C.DATA.FIX_SIZE = []

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

_C.DATA.DATASETS = ['ava_v2.2']

_C.DATA.OPEN_VOCABULARY = False

_C.DATA.REFINE_VOCAB = False

# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CN()

# Directory path of frames.
_C.AVA.FRAME_DIR = "frames/"

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = "frame_lists/"


# Directory path for annotation files.
_C.AVA.ANNOTATION_DIR = "annotations_v2.2/"

# Directory path for closed-world splits
_C.AVA.CW_SPLIT_DIR = ""

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TEST_GT_BOX_LISTS = ["ava_val_v2.2.csv"]
_C.AVA.VOCAB_REFINE = 'vocab_refine.json'

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.AVA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.AVA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

_C.AVA.STRICT_EVAL = True

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

_C.AVA.SINGLE_PROMPT = False

_C.KINETICS = CN()

# Directory path of k700 videos.
_C.KINETICS.VIDEO_DIR = ""

# Directory path for files of frame lists.
_C.KINETICS.FRAME_LIST_DIR = "kinetics/frame_lists/"

# Directory path for annotation files.
_C.KINETICS.ANNOTATION_DIR = "kinetics/annotations/"

# Filenames of training samples list files.
_C.KINETICS.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.KINETICS.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.KINETICS.TRAIN_GT_BOX_LISTS = ["kinetics_train_curated.csv"]
_C.KINETICS.TEST_GT_BOX_LISTS = ["kinetics_val_curated.csv"]

_C.AK = CN()
_C.AK.TEST_GT_BOX_LISTS = ["kinetics/ak_val_gt.csv"]


# -----------------------------------------------------------------------------
# JHMDB Dataset options
# -----------------------------------------------------------------------------
_C.JHMDB = CN()

# Directory path of frames.
_C.JHMDB.FRAME_DIR = "Frames/"

# Directory path of optical flows.
_C.JHMDB.FLOW_DIR = "FlowBrox04/"

# File path of annotations.
_C.JHMDB.GROUND_TRUTH_FILE = "JHMDB-GT.pkl"

# Directory path of Open-World splits
_C.JHMDB.OPEN_WORLD_DIR = 'openworld'

# seen/unseen class split
_C.JHMDB.CW_SPLIT_FILE = 'train50%/closed_world_0.pkl'
_C.JHMDB.OW_SPLIT_FILE = 'train50%/open_world_0.pkl'
_C.JHMDB.VOCAB_REFINE = 'vocab_refine.json'
_C.JHMDB.VOCAB_HN = '' # hard negative texts
_C.JHMDB.PRIOR_BOX_FILE = ''

# sample list split
_C.JHMDB.SAMPLES_SPLIT = 0  # [0, 1, 2]

# -----------------------------------------------------------------------------
# JHMDB Dataset Preprocess
# -----------------------------------------------------------------------------
# If use BGR as the format of input frames.
_C.JHMDB.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.JHMDB.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.JHMDB.TRAIN_PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.JHMDB.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.JHMDB.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.JHMDB.TEST_FORCE_FLIP = False

# _C.JHMDB.STRICT_EVAL = True

# -----------------------------------------------------------------------------
# UCF24 Dataset options
# -----------------------------------------------------------------------------
_C.UCF24 = CN()

# Directory path of frames.
_C.UCF24.FRAME_DIR = "rgb-images"

# Directory path of optical flows.
_C.UCF24.FLOW_DIR = "brox-images"

# Directory path of Open-World splits
_C.UCF24.OPEN_WORLD_DIR = 'openworld'

# seen/unseen class split
_C.UCF24.CW_SPLIT_FILE = 'train50%/closed_world_0.pkl'
_C.UCF24.OW_SPLIT_FILE = 'train50%/open_world_0.pkl'
_C.UCF24.VOCAB_REFINE = 'vocab_gpt4.json'
_C.UCF24.SINGLE_PROMPT = False
_C.UCF24.PRIOR_BOX_FILE = ''

# -----------------------------------------------------------------------------
# UCF24 Dataset Preprocess
# -----------------------------------------------------------------------------
# If use BGR as the format of input frames.
_C.UCF24.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.UCF24.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.UCF24.TRAIN_PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.UCF24.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.UCF24.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.UCF24.TEST_FORCE_FLIP = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of dataset loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 32
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False


_C.MODEL = CN()

_C.MODEL.WEIGHT = ""
_C.MODEL.DET = "STMDetector"
_C.MODEL.MULTI_LABEL_ACTION = True  # AVA actions are multi-label
_C.MODEL.PRE_EXTRACT_FEAT = False
_C.MODEL.USE_ROI_FEAT = False

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# The backbone conv body to use
# Available backbone conv-body should be registered in modeling.backbone.backbone.py
_C.MODEL.BACKBONE.CONV_BODY = "Slowfast-Resnet50"
_C.MODEL.BACKBONE.PATHWAYS = 1
_C.MODEL.BACKBONE.FROZEN_BN = False
_C.MODEL.BACKBONE.FREEZE_PRETRAINED_VISUAL = True
_C.MODEL.BACKBONE.FREEZE_PRETRAINED_TEXT = True
_C.MODEL.BACKBONE.RESIDUAL_LATERAL = False

# For alphaction backbones
_C.MODEL.BACKBONE.BN_MOMENTUM = 0.1
_C.MODEL.BACKBONE.BN_EPSILON = 1e-05

# Kaiming:
# We may use 0 to initialize the residual branch of a residual block,
# so the inital state of the block is exactly identiy. This helps optimizaiton.
_C.MODEL.BACKBONE.BN_INIT_GAMMA = 0.0

_C.MODEL.BACKBONE.I3D = CN()
_C.MODEL.BACKBONE.I3D.CONV3_NONLOCAL = True
_C.MODEL.BACKBONE.I3D.CONV4_NONLOCAL = True
_C.MODEL.BACKBONE.I3D.CONV3_GROUP_NL = False

# ---------------------------------------------------------------------------- #
# Slowfast options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE.SLOWFAST = CN()
_C.MODEL.BACKBONE.SLOWFAST.BETA = 1./8
_C.MODEL.BACKBONE.SLOWFAST.LATERAL = 'tconv'
_C.MODEL.BACKBONE.SLOWFAST.SLOW = CN()
_C.MODEL.BACKBONE.SLOWFAST.SLOW.ACTIVE = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV3_NONLOCAL = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV4_NONLOCAL = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV3_GROUP_NL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST = CN()
_C.MODEL.BACKBONE.SLOWFAST.FAST.ACTIVE = True
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV3_NONLOCAL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV4_NONLOCAL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV3_GROUP_NL = False

# ---------------------------------------------------------------------------- #
# Nonlocal options
# ---------------------------------------------------------------------------- #
_C.MODEL.NONLOCAL = CN()
_C.MODEL.NONLOCAL.CONV_INIT_STD = 0.01
_C.MODEL.NONLOCAL.USE_ZERO_INIT_CONV = False
_C.MODEL.NONLOCAL.NO_BIAS = False
_C.MODEL.NONLOCAL.USE_MAXPOOL = True
_C.MODEL.NONLOCAL.USE_SOFTMAX = True
_C.MODEL.NONLOCAL.USE_SCALE = True

_C.MODEL.NONLOCAL.USE_BN = True
_C.MODEL.NONLOCAL.FROZEN_BN = False

_C.MODEL.NONLOCAL.BN_MOMENTUM = 0.1
_C.MODEL.NONLOCAL.BN_EPSILON = 1e-05
_C.MODEL.NONLOCAL.BN_INIT_GAMMA = 0.0


# For PySlowFast backbones
_C.RESNET = CN()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# Whether use modulated DCN on Res2, Res3, Res4, Res5 or not
_C.RESNET.DEFORM_ON_PER_STAGE = [False, False, False, False]

_C.NONLOCAL = CN()
# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

_C.SLOWFAST = CN()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


_C.ViT = CN()
_C.ViT.TUBELET_SIZE = 2
_C.ViT.PATCH_SIZE = 16
_C.ViT.IN_CHANS = 3
_C.ViT.EMBED_DIM = 768
_C.ViT.PRETRAIN_IMG_SIZE = 224
_C.ViT.USE_LEARNABLE_POS_EMB = False
_C.ViT.DROP_RATE = 0.
_C.ViT.ATTN_DROP_RATE = 0.
_C.ViT.DROP_PATH_RATE = 0.2  #
_C.ViT.DEPTH = 12
_C.ViT.NUM_HEADS = 12
_C.ViT.MLP_RATIO = 4
_C.ViT.QKV_BIAS = True
_C.ViT.QK_SCALE = None
_C.ViT.INIT_VALUES = 0.
_C.ViT.USE_CHECKPOINT = True
_C.ViT.LAYER_DECAY = 0.75
_C.ViT.WEIGHT_DECAY = 0.05
_C.ViT.NO_WEIGHT_DECAY = ['pos_embed']


# STMixer
_C.MODEL.STM = CN()
_C.MODEL.STM.NUM_QUERIES = 100
_C.MODEL.STM.HIDDEN_DIM = 256
_C.MODEL.STM.NUM_STAGES = 6
_C.MODEL.STM.ACTION_CLASSES = 80
_C.MODEL.STM.OBJECT_CLASSES = 1
_C.MODEL.STM.NUM_HEADS = 8
_C.MODEL.STM.DROPOUT = 0.0
_C.MODEL.STM.DIM_FEEDFORWARD = 2048
_C.MODEL.STM.NUM_FCS = 2
_C.MODEL.STM.ACTIVATION = 'ReLU'
_C.MODEL.STM.SPATIAL_POINTS = 32
_C.MODEL.STM.TEMPORAL_POINTS = 4
_C.MODEL.STM.OUT_MULTIPLIER = 4
_C.MODEL.STM.N_GROUPS = 4
_C.MODEL.STM.NUM_CLS = 1
_C.MODEL.STM.NUM_ACT = 1
_C.MODEL.STM.NUM_REG = 1
_C.MODEL.STM.OBJECT_WEIGHT = 2.0
_C.MODEL.STM.ACTION_WEIGHT = 24.0
_C.MODEL.STM.GIOU_WEIGHT = 2.0
_C.MODEL.STM.L1_WEIGHT = 2.0
_C.MODEL.STM.BACKGROUND_WEIGHT = 0.1
_C.MODEL.STM.FOCAL_WEIGHT = 0.0
_C.MODEL.STM.INTERMEDIATE_SUPERVISION = True
_C.MODEL.STM.PERSON_THRESHOLD = 0.6
_C.MODEL.STM.USE_CLS_FEAT = False
_C.MODEL.STM.COND_CLS = False
_C.MODEL.STM.COND_MODALITY = 'visual'
_C.MODEL.STM.FUSE_CLS = False
_C.MODEL.STM.FUSE_METHOD = ''
_C.MODEL.STM.FUSE_FACTOR = -1.0
_C.MODEL.STM.CHN_WEIGHT = 1.0

_C.MODEL.STM.PRETRAIN_ACTION = False
_C.MODEL.STM.DeST = False
_C.MODEL.STM.FS_GAMMA = 0.0

_C.MODEL.TEXT_ENCODER = ""  # CLIP or CLIPViP
_C.MODEL.USE_PRIOR_MAP = False
_C.MODEL.PRIOR_BOXES_INIT = ''

_C.MODEL.FOCAL_LOSS = CN()
_C.MODEL.FOCAL_LOSS.ALPHA = 0.25
_C.MODEL.FOCAL_LOSS.GAMMA = 2.0


# CLIP
_C.MODEL.CLIP = CN()
_C.MODEL.CLIP.ARCH = 'ViT-B/16'
_C.MODEL.CLIP.LEN_CONTEXT_CLS = 24
_C.MODEL.CLIP.LEN_CONTEXT_PROMPT = 8
_C.MODEL.CLIP.CONTEXT_INIT = 'a photo of '
_C.MODEL.CLIP.SOFT_PROMPT = True
_C.MODEL.CLIP.ENABLE_POS_EMBED = True
_C.MODEL.CLIP.EMBED_DIM = 768
_C.MODEL.CLIP.USE_CHECKPOINT = True
_C.MODEL.CLIP.FREEZE_TEXT_BACKBONE = True
_C.MODEL.CLIP.CAM_SAMPLING = 'topk'
_C.MODEL.CLIP.CAM_METHOD = ''
_C.MODEL.CLIP.USE_PERSON_EMBED = False


# CLIP-ViP
_C.MODEL.CLIPViP = CN()
_C.MODEL.CLIPViP.ARCH = 'ViP-B/16'
_C.MODEL.CLIPViP.CLIP_NAME = "openai/clip-vit-base-patch16"  # load from huggingface
_C.MODEL.CLIPViP.WEIGHT = "pretrained/pretrain_clipvip_base_16.pt"
_C.MODEL.CLIPViP.TEMPORAL_SIZE = 12
_C.MODEL.CLIPViP.TEMPORAL_WINSIZE = 1
_C.MODEL.CLIPViP.USE_TEMPORAL_EMBED = True
_C.MODEL.CLIPViP.LOGIT_SCALE_INIT = 4.6
_C.MODEL.CLIPViP.ADD_CLS_NUM = 3
_C.MODEL.CLIPViP.CONTEXT_INIT = 'a video of '
_C.MODEL.CLIPViP.LEN_CONTEXT = 8
_C.MODEL.CLIPViP.EMBED_DIM = 512
_C.MODEL.CLIPViP.FREEZE_TEXT_BACKBONE = True
_C.MODEL.CLIPViP.SOFT_PROMPT = False
_C.MODEL.CLIPViP.ST_CROSS_ATTN = False
_C.MODEL.CLIPViP.NUM_XATTN = 1
_C.MODEL.CLIPViP.FINETUNE_ADDED_CLS = False
_C.MODEL.CLIPViP.USE_ATTN = False
_C.MODEL.CLIPViP.USE_GRAD = False
_C.MODEL.CLIPViP.USE_ATTN_LAST = False
_C.MODEL.CLIPViP.CAM_METHOD = ''
_C.MODEL.CLIPViP.CAM_SAMPLING = 'topk'
_C.MODEL.CLIPViP.USE_PERSON_EMBED = False


_C.MODEL.ViCLIP = CN()
_C.MODEL.ViCLIP.ARCH = 'ViT-L/14'
_C.MODEL.ViCLIP.WEIGHT_FILE = 'ViClip-InternVid-10M-FLT.pth'
_C.MODEL.ViCLIP.EMBED_DIM = 768
_C.MODEL.ViCLIP.CONTEXT_INIT = ''
_C.MODEL.ViCLIP.CAM_METHOD = ''
_C.MODEL.ViCLIP.USE_ATTN = False
_C.MODEL.ViCLIP.CAM_SAMPLING = 'topk'

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.ITER_PER_EPOCH = -1
_C.SOLVER.MAX_EPOCH = 12

_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.BETAS = (0.9, 0.999)
_C.SOLVER.WEIGHT_DECAY = 0.0001
# Use for bn
_C.SOLVER.WEIGHT_DECAY_BN = 0.0
_C.SOLVER.SCHEDULER = "warmup_multi_step"
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (5, 8)

_C.SOLVER.WARMUP_ON = True
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_EPOCH = 2
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.EVAL_AFTER = 2

# Number of video clips per batch
# This is global, so if we have 8 GPUs and VIDEOS_PER_BATCH = 16, each GPU will
# see 2 clips per batch
_C.SOLVER.VIDEOS_PER_BATCH = 16
_C.SOLVER.OPTIMIZING_METHOD = 'adamw'

# ---------------------------------------------------------------------------- #
# Specific test options
# ------------------------------------------------------FCLIP---------------------- #
_C.TEST = CN()
# Number of video clips per batch
# This is global, so if we have 8 GPUs and VIDEOS_PER_BATCH = 16, each GPU will
# see 2 clips per batch
_C.TEST.VIDEOS_PER_BATCH = 16
_C.TEST.EVAL_OPEN = False
_C.TEST.METRIC = 'frame_ap'
_C.TEST.SMALL_OPEN_WORLD = False
_C.TEST.INDEPENDENT_EVAL = False
_C.TEST.IOU_THRESH = 0.5
_C.TEST.PRIOR_BOX_TEST = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

