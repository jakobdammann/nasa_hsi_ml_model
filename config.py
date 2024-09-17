import torch
# import albumentations as A
# from volumentations import volumentations as V
# from albumentations.pytorch import ToTensorV2

# Data Config
NEPTUNE_PROJECT = "jakobdammann/HSI-Frosted-Plastic"
TRAIN_DIR_X = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//train//thorlabs"
TRAIN_DIR_Y = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//train//cubert"
VAL_DIR_X = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//val//thorlabs"
VAL_DIR_Y = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//val//cubert"
MODEL_DIR = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset"
SHAPE_X = (1, 900, 900)
SHAPE_Y = (106, 104, 104)
NEAR_SQUARE = 121 # Nearest square number to the spectral dim of Y, change this when changing SHAPE_Y
RAW_TL_IMAGE = True

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_MODEL = "fp_unet" # unet, unet2d, fp_unet
LEARNING_RATE = 1e-4 # Starting learning rate
LR_GAMMA = 0.99985 # Gamma for exponential decay function
LR_START_DECAY = 5000 # start exp decay after x steps
DROPOUT = 0.2 # Dropout used for the generator

BATCH_SIZE = 1
NUM_WORKERS = 2

# Generator loss function
ADV_LAMDA = 1**2
L1_LAMBDA = 30**2
SPEC_LAMBDA = 50**2
LFM_LAMBDA = 100**2

# Training
NUM_EPOCHS = 3
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "model/disc.pth.tar"
CHECKPOINT_GEN = "model/gen.pth.tar"


# augm_3d = V.Compose([V.Resize((106, 900, 900), always_apply=True),
#                      V.PadIfNeeded((106,1024,1024), value=0)
#                      ])

# augm_2d = A.PadIfNeeded(1024, 1024, border_mode=cv2.BORDER_CONSTANT, value=0)


# Augmentations

# both_transform = A.Compose(
#     [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
# )

# transform_only_input = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         A.ColorJitter(p=0.2),
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )

# transform_only_mask = A.Compose(
#     [
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )
