import torch
# import albumentations as A
# from volumentations import volumentations as V
# from albumentations.pytorch import ToTensorV2

# Data Config
TRAIN_DIR_X = "images//display_dataset//thorlabs"
TRAIN_DIR_Y = "images//display_dataset//cubert"
VAL_DIR_X = "images//validation//thorlabs"
VAL_DIR_Y = "images//validation//cubert"
SHAPE_X = (1, 900, 900)
SHAPE_Y = (106, 42, 42)
NEAR_SQUARE = 121 # Nearest square number to the spectral dim of Y, change this when changing SHAPE_Y
RAW_TL_IMAGE = True

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_MODEL = "unet" # unet, unet2d, fp_unet
LEARNING_RATE = 2e-4
LR_GAMMA = 0.99985
LR_START_DECAY = 10000 # start exp decay after ?? steps
DROPOUT = 0.2

BATCH_SIZE = 1
NUM_WORKERS = 2

ADV_LAMDA = 1
L1_LAMBDA = 30
SPEC_LAMBDA = 50
LFM_LAMBDA = 100

# Training
NUM_EPOCHS = 50
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
