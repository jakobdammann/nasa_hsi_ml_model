import torch
import albumentations as A
from volumentations import volumentations as V
import cv2
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_X = "images//first_dataset//thorlabs"
TRAIN_DIR_Y = "images//first_dataset//cubert"
VAL_DIR_X = "images//first_dataset//thorlabs"
VAL_DIR_Y = "images//first_dataset//cubert"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
# IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 400
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


# Augmentations

augm_3d = V.Compose([V.Resize((106, 900, 900), always_apply=True),
                      V.PadIfNeeded((106,1024,1024), value=0)
                      ])

augm_2d = A.PadIfNeeded(1024, 1024, border_mode=cv2.BORDER_CONSTANT, value=0)

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
