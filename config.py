import torch

# Data Config
NEPTUNE_PROJECT = "jakobdammann/HSI-Frosted-Plastic"
TRAIN_DIR_X = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//train//thorlabs"
TRAIN_DIR_Y = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//train//cubert"

VAL_DIR_X = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//val//thorlabs"
VAL_DIR_Y = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset//val//cubert"
MODEL_DIR = "//scratch//general//nfs1/u6060933//frosted_plastic_dataset"

TEST_DIR_X = VAL_DIR_X
TEST_DIR_Y = VAL_DIR_Y

SHAPE_X = (1, 900, 900)
SHAPE_Y = (106, 104, 104)
NEAR_SQUARE = 121 # Nearest square number to the spectral dim of Y, change this when changing SHAPE_Y
RAW_TL_IMAGE = True

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_MODEL = "fp_unet" # unet, unet2d, fp_unet
LEARNING_RATE = 1e-4 # Starting learning rate
LR_GAMMA = 0.9992 # Gamma for exponential decay function
LR_START_DECAY = 750 # start exp decay after x steps
DROPOUT = 0.2 # Dropout used for the generator

BATCH_SIZE = 1
NUM_WORKERS = 2

# Generator loss function
LAMBDA_ADV = 1
LAMBDA_L1 = 25
LAMBDA_SAM = 0
LAMBDA_LFM = 25
LAMBDA_RASE = 25
LAMBDA_SSIM = 25

# Training
NUM_EPOCHS = 3
LOAD_MODEL = False
SAVE_MODEL = True
LOG_IMAGES = False
# CHECKPOINT_DISC = "model/disc.pth.tar"
# CHECKPOINT_GEN = "model/gen.pth.tar"
