import torch

# Data Config
NEPTUNE_PROJECT = "jakobdammann/HSI-Frosted-Plastic"
COMMON_DIR = "//scratch//general//nfs1//u6060933//test_24-10-11_outside//"

TRAIN_DIR_X = COMMON_DIR + "train//thorlabs"
TRAIN_DIR_Y = COMMON_DIR + "train//cubert"
VAL_DIR_X = COMMON_DIR + "val//thorlabs"
VAL_DIR_Y = COMMON_DIR + "val//cubert"
MODEL_DIR = COMMON_DIR + ""

TEST_DIR_X = VAL_DIR_X
TEST_DIR_Y = VAL_DIR_Y

SHAPE_X = (1, 1000, 1000)
SHAPE_Y = (106, 120, 120)
USE_WL_CHANNELS = [0, 106] # also adjust SHAPE_Y and NEAR_SQUARE
RESIZE_Y_SPECTRAL_DIM_TO = 106 # also adjust NEAR_SQUARE and SHAPE_Y, DON'T adjust USE_WL_CHANNELS
# if an error occurs, look at the RGB reconstruction in utils.py
NEAR_SQUARE = 121 # Nearest higher square number to the spectral dim of Y, change this when changing SHAPE_Y
RAW_TL_IMAGE = True

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_MODEL = "outside_fp_unet" # unet, unet2d, fp_unet, simple_fp_unet, outside_fp_unet
LEARNING_RATE = 5e-5 # Starting learning rate
LR_GAMMA = 0.9992 # Gamma for exponential decay function (smaller => steeper)
LR_START_DECAY = 1000 / 2 # start exp decay after x steps
DROPOUT = 0.2 # Dropout used for the generator

BATCH_SIZE = 1
NUM_WORKERS = 2

# Generator loss function
LAMBDA_ADV = 1 # adverserial (discriminator) loss
LAMBDA_L1 = 25 # l1 loss
LAMBDA_SAM = 0 # spectral angle map
LAMBDA_LFM = 25 # feature matching loss
LAMBDA_RASE = 25 # relative average spectral error
LAMBDA_SSIM = 25 # structural similarity index map

# Training
NUM_EPOCHS = 30 # epochs
LOAD_MODEL = False
SAVE_MODEL = True

# Logging
LOG_IMAGES = True
LOG_ALL = False
FALSE_COLOR_IR = True