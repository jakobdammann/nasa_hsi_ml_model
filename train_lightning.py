import torch
from src.utils import save_checkpoint, load_checkpoint, log_examples
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import SpectralAngleMapper
from torchmetrics.functional.image import relative_average_spectral_error
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.utils.data import DataLoader
from tqdm import tqdm
import neptune
import time
import os
import glob

import config as c
from src.dataset import Dataset
from src.pix2pix import Pix2Pix

torch.backends.cudnn.benchmark = True

def main():
    print("\nLoading...\n")

    # Neptune
    run = neptune.init_run(project="jakobdammann/NASA-HSI-ML-Model",
                           capture_hardware_metrics=True,
                           capture_stderr=True,
                           capture_stdout=True,)
    params = {'Epoch': c.NUM_EPOCHS,
              'Batch Size': c.BATCH_SIZE,
              'Optimizer': 'Adam',
              'Metrics': 'Binary Cross Entropy, L1 Loss, Spectral Angle',
              'Activation': 'Leaky Relu, Relu, Tanh',
              'ADV_lambda': c.ADV_LAMDA,
              'L1_lambda': c.L1_LAMBDA,
              'Spectral_lambda': c.SPEC_LAMBDA,
              'LFM_lambda': c.LFM_LAMBDA,
              'Learning Rate': c.LEARNING_RATE,
              'Device': c.DEVICE,
              'Workers': c.NUM_WORKERS,
              'Load Model': c.LOAD_MODEL,
              'Save Model': c.SAVE_MODEL
              }
    run['parameters'] = params

    # Definition of Model & Trainer
    model = Pix2Pix(run)
    checkpoint_callback = ModelCheckpoint(dirpath="./model/", filename=f'trained_pix2pix-{{epoch:02d}}-{{step:02d}}')

    trainer = pl.Trainer(max_epochs=c.NUM_EPOCHS, val_check_interval=1.0, 
                         default_root_dir='./model/', callbacks=[checkpoint_callback])

    # Load model
    if c.LOAD_MODEL:
        checkpoint = sorted(glob.glob('./model/*.ckpt'), key=os.path.getmtime, reverse=True)[0]
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("Loaded last checkpoint.")
    
    # Load datasets
    train_dataset = Dataset(root_dir_x=c.TRAIN_DIR_X, root_dir_y=c.TRAIN_DIR_Y)
    train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True, 
                              num_workers=c.NUM_WORKERS)
    val_dataset = Dataset(root_dir_x=c.VAL_DIR_X, root_dir_y=c.VAL_DIR_Y)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    print("\nTraining...\n")
    start = time.time()

    trainer.fit(model, train_loader, val_loader)

    end = time.time()
    print(f"\nTime (min): {(end-start)/60.0}\n")

    run.stop()
    print("\nTraining done.\n")


if __name__ == "__main__":
    main()
