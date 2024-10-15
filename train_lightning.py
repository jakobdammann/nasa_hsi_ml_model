import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar

from torch.utils.data import DataLoader
import neptune
import time
import os
import glob
from datetime import datetime

import config as c
from src.dataset import Dataset
from src.pix2pix import Pix2Pix

torch.backends.cudnn.benchmark = True

def main():
    print("\nLoading...\n")

    # Neptune
    run = neptune.init_run(project=c.NEPTUNE_PROJECT,
                           capture_hardware_metrics=True,
                           capture_stderr=True,
                           capture_stdout=True,)
    params = {'Epochs': c.NUM_EPOCHS,
              'Batch Size': c.BATCH_SIZE,
              'Optimizer': 'Adam',
              'Metrics': 'Binary Cross Entropy, L1 Loss, Spectral Angle',
              'Activation': 'Leaky Relu, Relu, Tanh',
              'Lambda_ADV': c.LAMBDA_ADV,
              'Lambda_L1': c.LAMBDA_L1,
              'Lambda_SAM': c.LAMBDA_SAM,
              'Lambda_LFM': c.LAMBDA_LFM,
              'Lambda_RASE': c.LAMBDA_RASE,
              'Lambda_SSIM': c.LAMBDA_SSIM,
              'Learning Rate': c.LEARNING_RATE,
              'LR Gamma': c.LR_GAMMA,
              'LR Start Decay': c.LR_START_DECAY,
              'Dropout': c.DROPOUT,
              'Device': c.DEVICE,
              'Workers': c.NUM_WORKERS,
              'Load Model': c.LOAD_MODEL,
              'Save Model': c.SAVE_MODEL,
              'Generator Model': c.GENERATOR_MODEL,
              'Dataset': c.COMMON_DIR
              }
    run['parameters'] = params

    # Definition of Model & Trainer
    model = Pix2Pix(run)
    checkpoint_callback = ModelCheckpoint(dirpath=c.MODEL_DIR, monitor='rase',
                                          filename=f'pix2pix-{datetime.now()}-{{epoch:02d}}-{{step:02d}}')
    # bar_callback = TQDMProgressBar(refresh_rate=50)

    trainer = pl.Trainer(max_epochs=c.NUM_EPOCHS, val_check_interval=0.1, 
                         default_root_dir=c.MODEL_DIR, callbacks=[checkpoint_callback])

    # Load model
    if c.LOAD_MODEL:
        checkpoint = sorted(glob.glob(c.MODEL_DIR + '/*.ckpt'), key=os.path.getmtime, reverse=True)[0]
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("Loaded last checkpoint.")
    
    # Load datasets
    train_dataset = Dataset(root_dir_x=c.TRAIN_DIR_X, root_dir_y=c.TRAIN_DIR_Y)
    train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True, 
                              num_workers=c.NUM_WORKERS)
    val_dataset = Dataset(root_dir_x=c.VAL_DIR_X, root_dir_y=c.VAL_DIR_Y)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    run_id = run["sys/id"].fetch()
    print(f"\nTraining {run_id}...\n")
    start = time.time()

    trainer.fit(model, train_loader, val_loader)

    end = time.time()
    print(f"\nTime (min): {(end-start)/60.0}\n")

    run.stop()
    print("\nTraining done.\n")


if __name__ == "__main__":
    main()
