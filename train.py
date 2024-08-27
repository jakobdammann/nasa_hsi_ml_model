import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import SpectralAngleMapper

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import neptune

import config
from dataset import Dataset
from generator_model import Generator
from discriminator_model import Discriminator

torch.backends.cudnn.benchmark = True


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss: nn.L1Loss, bce, spectral_loss, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True, mininterval=3)
    n = len(loop)
    running_loss = {
        "gen_loss": 0,
        "dis_loss": 0,
        "gen_l1_loss": 0,
        "gen_gan_loss": 0,
        "gen_spec_loss": 0,
        "dis_real": 0,
        "dis_fake": 0
    }

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.amp.autocast('cuda'):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.amp.autocast('cuda'):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            SPEC = spectral_loss(y_fake, y) * config.SPEC_LAMBDA
            G_loss = G_fake_loss + L1 + SPEC

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
        # log into running_loss
        running_loss["gen_loss"] += G_loss.item() / n
        running_loss["dis_loss"] += D_loss.item() / n
        running_loss['gen_l1_loss'] += L1.item() / n
        running_loss['gen_gan_loss'] += G_fake_loss.item() / n
        running_loss['gen_spec_loss'] += SPEC.item() / n
        running_loss['dis_real'] += torch.sigmoid(D_real).mean().item() / n
        running_loss['dis_fake'] += torch.sigmoid(D_fake).mean().item() / n

    return running_loss

def main():
    print("\nTraining...\n")

    # Neptune
    run = neptune.init_run(project="jakobdammann/NASA-HSI-ML-Model",
                           capture_hardware_metrics=True,
                           capture_stderr=True,
                           capture_stdout=True,)
    params = {'Epoch': config.NUM_EPOCHS,
              'Batch Size': config.BATCH_SIZE,
              'Optimizer': 'Adam',
              'Metrics': ['Binary Cross Entropy', 'L1 Loss'],
              'Activation': ['Leaky Relu', 'Relu', 'Tanh',],
              'L1_lambda': config.L1_LAMBDA,
              'Spectral_lambda': config.SPEC_LAMBDA,
              'Learning Rate': config.LEARNING_RATE,
              'Device': config.DEVICE,
              'Workers': config.NUM_WORKERS,
              'Load Model': config.LOAD_MODEL,
              'Save Model': config.SAVE_MODEL
              }
    run['parameters'] = params

    # Definition of Models and other stuff
    disc = Discriminator(in_channels_x=1, in_channels_y=106).to(config.DEVICE)
    gen = Generator(in_channels=1, out_channels=106, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    SPECTRAL_LOSS = SpectralAngleMapper().to('cuda')

    # Load model
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,)
    
    # Load dataset
    train_dataset = Dataset(root_dir_x=config.TRAIN_DIR_X, root_dir_y=config.TRAIN_DIR_Y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')
    val_dataset = Dataset(root_dir_x=config.VAL_DIR_X, root_dir_y=config.VAL_DIR_Y)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    # epoch loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch}")

        loss = train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, SPECTRAL_LOSS, g_scaler, d_scaler)
        print(loss)
        # neptune log
        run["gen_loss"].log(loss['gen_loss'])
        run["dis_loss"].log(loss['dis_loss'])
        run['gen_l1_loss'].log(loss['gen_l1_loss'])
        run['gen_gan_loss'].log(loss['gen_gan_loss'])
        run['gen_spec_loss'].log(loss['gen_spec_loss'])
        run['dis_real'].log(loss['dis_real'])
        run['dis_fake'].log(loss['dis_fake'])


        if config.SAVE_MODEL and epoch % 1 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, run=run)
    
    run.stop()

    print("\nTraining done.\n")


if __name__ == "__main__":
    main()
