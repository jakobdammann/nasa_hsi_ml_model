import torch
from src.utils import load_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from src.dataset import Dataset
from src.models.unet_model import Generator
from torch.utils.data import DataLoader
from tifffile import imwrite
import numpy as np

torch.backends.cudnn.benchmark = True

folder = "test/imgs"
n = 3

def main():
    print(f"\nTesting... n={n}\n")
    gen = Generator(in_channels=config.SHAPE_X[0], out_channels=config.SHAPE_Y[0], features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,)

    val_dataset = Dataset(root_dir_x=config.VAL_DIR_X, root_dir_y=config.VAL_DIR_Y)
    val_loader = DataLoader(val_dataset, batch_size=n, shuffle=True)

    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        y_fake = y_fake.cpu().numpy()
        y = y.cpu().numpy()
        x = x.cpu().numpy()
        print(y_fake.shape, np.min(y_fake), np.max(y_fake))
        print(y.shape, np.min(y), np.max(y))
        print(x.shape, np.min(x), np.max(x))
        for i in range(n):
            save_image(y_fake[i], folder + f"/tl_gen_{i}.tif")
            save_image(x[i] * 0.5 + 0.5, folder + f"/tl_raw_{i}.tif")
            save_image(y[i] * 0.5 + 0.5, folder + f"/cb_raw_{i}.tif")
    print("\nTesting done\n")

def save_image(image, path):
    imwrite(path, image)

if __name__ == "__main__":
    main()
