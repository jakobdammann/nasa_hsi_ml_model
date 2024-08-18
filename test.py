import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import Dataset
from generator_model import Generator
from torch.utils.data import DataLoader
from tifffile import imwrite

torch.backends.cudnn.benchmark = True

folder = "test"
n = 3

def main():
    gen = Generator(in_channels=1, out_channels=106, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,)

    val_dataset = Dataset(root_dir_x=config.VAL_DIR_X, root_dir_y=config.VAL_DIR_Y)
    val_loader = DataLoader(val_dataset, batch_size=n, shuffle=False)

    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        print(y_fake.shape, x.shape, y.shape)
        for i in range(n):
            save_image(y_fake[i], folder + f"/tl_gen_{i}.tif")
            save_image(x[i] * 0.5 + 0.5, folder + f"/tl_raw_{i}.tif")
            save_image(y[i] * 0.5 + 0.5, folder + f"/cb_raw_{i}.tif")

def save_image(image, path):
    image = image.numpy()
    imwrite(path, image)

if __name__ == "__main__":
    main()
