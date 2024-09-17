import torch
from torch.utils.data import DataLoader
import numpy as np
from tifffile import imwrite

from src.pix2pix import Pix2Pix
from src.dataset import Dataset
import config as c

import glob
import os

torch.backends.cudnn.benchmark = True

folder = "test/imgs"
n = 3

def main():
    print(f"\nTesting... n={n}\n")

    checkpoint = sorted(glob.glob(c.MODEL_DIR + '/*.ckpt'), key=os.path.getmtime, reverse=True)[0]
    print("Model:", checkpoint)
    checkpoint = torch.load(checkpoint)
    model = Pix2Pix(run=None)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    val_dataset = Dataset(root_dir_x=c.VAL_DIR_X, root_dir_y=c.VAL_DIR_Y)
    val_loader = DataLoader(val_dataset, batch_size=n, shuffle=False)

    x, y = next(iter(val_loader))
    x, y = x.to(c.DEVICE), y.to(c.DEVICE)
    with torch.no_grad():
        y_fake = model(x).cpu().numpy()
        y = y.cpu().numpy()
        x = x.cpu().numpy()
        print(y_fake.shape, np.min(y_fake), np.max(y_fake))
        print(y.shape, np.min(y), np.max(y))
        print(x.shape, np.min(x), np.max(x))
        for i in range(n):
            save_image(y_fake[i] * 0.5 + 0.5, folder + f"/tl_gen_{i}.tif")
            save_image(x[i] * 0.5 + 0.5, folder + f"/tl_raw_{i}.tif")
            save_image(y[i] * 0.5 + 0.5, folder + f"/cb_raw_{i}.tif")
    print("\nTesting done\n")

def save_image(image, path):
    imwrite(path, image)

if __name__ == "__main__":
    main()
