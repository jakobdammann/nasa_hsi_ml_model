import numpy as np
import tifffile
import os
import config as c
from pathlib import Path

import matplotlib.pyplot as plt

DIR_X = "//scratch//general//nfs1/u6060933//outside_plastic_dataset_24-09-26//thorlabs"
DIR_Y = "//scratch//general//nfs1/u6060933//outside_plastic_dataset_24-09-26//cubert"

SAVE_DIR_X = c.TRAIN_DIR_X
SAVE_DIR_Y = c.TRAIN_DIR_Y

crop_x = ((750-250, 1650+250), (750-250-400, 1650+250-400))
crop_y = ((110+20, 290+14), (110-20, 290-26)) 

show_image = True
verify_images = False
crop_all_images = False


# Workflow:
# 1. Take images
# 2. Upload Images via the Lab PC to the chpc with these commands:
#    cd E:/hyperspectral_cam_control/images
#    rsync -a --progress ./cubert u???????@intdtn01.chpc.utah.edu:/scratch/general/nfs1/u6060933/folder_name/cubert
#    rsync -a --progress ./thorlabs u???????@intdtn01.chpc.utah.edu:/scratch/general/nfs1/u6060933/folder_name/thorlabs
# 3. change the directories in this file and the config file
# 4. run this code in an interactive session (e.g. with FastX) and use show_images = True to preview the crop
#    and change accordingly
# 5. run this code with sbatch with verify_images = True (everything else false)
# 6. run this code with crop_all_images = True to crop these and save them in the SAVE_DIR_?


def main():
    if show_image:
        show_crop()

    if verify_images or crop_all_images:
        image_loop()


def show_crop():
    print("")
    print("Loading image paths.")
    x_paths = sorted([os.path.join(DIR_X, f) for f in os.listdir(DIR_X)[20:23]])
    y_paths = sorted([os.path.join(DIR_Y, f) for f in os.listdir(DIR_Y)[20:23]])
    print(x_paths)
    print(y_paths)

    print("Cropping images to show.")

    x_imgs = []
    y_imgs = []

    for i, (x_path, y_path) in enumerate(zip(x_paths, y_paths)): 
        print("Index:", i)
        x_imgs.append(do_crop_x(tifffile.imread(x_path)))
        y_imgs.append(do_crop_y(tifffile.imread(y_path)))
        print(x_imgs[-1].shape, y_imgs[-1].shape)       

    print("Plotting.")

    plt.figure(figsize=(10,10))
    plt.subplot(321)
    plt.imshow(x_imgs[0][4])
    plt.subplot(322)
    plt.imshow(y_imgs[0][53])
    plt.subplot(323)
    plt.imshow(x_imgs[1][4])
    plt.subplot(324)
    plt.imshow(y_imgs[1][53])
    plt.subplot(325)
    plt.imshow(x_imgs[2][4])
    plt.subplot(326)
    plt.imshow(y_imgs[2][53])
    plt.show()


def image_loop():
    print("Loading image paths.")
    # List available files in each folder
    x_train = sorted([os.path.join(DIR_X, f) for f in os.listdir(DIR_X)])
    y_train = sorted([os.path.join(DIR_Y, f) for f in os.listdir(DIR_Y)])
    # create save dir
    Path(SAVE_DIR_X).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_Y).mkdir(parents=True, exist_ok=True)

    print(f"Dataset has {len(x_train)} X and {len(y_train)} Y images.")
    if len(x_train) != len(y_train):
        print("THAT'S BAD!")

    print("Dataset:")
    for i, (x_path, y_path) in enumerate(zip(x_train, y_train)):
        print("Index:", i)
        try:
            x_img = tifffile.imread(x_path)
            y_img = tifffile.imread(y_path)

            if verify_images: 
                check_for_errors(i, x_img, y_img, x_path, y_path)
            if crop_all_images:
                x_img, y_img = do_crop(x_img, y_img)
                tifffile.imwrite(os.path.join(SAVE_DIR_X, f"{i}c_thorlabs.tif"), x_img)
                tifffile.imwrite(os.path.join(SAVE_DIR_Y, f"{i}c_cubert.tif"), y_img)
        except:
            print("Could not process images.")
    print("Image loop done.")


def check_for_errors(i, x_img, y_img, x_path, y_path):
    if y_path.split('/')[-1].split('_')[0] != x_path.split('/')[-1].split('_')[0]:
        print(f"Image index {i}: ")
        print(f"X ({x_path}) and Y ({y_path}) are probably not the same image.")

    x_nan = np.count_nonzero(np.isnan(x_img))
    if x_nan != 0:
        print(f"X image {i}  ({x_path}) contains {x_nan} nan values.")

    y_nan = np.count_nonzero(np.isnan(y_img))
    if y_nan != 0:
        print(f"Y image {i} ({y_path}) contains {y_nan} nan values.")   

def do_crop(x, y):
    x = x[:, crop_x[1][0]:crop_x[1][1], crop_x[0][0]:crop_x[0][1]]
    y = y[:, crop_y[1][0]:crop_y[1][1], crop_y[0][0]:crop_y[0][1]]
    return x, y

def do_crop_x(x):
    x_new = x[:, crop_x[1][0]:crop_x[1][1], crop_x[0][0]:crop_x[0][1]]
    return x_new

def do_crop_y(y):
    y_new = y[:, crop_y[1][0]:crop_y[1][1], crop_y[0][0]:crop_y[0][1]]
    return y_new


if __name__ == "__main__":
    print("Starting verfication.")
    main()