import numpy as np
import tifffile
import os
import config as c

print("Loading main function.")

def main():
    print("Loading image paths.")
    # List available files in each folder
    x_train = sorted([os.path.join(c.TRAIN_DIR_X, f) for f in os.listdir(c.TRAIN_DIR_X)])
    y_train = sorted([os.path.join(c.TRAIN_DIR_Y, f) for f in os.listdir(c.TRAIN_DIR_Y)])
    x_val = sorted([os.path.join(c.VAL_DIR_X, f) for f in os.listdir(c.VAL_DIR_X)])
    y_val = sorted([os.path.join(c.VAL_DIR_Y, f) for f in os.listdir(c.VAL_DIR_Y)])

    if len(x_train) != len(y_train):
        print(f"Training dataset has {len(x_train)} X and {len(y_train)} Y images.")
    if len(x_val) != len(y_val):
        print(f"Validation dataset has {len(x_val)} X and {len(y_val)} Y images.")

    print("Train dataset:")
    for i, (x_path, y_path) in enumerate(zip(x_train, y_train)):
        check_for_errors(i, x_path, y_path)
    print("")
    print("Validation dataset:")
    for i, (x_path, y_path) in enumerate(zip(x_val, y_val)):
        check_for_errors(i, x_path, y_path)
    print("")

    print("Verification done.")

print("Loading check function.")

def check_for_errors(i, x_path, y_path):
    x_img = tifffile.imread(x_path)
    y_img = tifffile.imread(y_path)

    if y_path[:12] != x_path[:12]:
        print(f"Image index {i}: ")
        print(f"X ({x_path}) and Y ({y_path}) are probably not the same image.")

    x_nan = np.count_nonzero(np.isnan(x_img))
    if x_nan != 0:
        print(f"X image {i}  ({x_path}) contains {x_nan} nan values.")

    y_nan = np.count_nonzero(np.isnan(y_img))
    if y_nan != 0:
        print(f"Y image {i} ({y_path}) contains {y_nan} nan values.")


if __name__ == "__main__":
    print("Starting verfication.")
    main()