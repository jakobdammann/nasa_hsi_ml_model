import numpy as np
import config
import os
from PIL import Image
import tifffile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class Dataset(Dataset):
    def __init__(self, root_dir_x, root_dir_y):
        self.root_dir_x = root_dir_x
        self.root_dir_y = root_dir_y
        self.list_files_x = os.listdir(self.root_dir_x)
        self.list_files_y = os.listdir(self.root_dir_y)

    def __len__(self):
        return len(self.list_files_x)

    def __getitem__(self, index):
        img_file_x = self.list_files_x[index]
        img_path_x = os.path.join(self.root_dir_x, img_file_x)
        image_x = np.array(tifffile.imread(img_path_x))

        img_file_y = self.list_files_y[index]
        img_path_y = os.path.join(self.root_dir_y, img_file_y)
        image_y = np.array(tifffile.imread(img_path_y))

        input_image = np.array([image_x[3].astype('float32')])
        #input_image = config.augm_2d(image=input_image)["image"]
        print("input_image.shape:", input_image.shape)

        target_image = image_y.astype('float32')
        #target_image = config.augm_3d(image=target_image)["image"]
        print("target_image.shape:", target_image.shape)

        # augmentations commented out for now

        #augmentations = config.both_transform(image=input_image, image0=target_image)
        #input_image = augmentations["image"]
        #target_image = augmentations["image0"]

        #input_image = config.transform_only_input(image=input_image)["image"]
        #target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = Dataset("images/first_dataset/thorlabs", "images/first_dataset/cubert")
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        import sys

        sys.exit()
