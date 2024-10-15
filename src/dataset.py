import numpy as np
import config as c
import os
import tifffile
from torch.utils.data import Dataset, DataLoader
import volumentations.volumentations as v


class Dataset(Dataset):
    def __init__(self, root_dir_x, root_dir_y):
        self.root_dir_x = root_dir_x
        self.root_dir_y = root_dir_y
        self.list_files_x = sorted(os.listdir(self.root_dir_x))
        self.list_files_y = sorted(os.listdir(self.root_dir_y))
        if len(self.list_files_x) != len(self.list_files_y):
            print("X file list and Y file list are differently long. This may cause wrong learning behaviour.")

    def __len__(self):
        return len(self.list_files_x)

    def __getitem__(self, index):
        img_file_x = self.list_files_x[index]
        img_path_x = os.path.join(self.root_dir_x, img_file_x)
        image_x = np.array(tifffile.imread(img_path_x))

        img_file_y = self.list_files_y[index]
        img_path_y = os.path.join(self.root_dir_y, img_file_y)
        image_y = np.array(tifffile.imread(img_path_y))

        if c.RAW_TL_IMAGE == True and image_x.shape[0] == 4:
            # recreate not demosaiced image out of the mosaic
            input_image = np.empty(shape=(1, *image_x[0].shape), dtype=np.float32)
            input_image[0, 0::2, 0::2] = image_x[0, 0::2, 0::2] # top left, 0 deg
            input_image[0, 0::2, 1::2] = image_x[1, 0::2, 1::2] # top right, 45 deg
            input_image[0, 1::2, 0::2] = image_x[3, 1::2, 0::2] # bottom left, 135 deg or -45 deg
            input_image[0, 1::2, 1::2] = image_x[2, 1::2, 1::2] # bottom right, 90 deg
        elif c.RAW_TL_IMAGE == True and image_x.shape[0] == 5:
            input_image = np.array([image_x[4].astype('float32')])
        else:
            # Take 135 deg polarization channel
            input_image = np.array([image_x[3].astype('float32')])
        # norm
        min = np.min(input_image)
        max = np.max(input_image)
        input_image = (input_image - min) / (max - min + 1e-12)
        input_image = 2 * input_image - 1
        # print(input_image.shape)
        # input_image = input_image[:, :900, :900] # crop test
        # print(input_image.shape)

        target_image = image_y.astype('float32')
        # Removing wl channels
        target_image = target_image[c.USE_WL_CHANNELS[0]:c.USE_WL_CHANNELS[1]]
        # Resizing CB image if needed/wanted
        if target_image.shape[0] != c.RESIZE_Y_SPECTRAL_DIM_TO:
            print("Resizing.")
            downsample = v.Compose([v.Resize((c.RESIZE_Y_SPECTRAL_DIM_TO,c.SHAPE_Y[1],c.SHAPE_Y[2]), always_apply=True)])
            target_image = downsample(image=target_image)["image"]
        # norm
        min = np.min(target_image)
        max = np.max(target_image)
        target_image = (target_image - min) / (max - min + 1e-12)
        target_image = 2 * target_image - 1
        # print(target_image.shape)
        # target_image = target_image[:, :104, :104] # crop test
        # print(target_image.shape)

        # augmentations

        #augmentations = config.both_transform(image=input_image, image0=target_image)
        #input_image = augmentations["image"]
        #target_image = augmentations["image0"]

        #input_image = config.transform_only_input(image=input_image)["image"]
        #target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = Dataset(c.VAL_DIR_X, c.VAL_DIR_Y)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        # plt.imshow(x[0, 0, 400:450, 400:450])
        # plt.colorbar()
        # plt.show()
        break

