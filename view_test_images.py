import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from src.utils import reconstruct_rgb

from torchmetrics import MeanSquaredError

# folder paths for images
tl_folder = 'test/imgs/'
cb_folder = 'test/imgs/'
gen_folder = 'test/imgs/'

# load images
# tl_imgs = np.array([tifffile.imread(tl_folder + f) for f in os.listdir(tl_folder)[:3] if f.endswith(".tif") and f.startswith("tl_raw")])
cb_imgs = np.array([tifffile.imread(cb_folder + f) for f in os.listdir(cb_folder) if f.endswith(".tif") and f.startswith("cb_raw")][:3])
gen_imgs = np.array([tifffile.imread(gen_folder + f) for f in os.listdir(gen_folder) if f.endswith(".tif") and f.startswith("tl_gen")][:3])

wavelengths = np.linspace(450, 870, 106)

print(gen_imgs.shape)
print(cb_imgs.shape)

fig, ax = plt.subplots(3, 3, figsize=(14,10))
fig.suptitle(f"Testing images")

for i in range(3):
    cb_img, gen_img = cb_imgs[i], gen_imgs[i]

    mae = np.mean(np.abs(cb_img - gen_img), axis=(1,2))
    rmse = np.sqrt(np.mean((cb_img - gen_img)**2, axis=(1,2)))
    rase = np.sqrt(np.mean(rmse**2)) * 100 / np.mean(cb_img)

    rrmse_channel = rmse / np.mean(cb_img, axis=(1,2))
    rrmse_image = rmse / np.mean(cb_img)
    arse = np.sqrt(np.mean(rrmse_channel**2)) * 100
    ase = np.sqrt(np.mean(rmse**2)) * 100

    rmae_channel = mae / np.mean(cb_img, axis=(1,2)) # same as dividing by cb_img inside mean
    rmae_image = mae / np.mean(cb_img)
    sre = np.mean(rmae_channel) * 100 # first mean over spatial dim, then spectral dim
    sre_alt = np.mean(np.abs(cb_img - gen_img)) / np.mean(cb_img) * 100
    sre_menon = np.mean(np.abs(cb_img - gen_img) / (cb_img + 1e-12)) * 100

    ax[0,i].imshow(reconstruct_rgb(cb_img))
    ax[0,i].set_title("Ground Truth")
    ax[0,i].set_axis_off()
    ax[1,i].imshow(reconstruct_rgb(gen_img))
    ax[1,i].set_title(f"Gen. Img.")
    ax[1,i].set_axis_off()

    # ax[2,i].plot(wavelengths, mae, label='MAE')
    # ax[2,i].plot(wavelengths, rmse, label='RMSE')
    ax[2,i].plot(wavelengths, rrmse_image, label='Image RRMSE')
    # ax[2,i].plot(wavelengths, rrmse_channel, label='Channel RRMSE')
    ax[2,i].plot(wavelengths, rmae_image, label='Image RMAE')
    # ax[2,i].plot(wavelengths, rmae_channel, label='Channel RMAE')
    ax[2,i].set_title(f"RASE: {rase:.1f}%, SRE1: {sre:.1f}%, SRE2: {sre_alt:.1f}%")
    ax[2,i].legend()

plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()