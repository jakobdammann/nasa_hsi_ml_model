import torch
import config
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from RGB.HSI2RGB import HSI2RGB
from torchmetrics.functional.image import relative_average_spectral_error



# unused
def log_examples(gen, val_loader, epoch, step, run):
    fig, ax = plt.subplots(2, 3, figsize=(10,7))
    fig.suptitle(f"Example Images, Epoch {epoch+1}")

    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            RASE_val = relative_average_spectral_error(y_fake, y).mean().item()
            y_fake = y_fake.cpu().numpy() * 0.5 + 0.5  # remove normalization
            y = y.cpu().numpy() * 0.5 + 0.5
            # Pyplot
            ax[0,i].imshow(reconstruct_rgb(y[0]))
            ax[0,i].set_title("Ground Truth")
            ax[0,i].set_axis_off()
            ax[1,i].imshow(reconstruct_rgb(y_fake[0]))
            ax[1,i].set_title(f"Gen. Img, RASE={RASE_val:.1f}")
            ax[1,i].set_axis_off()
            print("Uploaded example plot.")
    plt.subplots_adjust(hspace=0.3)
    run[f"examples"].append(value=fig, step=step)
    gen.train()


def create_plot(generator_prediction, target, epoch=0, step=0):
    fig, ax = plt.subplots(4, 3, figsize=(10,10), num=1, clear=True) # parameters so that no memory leak occurs, doesn't work
    fig.suptitle(f"Example Images, Epoch {epoch}, Step {step}")
    # loop over first three images in batch
    for i, (y_fake, y) in enumerate(zip(generator_prediction[:3], target[:3])):
        y_fake = np.nan_to_num(y_fake.cpu().numpy()) * 0.5 + 0.5  # remove normalization
        y = np.nan_to_num(y.cpu().numpy()) * 0.5 + 0.5
        RASE_val = calc_RASE(y_fake, y)
        # Pyplot
        ax[0,i].imshow(reconstruct_rgb(y), vmin=0, vmax=1)
        ax[0,i].set_title("Ground Truth")
        ax[0,i].set_axis_off()
        ax[1,i].imshow(reconstruct_rgb(y_fake), vmin=0, vmax=1)
        ax[1,i].set_title(f"Gen. Img. (RASE={RASE_val:.1f})")
        ax[1,i].set_axis_off()
    for i, (y_fake, y) in enumerate(zip(generator_prediction[3:6], target[3:6])):
        y_fake = np.nan_to_num(y_fake.cpu().numpy()) * 0.5 + 0.5  # remove normalization
        y = np.nan_to_num(y.cpu().numpy()) * 0.5 + 0.5
        RASE_val = calc_RASE(y_fake, y)
        # Pyplot
        ax[2,i].imshow(reconstruct_rgb(y), vmin=0, vmax=1)
        ax[2,i].set_title("Ground Truth")
        ax[2,i].set_axis_off()
        ax[3,i].imshow(reconstruct_rgb(y_fake), vmin=0, vmax=1)
        ax[3,i].set_title(f"Gen. Img. (RASE={RASE_val:.1f})")
        ax[3,i].set_axis_off()
    plt.subplots_adjust(wspace=-0.2, hspace=0.2)
    del y_fake
    del y
    del RASE_val
    del ax
    return fig


def create_plot_with_spectra(generator_prediction, target, epoch=0, step=0):
    wavelengths = np.linspace(450, 850, config.SHAPE_Y[0])# [config.USE_WL_CHANNELS[0]:config.USE_WL_CHANNELS[1]]  # Assuming 106 channels from 450-850 nm
    crops = [(10,30,75,95), # red
             (65,85,95,115),
             (55,75,5,25),
             (80,100,80,100)]
    # Pyplot
    fig, ax = plt.subplots(3, 4, figsize=(10,7), num=1, clear=True) # parameters so that no memory leak occurs, doesn't work
    fig.suptitle(f"Example Images, Epoch {epoch}, Step {step}")
    # loop over first three images in batch
    for i, (y_fake, y) in enumerate(zip(generator_prediction[:4], target[:4])):
        y_fake = np.nan_to_num(y_fake.cpu().numpy()) * 0.5 + 0.5  # remove normalization
        y = np.nan_to_num(y.cpu().numpy()) * 0.5 + 0.5
        RASE_val = calc_RASE(y_fake, y)
        # rectangle
        crop = crops[i]
        rect = patches.Rectangle(xy=(crop[0], crop[2]), width=crop[1]-crop[0], height=crop[3]-crop[2], 
                                 linewidth=1, edgecolor='tab:blue', facecolor='none',
                                 path_effects=[patheffects.withStroke(linewidth=1.5, foreground="w")])
        rect_same = patches.Rectangle(xy=(crop[0], crop[2]), width=crop[1]-crop[0], height=crop[3]-crop[2], 
                                 linewidth=1, edgecolor='tab:orange', facecolor='none',
                                 path_effects=[patheffects.withStroke(linewidth=1.5, foreground="w")])
        # show GT
        ax[0,i].imshow(reconstruct_rgb(y), vmin=0, vmax=1)
        ax[0,i].add_patch(rect)
        ax[0,i].set_title("Ground Truth")
        # ax[0,i].set_axis_off()
        # show generated
        ax[1,i].imshow(reconstruct_rgb(y_fake), vmin=0, vmax=1)
        ax[1,i].add_patch(rect_same)
        ax[1,i].set_title(f"Gen. Img. (RASE={RASE_val:.1f})")
        # ax[1,i].set_axis_off()
        # plot spectra
        y_crop = y[:,crop[2]:crop[3], crop[0]:crop[1]]
        y_fake_crop = y_fake[:,crop[2]:crop[3], crop[0]:crop[1]]
        ax[2,i].plot(wavelengths, np.mean(y_crop, axis=(1,2)), label="GT")
        ax[2,i].plot(wavelengths, np.mean(y_fake_crop, axis=(1,2)), label="Gen")
        ax[2,i].set_title(f"Sample Spectra (SRE={spec_error(y_fake_crop, y_crop):.1f})")
        ax[2,i].set_xlabel("wavelength [nm]")
        ax[2,i].legend(loc='upper right')
    plt.tight_layout()
    del y_fake
    del y
    del RASE_val
    del ax
    return fig


def reconstruct_rgb(img):
    if config.FALSE_COLOR_IR:
        wl = np.linspace(380, 700, config.SHAPE_Y[0])# [config.USE_WL_CHANNELS[0]:config.USE_WL_CHANNELS[1]]
    else:
        wl = np.linspace(400, 1000, config.SHAPE_Y[0])# [config.USE_WL_CHANNELS[0]:config.USE_WL_CHANNELS[1]] # this may not be the real range, just what looks normal

    img = img.transpose(1, 2, 0)
    data = np.reshape(img, [-1, config.SHAPE_Y[0]])
    RGB = HSI2RGB(wl, data, config.SHAPE_Y[1], config.SHAPE_Y[2], 65, 0.002)
    #RGB = RGB.transpose(2,0,1)
    return RGB

# for Numpy tensors
def calc_RASE(prediction, target):
    # This RASE is different from the RASE in torchmetrics
    # it does not use a sliding window to calc the RMSE
    rmse = np.sqrt(np.mean((target - prediction)**2, axis=(1,2)))
    rase = np.sqrt(np.mean(rmse**2)) * 100 / np.mean(target)
    return rase

# for Pytorch tensors
def rase_error(prediction_image, target_image):
    diff = target_image - prediction_image
    target_mean = torch.mean(target_image)
    rmse_squared = torch.mean(diff.square(), dim=(2,3))
    rase = torch.mean(rmse_squared, dim=1).sqrt().mul(100).div(target_mean)
    return torch.mean(rase)

# for Numpy tensors
def spec_error(prediction_image, target_image):
    # ignoring all voxels where the target is close to 0
    try:
        rel_error = np.where(target_image<1/4096, None, np.abs(target_image - prediction_image) / (target_image))
        sre_menon = np.nanmean(rel_error) * 100 # mean ignores nan
        return sre_menon
    except:
        return 0

# Adapted from https://github.com/jinh0park/pytorch-ssim-3D
def ssim3D(img1, img2, window_size = 11, channel = 1, size_average = True):
    window = create_window_3D(window_size=window_size, channel=channel).to(config.DEVICE)

    mu1 = torch.nn.functional.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = torch.nn.functional.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = torch.nn.functional.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = torch.nn.functional.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


# unused
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# unused
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def print_info(tensor, name=''):
    try:
        img = tensor.detach().numpy()
    except:
        img=tensor
    print(f"{name}, Shape: {tensor.shape[:]} Max: {np.max(img):.2f}, Min: {np.min(img):.2f}, Std: {np.std(img):.2f}")


if __name__ == "__main__":
    print(reconstruct_rgb(np.random.rand(106,42,42)))