import torch
import config
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from RGB.HSI2RGB import HSI2RGB
from torchmetrics.functional.image import relative_average_spectral_error

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

def create_plot(generator_prediction, target, epoch=0):
    fig, ax = plt.subplots(2, 3, figsize=(10,7))
    fig.suptitle(f"Example Images, Epoch {epoch+1}")

    for i, (y_fake, y) in enumerate(zip(generator_prediction[:3], target[:3])):
        RASE_val = relative_average_spectral_error(y_fake.unsqueeze(0).add(1), y.unsqueeze(0).add(1)).item()
        y_fake = y_fake.cpu().numpy() * 0.5 + 0.5  # remove normalization
        y = y.cpu().numpy() * 0.5 + 0.5
        # Pyplot
        ax[0,i].imshow(reconstruct_rgb(y))
        ax[0,i].set_title("Ground Truth")
        ax[0,i].set_axis_off()
        ax[1,i].imshow(reconstruct_rgb(y_fake))
        ax[1,i].set_title(f"Gen. Img. (RASE={RASE_val:.1f})")
        ax[1,i].set_axis_off()
        print("Uploaded example plot.")
    plt.subplots_adjust(hspace=0.3)
    return fig

def reconstruct_rgb(img):
    wl = np.linspace(400,1000,config.SHAPE_Y[0]) # this may not be real values, just what looks best
    img = img.transpose(1,2,0)
    data = np.reshape(img, [-1, config.SHAPE_Y[0]])
    RGB = HSI2RGB(wl, data, 42, 42, 65, 0.002)
    #RGB = RGB.transpose(2,0,1)
    return RGB

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


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