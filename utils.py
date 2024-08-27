import torch
import config
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from RGB.HSI2RGB import HSI2RGB

def save_some_examples(gen, val_loader, epoch, run):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake.cpu().numpy() * 0.5 + 0.5  # remove normalization#
        y = y.cpu().numpy() * 0.5 + 0.5
        # Pyplot
        figure, ax = plt.subplots(2, 3, figsize=(10,6))
        plt.title(f"Example Images, Epoch {epoch}")
        for i in range(3):
            ax[0,i].imshow(reconstruct_rgb(y[i]))
            ax[1,i].imshow(reconstruct_rgb(y_fake[i]))
        run[f"example_{epoch}"].upload(figure)
        print("Uploaded example plot.")
    gen.train()

def reconstruct_rgb(img):
    wl = np.linspace(350,850,106)
    img = img.transpose(1,2,0)
    data = np.reshape(img, [-1, 106])
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