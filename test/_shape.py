import numpy as np

arr = np.array([])
print(arr)

for i in range(10):
    array = [1,2,3,51,12,5]
    arr = np.append(arr, array)
    arr.shape = (-1, len(array))
    print(arr)

print(arr)

# Test plot

import matplotlib.pyplot as plt

generator_prediction = np.zeros((12,106,120,120))
target = np.random.rand(12,106,120,120)

def reconstruct_rgb(y):
    return y[0]

def calc_RASE(y_fake, y):
    return 0

fig, ax = plt.subplots(4, 3, figsize=(10,10), num=1, clear=True) # parameters so that no memory leak occurs
fig.suptitle(f"Example Images, Epoch {0}, Step {0}")
# loop over first three images in batch
for i, (y_fake, y) in enumerate(zip(generator_prediction[:3], target[:3])):
    y_fake = np.nan_to_num(y_fake) * 0.5 + 0.5  # remove normalization
    y = np.nan_to_num(y) * 0.5 + 0.5
    RASE_val = calc_RASE(y_fake, y)
    # Pyplot
    ax[0,i].imshow(reconstruct_rgb(y))
    ax[0,i].set_title("Ground Truth")
    ax[0,i].set_axis_off()
    ax[1,i].imshow(reconstruct_rgb(y_fake))
    ax[1,i].set_title(f"Gen. Img. (RASE={RASE_val:.1f})")
    ax[1,i].set_axis_off()
for i, (y_fake, y) in enumerate(zip(generator_prediction[3:6], target[3:6])):
    y_fake = np.nan_to_num(y_fake) * 0.5 + 0.5  # remove normalization
    y = np.nan_to_num(y) * 0.5 + 0.5
    RASE_val = calc_RASE(y_fake, y)
    # Pyplot
    ax[2,i].imshow(reconstruct_rgb(y))
    ax[2,i].set_title("Ground Truth")
    ax[2,i].set_axis_off()
    ax[3,i].imshow(reconstruct_rgb(y_fake))
    ax[3,i].set_title(f"Gen. Img. (RASE={RASE_val:.1f})")
    ax[3,i].set_axis_off()
plt.tight_layout()

plt.show()