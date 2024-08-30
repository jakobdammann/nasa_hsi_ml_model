import numpy as np
import tifffile
import os

n_images = 10

#Thorlabs images
for i in range(n_images):
    a = np.random.randint(low=0, high=4096, size=(1,900,900))
    tifffile.imwrite(os.path.join("images", "rnd", "thorlabs", f"{i}_thorlabs.tif"), a)

#Cubert images
for i in range(n_images):
    a = np.random.randint(low=0, high=4096, size=(106,42,42))
    tifffile.imwrite(os.path.join("images", "rnd", "cubert", f"{i}_cubert.tif"), a)