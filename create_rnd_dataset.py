import numpy as np
import tifffile
import os

n_images = 10

#Thorlabs images
for i in range(n_images):
    a = np.random.randint(low=0, high=4096, size=(1,256,256))
    tifffile.imwrite(os.path.join("images", "thorlabs", f"{i}_thorlabs.tif"), a)

#Cubert images
for i in range(n_images):
    a = np.random.randint(low=0, high=4096, size=(106,256,256))
    tifffile.imwrite(os.path.join("images", "cubert", f"{i}_cubert.tif"), a)