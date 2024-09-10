import matplotlib.pyplot as plt
import numpy as np

values = np.load("loss//loss_values.npy")
print(values.shape)
loss = np.mean(values, axis=1)
n_img = values.shape[1]
loss_g = loss[:,0] / n_img
loss_d = loss[:,1] / n_img
print(loss.shape)
epoch = np.arange(1, len(loss)+1, 1)

plt.figure(figsize=(10, 4))
plt.subplot(211)
plt.plot(epoch, loss_g, label="Disc")
plt.legend()
plt.subplot(212)
plt.plot(epoch, loss_d, label="Gen")
plt.legend()
plt.savefig("loss/loss.png")