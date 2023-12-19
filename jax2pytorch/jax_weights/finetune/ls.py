import numpy as np

npz = np.load('ViT-B_16.npz')
for k, v in npz.items():
    print(k)
    print(v.shape)
