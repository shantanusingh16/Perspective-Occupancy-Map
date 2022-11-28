import torch
import numpy as np

def to_onehot_tensor(arr, nclass):
        out = np.zeros((nclass, *arr.shape), dtype=np.float32)
        for idx in np.arange(nclass):
            out[idx, arr==idx] = 1
        return torch.from_numpy(out)