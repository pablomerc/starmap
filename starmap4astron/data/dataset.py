import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class StarryNPZDataset(Dataset):
    """
    Dataset for .npz files containing multiple examples per file.
    Each file should have:
      - a 'flux' array of shape (n_examples, N_PTS)
      - an 'image' array of shape (n_examples, H, W, C) or (n_examples, C, H, W)
    Args:
      path (str): directory of .npz files or single .npz file
      lc_key (str): key for flux in the .npz (default 'flux')
      img_key (str): key for image in the .npz (default 'image')
      img_size (int or None): if set, Resize image to (img_size, img_size)
      normalize (bool): if True, normalize image to mean=0.5, std=0.5
    """
    def __init__(self, path, lc_key='flux', img_key='image', img_size=None, normalize=True):
        super().__init__()
        # gather .npz files
        if os.path.isdir(path):
            self.files = sorted(
                os.path.join(path, f)
                for f in os.listdir(path) if f.endswith('.npz')
            )
        else:
            self.files = [path]

        # load all data into memory
        self.data = []  # list of (flux_array, image_array)
        for fn in self.files:
            arr = np.load(fn, allow_pickle=True)
            flux = arr[lc_key]       # shape (n, N_PTS)
            img  = arr[img_key]      # shape (n, H, W, C) or (n, C, H, W)
            self.data.append((flux, img))

        # build flat index mapping
        self.idx_map = []  # list of (file_idx, example_idx)
        for file_i, (flux, _) in enumerate(self.data):
            for ex_i in range(flux.shape[0]):
                self.idx_map.append((file_i, ex_i))

        # image transform: ToTensor -> Resize -> Normalize
        transforms = [T.ToTensor()]
        if img_size is not None:
            transforms.append(T.Resize((img_size, img_size)))
        if normalize:
            # map [0,1] or [0,255] to [-1,1]
            transforms.append(T.Normalize((0.5,), (0.5,)))
        self.img_transform = T.Compose(transforms)

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        file_i, ex_i = self.idx_map[idx]
        flux_arr, img_arr = self.data[file_i]

        # get light curve: (N_PTS,)
        lc = flux_arr[ex_i].astype(np.float32)
        # to tensor and add channel dim: (1, N_PTS)
        lc = torch.from_numpy(lc).unsqueeze(0)

        # get image array
        img = img_arr[ex_i]
        # ensure HWC ordering for ToTensor
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.moveaxis(img, 0, -1)
        # apply transforms (ToTensor handles numpy -> tensor)
        img = self.img_transform(img)

        return lc, img
