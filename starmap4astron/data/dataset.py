# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as T

# class StarryNPZDataset(Dataset):
#     """
#     Dataset for .npz files containing multiple examples per file.
#     Each file should have:
#       - a 'flux' array of shape (n_examples, N_PTS)
#       - an 'image' array of shape (n_examples, H, W, C) or (n_examples, C, H, W)
#     Args:
#       path (str): directory of .npz files or single .npz file
#       lc_key (str): key for flux in the .npz (default 'flux')
#       img_key (str): key for image in the .npz (default 'image')
#       img_size (int or None): if set, Resize image to (img_size, img_size)
#       normalize (bool): if True, normalize image to mean=0.5, std=0.5
#     """
#     def __init__(self, path, lc_key='flux', img_key='image', img_size=None, normalize=True):
#         super().__init__()
#         # gather .npz files
#         if os.path.isdir(path):
#             self.files = sorted(
#                 os.path.join(path, f)
#                 for f in os.listdir(path) if f.endswith('.npz')
#             )
#         else:
#             self.files = [path]

#         # load all data into memory
#         self.data = []  # list of (flux_array, image_array)
#         for fn in self.files:
#             arr = np.load(fn, allow_pickle=True)
#             flux = arr[lc_key]       # shape (n, N_PTS)
#             img  = arr[img_key]      # shape (n, H, W, C) or (n, C, H, W)
#             self.data.append((flux, img))

#         # build flat index mapping
#         self.idx_map = []  # list of (file_idx, example_idx)
#         for file_i, (flux, _) in enumerate(self.data):
#             for ex_i in range(flux.shape[0]):
#                 self.idx_map.append((file_i, ex_i))

#         # image transform: ToTensor -> Resize -> Normalize
#         transforms = [T.ToTensor()]
#         if img_size is not None:
#             transforms.append(T.Resize((img_size, img_size)))
#         if normalize:
#             # map [0,1] or [0,255] to [-1,1]
#             transforms.append(T.Normalize((0.5,), (0.5,)))
#         self.img_transform = T.Compose(transforms)

#     def __len__(self):
#         return len(self.idx_map)

#     def __getitem__(self, idx):
#         file_i, ex_i = self.idx_map[idx]
#         flux_arr, img_arr = self.data[file_i]

#         # get light curve: (N_PTS,)
#         lc = flux_arr[ex_i].astype(np.float32)
#         # to tensor and add channel dim: (1, N_PTS)
#         lc = torch.from_numpy(lc).unsqueeze(0)

#         # get image array
#         img = img_arr[ex_i]
#         # ensure HWC ordering for ToTensor
#         if img.ndim == 3 and img.shape[0] in (1, 3):
#             img = np.moveaxis(img, 0, -1)
#         # apply transforms (ToTensor handles numpy -> tensor)
#         img = self.img_transform(img)

#         return lc, img

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class StarryNPZDataset(Dataset):
    """
    PyTorch Dataset for loading synthetic starry NPZ files containing a single-channel image and a light curve per example.

    Each .npz file contains multiple examples:
      - 'flux': light curve array of shape (n_examples, N_PTS)
      - 'image': single-channel image array of shape (n_examples, H, W)

    This class does the following:
      1. Collect .npz files from a directory or use a single file.
      2. Load all arrays into memory at initialization.
      3. Build a flat index mapping for fast lookup.
      4. In __getitem__, returns a tuple:
         - light curve tensor of shape (1, N_PTS)
         - image tensor of shape (1, H, W)
      5. Replace any NaN values in the image with 0 (no normalization applied).
      6. Optionally resize the image to a fixed square size.
    """

    def __init__(self, path, lc_key='flux', img_key='image', img_size=None):
        super().__init__()
        # 1) Gather all .npz files
        if os.path.isdir(path):
            self.files = sorted(
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith('.npz')
            )
        else:
            self.files = [path]

        # 2) Load data into memory: list of (flux_array, image_array)
        self.data = []
        for file_path in self.files:
            arr = np.load(file_path, allow_pickle=True)
            flux_array = arr[lc_key]       # (n_examples, N_PTS)
            image_array = arr[img_key]     # (n_examples, H, W)
            self.data.append((flux_array, image_array))

        # 3) Build flat index map: maps dataset idx -> (file_idx, example_idx)
        self.idx_map = []
        for file_idx, (flux_array, _) in enumerate(self.data):
            for ex_idx in range(flux_array.shape[0]):
                self.idx_map.append((file_idx, ex_idx))

        # 4) Set up transforms: ToTensor and optional Resize
        transform_list = [T.ToTensor()]  # converts HxW or HxWx1 to tensor in [0,1]
        if img_size is not None:
            transform_list.append(T.Resize((img_size, img_size)))
        self.img_transform = T.Compose(transform_list)

    def __len__(self):
        # Total number of examples across all files
        return len(self.idx_map)

    def __getitem__(self, idx):
        # Locate which file and which example within that file
        file_idx, ex_idx = self.idx_map[idx]
        flux_array, image_array = self.data[file_idx]

        # 5a) Prepare light curve:
        lc = flux_array[ex_idx].astype(np.float32)       # 1D array: (N_PTS,)
        lc_tensor = torch.from_numpy(lc).unsqueeze(0)   # shape: (1, N_PTS)

        # 5b) Prepare single-channel image:
        img = image_array[ex_idx]  # shape: (H, W)
        # Replace NaNs with zero to avoid invalid values
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # If the image is 2D (H, W), add a channel axis -> (H, W, 1)
        if img.ndim == 2:
            img = img[:, :, None]

        # Apply transforms: ToTensor -> (C=1, H, W) and Resize if set
        img_tensor = self.img_transform(img)

        # Return the light curve and image tensors
        return lc_tensor, img_tensor
