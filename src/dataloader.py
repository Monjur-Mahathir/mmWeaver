import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def get_single_frame_mgrid(doppler=16, range=64, azimuth=64, elevation=8):
    tensor_doppler = torch.linspace(-1, 1, steps= doppler)
    tensor_range = torch.linspace(-1, 1, steps=range)
    tensor_azimuth = torch.linspace(-1, 1, steps=azimuth)
    tensor_elevation = torch.linspace(-1, 1, steps=elevation)

    tensors = tuple([tensor_doppler, tensor_range, tensor_azimuth, tensor_elevation]) # 16 x 64 x 64 x 8
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, 4)

    return mgrid

class SingleFrameFullDataset(Dataset):
    def __init__(self, frame_path):
        super().__init__()
        heatmap = np.load(frame_path)
        heatmap = heatmap / np.max(np.abs(heatmap))
        self.heatmap = heatmap
        
        self.values = np.reshape(heatmap, newshape=heatmap.shape[0]*heatmap.shape[1]*heatmap.shape[2]*heatmap.shape[3])
        self.values = np.stack([self.values.real, self.values.imag], axis=1, dtype=np.float32) # (16*64*64*8, 2)
        self.values = self.values[None, :, :] # (1, 16*64*64*8, 2)
        
        self.coords = get_single_frame_mgrid(doppler=16, range=64, azimuth=64, elevation=8) # (16*64*64*8, 4)
        self.coords = self.coords[None, :, :] # (1, 16*64*64*8, 4)

        assert self.coords.shape[1] == self.values.shape[1], "Shape of Coordinates and Values do not match"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.coords[index], self.values[index], self.heatmap
    

class SingleFrameDataset(Dataset):
    def __init__(self,
        frame_path,
        top_ratio: float = 0.05,
        non_top_ratio: float = 0.05,
        seed: int = 0
    ):
        super().__init__()

        # Load complex heatmap: shape (D,R,A,E) = (16,64,64,8)
        heatmap = np.load(frame_path)   # complex-valued array
        assert np.iscomplexobj(heatmap), "Heatmap must be complex-valued"
        heatmap = heatmap / np.max(np.abs(heatmap))

        self.dims = heatmap.shape      # (16,64,64,8)
        self.N = np.prod(self.dims)    # total number of coordinates
        self.top_ratio = top_ratio
        self.non_top_ratio = non_top_ratio

        flat_complex = heatmap.reshape(-1)
        magnitudes = np.abs(flat_complex)  # (N,)

        # Number of top and non-top samples
        self.k_top = max(1, int(round(self.top_ratio * self.N)))
        self.k_non_top = max(1, int(round(self.non_top_ratio * self.N)))
        assert self.k_top + self.k_non_top <= self.N, "Requested too many samples"

        # Sort indices by descending magnitude
        sorted_indices = np.argsort(-magnitudes)              # descending
        self.top_indices = sorted_indices[: self.k_top]       # fixed set
        self.non_top_pool = sorted_indices[self.k_top :]      # pool to sample from

        self.all_values = np.stack(
            [flat_complex.real, flat_complex.imag],
            axis=1
        ).astype(np.float32)  # (N, 2)

        all_coords = get_single_frame_mgrid(
            doppler=self.dims[0],
            range=self.dims[1],
            azimuth=self.dims[2],
            elevation=self.dims[3]
        )
        assert all_coords.shape[0] == self.N
        self.all_coords = all_coords

        # RNG for non-top resampling
        self.rng = np.random.default_rng(seed)

        # Build initial selected subset (top + one draw from non-top pool)
        self.resample_non_top()

    def resample_non_top(self):
        """Sample a new set of non-top indices and build the active subset."""
        non_top_indices = self.rng.choice(
            self.non_top_pool,
            size=self.k_non_top,
            replace=False
        )

        self.selected_indices = np.concatenate([self.top_indices, non_top_indices])
        self.rng.shuffle(self.selected_indices)

        # Build current coords and values views
        self.coords = self.all_coords[self.selected_indices]    # (k_top + k_non_top, 4)
        self.values = self.all_values[self.selected_indices]    # (k_top + k_non_top, 2)

    def __len__(self):
        # Size is always k_top + k_non_top
        return self.coords.shape[0]

    def __getitem__(self, index):
        # Return torch tensors for convenience
        coord = self.coords[index]                     # (4,)
        value = torch.from_numpy(self.values[index])   # (2,)

        if index == self.__len__() - 1:
            self.resample_non_top()
        return coord, value