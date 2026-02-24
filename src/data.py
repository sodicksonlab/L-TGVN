import torch
import os
import pandas as pd
import numpy as np
from typing import Union, Callable, List, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VarNetSample:
    kspace: torch.Tensor
    prior: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    min_target: torch.Tensor
    max_target: torch.Tensor
    acceleration: Union[float, torch.Tensor]
    fname: Union[str, List[str]]
    slice_idx: Union[int, List[int]]
    num_slices: Union[int, List[int]]
    recon_size: Union[Tuple[int, int], List[Tuple[int, int]]]
    prior_accession: Union[int, List[int]]


class SliceDataset(torch.utils.data.Dataset):
    """
    Dataset for TGVN training.
    """
    def __init__(
        self,
        csv_path: Union[str, Path, os.PathLike],
        transform: Callable,
    ) -> None:
        self.transform = transform
        self.csv = pd.read_csv(csv_path)
        self.npz_path = '/gpfs/data/longitudinalprostatelab/npz'

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, idx: int) -> dict:
        """Return the numpy arrays"""
        row = self.csv.iloc[idx]
        fname = row['npz']
        prior_accession = row['prior_T2w']
        slice_idx = row['slice_idx']
        vol_fname = '#'.join(fname.split('#')[:4])
        vol_num_slices = row['num_slices']
        with np.load(
            os.path.join(self.npz_path, 'current_t2w', fname),
            mmap_mode='r'
        ) as npz:
            kspace = npz['kspace']
            target = np.abs(npz['reconstruction'][np.newaxis, ...])
        with np.load(
            os.path.join(
                self.npz_path, 'prior_t2w', str(prior_accession) + '.npz'
            ), mmap_mode='r'
        ) as npz:
            prior = npz['volume'].astype(np.float32, copy=False)

        num_pe = row['num_pe']
        recon_size = eval(row['recshape'])
        return self.transform(
            kspace, target, prior,
            recon_size, num_pe, vol_fname,
            slice_idx, vol_num_slices, prior_accession
        )


class TGVNDataTransform:
    def __init__(
        self,
        buffer_size: int = 11,
        acceleration: int = 15,
        center_fraction: float = 0.5,
        randomize_mask: bool = True,
    ):
        assert isinstance(buffer_size, int) and buffer_size % 2 == 1, \
            "Buffer size must be odd"
        assert isinstance(acceleration, int) and acceleration >= 1, \
            "Acceleration must be at least 1"
        assert 0 < center_fraction <= 1.0, \
            "Center fraction must be in (0, 1]"

        self.buffer_size = buffer_size
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.randomize_mask = randomize_mask
        self.eps = 1e-6

    def __call__(
        self,
        kspace: np.ndarray,
        target: np.ndarray,
        prior: np.ndarray,
        recon_size: Tuple[int, int],
        num_pe: int,
        vol_fname: str,
        slice_idx: int,
        vol_num_slices: int,
        prior_accession: int,
    ) -> VarNetSample:

        kspace = torch.from_numpy(kspace)
        target = torch.from_numpy(target)
        prior = torch.from_numpy(prior)

        # We store the minimum and maximum of the target
        # to use it in the loss calculation
        min_target, max_target = target.min(), target.max()
        num_samples = num_pe // self.acceleration
        num_center_lines = int(self.center_fraction * num_samples)
        num_outside_lines = num_samples - num_center_lines
        l_outside_lines = num_outside_lines // 2
        r_outside_lines = num_outside_lines - l_outside_lines

        # Select matching slices from prior volume
        lp = prior.shape[0]
        lc = vol_num_slices
        half = self.buffer_size // 2
        left = slice_idx + lp - lc - half
        right = slice_idx + lp - lc + half + 1
        indices = np.arange(left, right)
        prior = prior[indices, ...]

        # Normalize prior volume so that it is zero mean and unit std
        mus = prior.mean(dim=(-2, -1), keepdim=True)
        stds = prior.std(dim=(-2, -1), keepdim=True)
        prior = (prior - mus) / (stds + self.eps)

        # Create the undersampling mask and apply it to k-space
        mask = torch.zeros(
            1, 1, num_pe,
            dtype=torch.bool,
            device=kspace.device
        )
        left_idx = (num_pe - num_center_lines) // 2
        right_idx = (num_pe + num_center_lines) // 2
        mask[..., left_idx:right_idx] = 1

        # Set the outside lines randomly
        if self.randomize_mask:
            left_perm = torch.randperm(left_idx)[:l_outside_lines]
            right_perm = torch.randperm(num_pe - right_idx)[:r_outside_lines]
            mask[..., left_perm] = 1
            mask[..., right_idx + right_perm] = 1
        else:
            # Use a seed derived from the vol_fname for reproducibility
            seed = sum(ord(c) for c in vol_fname) % (2**32)
            generator = torch.Generator().manual_seed(seed)
            left_perm = torch.randperm(left_idx, generator=generator)
            left_perm = left_perm[:l_outside_lines]
            right_perm = torch.randperm(
                num_pe - right_idx, generator=generator
            )
            right_perm = right_perm[:r_outside_lines]
            mask[..., left_perm] = 1
            mask[..., right_idx + right_perm] = 1

        # Apply the undersampling mask
        kspace = mask * kspace
        return VarNetSample(
            kspace, prior, mask, target,
            min_target, max_target, num_pe / num_samples,
            vol_fname, slice_idx, lc, recon_size, prior_accession
        )  # type: ignore


def collate_fn(batch: List[VarNetSample]) -> VarNetSample:
    """
    Collate function for dataloader to batch VarNetSamples
    """
    # Number of coils is uniform across all samples, so we can stack them
    kspace = torch.stack([s.kspace for s in batch])
    prior = torch.stack([s.prior for s in batch])
    mask = torch.stack([s.mask for s in batch])
    target = torch.stack([s.target for s in batch])
    mins = torch.stack([s.min_target for s in batch])
    maxs = torch.stack([s.max_target for s in batch])
    accelerations = torch.tensor([s.acceleration for s in batch])
    vol_fnames = [s.fname for s in batch]
    slice_inds = [s.slice_idx for s in batch]
    num_slices = [s.num_slices for s in batch]
    recon_size = batch[0].recon_size  # all iterms have the same recon size
    prior_accessions = [s.prior_accession for s in batch]

    return VarNetSample(
        kspace, prior, mask, target, mins, maxs, accelerations, vol_fnames,
        slice_inds, num_slices, recon_size, prior_accessions
    )  # type: ignore
