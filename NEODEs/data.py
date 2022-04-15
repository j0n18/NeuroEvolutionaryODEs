import logging
import os

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dysts import flows

def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


class NeuralODEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        system: str = "Lorenz",
        obs_dim: int = None,
        n_samples: int = 2000,
        n_timesteps: int = 150,
        pts_per_period: int = 50,
        seed: int = 0,
        batch_size: int = 64,
        noise: float = 0.1,
        make_dist: bool = False,
        save_dist: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Instantiate the dynamical system
        self.model = getattr(flows, system)()
        self.fpath = f"data/Lorenz_system_{self.hparams.n_samples}_{self.hparams.n_timesteps}_{self.hparams.noise}.h5"        

    def save_dist(self, fpath, train_dist, val_dist, test_dist):
        with h5py.File(fpath, "w") as h5file:
            h5file.create_dataset("train_data", data=train_dist)
            h5file.create_dataset("valid_data", data=val_dist)
            h5file.create_dataset("test_data", data=test_dist)

    def load_dist(self, fpath):
        with h5py.File(fpath, "r") as h5file:
            # Load the data
            train_dist = to_tensor(h5file["train_data"][()])
            valid_dist = to_tensor(h5file["valid_data"][()])
            test_dist = to_tensor(h5file["test_data"][()])
        return train_dist, valid_dist, test_dist
    
    def setup(self):
        hps = self.hparams
        # Load data arrays from file
        if hps.make_dist:
            trajectory = self.model.make_trajectory(
                n=hps.n_samples * hps.n_timesteps,
                resample=True,
                pts_per_period=hps.pts_per_period,
                return_times=False,
                noise=hps.noise,
            )
            trajectory = trajectory.reshape(hps.n_samples, hps.n_timesteps, -1)
            inds = np.arange(hps.n_samples)
            train_inds, test_inds = train_test_split(
                inds, test_size=0.2, random_state=hps.seed
            )
            train_inds, valid_inds = train_test_split(
                train_inds, test_size=0.2, random_state=hps.seed
            )
            train_dist = trajectory[train_inds]
            val_dist = trajectory[valid_inds]
            test_dist = trajectory[test_inds]
            if hps.save_dist:
                save_dist(self.fpath)
        else:
            train_dist, valid_dist, test_dist = load_dist(self.fpath)
        # Store datasets
        self.train_ds = TensorDataset(train_dist)
        self.valid_ds = TensorDataset(valid_dist)
        self.test_ds = TensorDataset(test_dist)
    
    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            #num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            #num_workers=self.hparams.num_workers,
        )
        return valid_dl
    
    def test_dataloader(self):
        test_dl = DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            #num_workers=self.hparams.num_workers,
        )
        return test_dl