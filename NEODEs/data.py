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
        make_data: bool = False,
        save_data: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Instantiate the dynamical system
        self.model = getattr(flows, system)()
        self.fpath = f"datasets\\Lorenz_system_{self.hparams.n_samples}_{self.hparams.n_timesteps}_{self.hparams.noise}.h5"        

    def save_data(self, fpath, train_data, valid_data, test_data):

        import pdb; pdb.set_trace();

        train_dist, train_tpts = train_data
        valid_dist, valid_tpts = valid_data
        test_dist, test_tpts = test_data

        with h5py.File(fpath, "w") as h5file:

            h5file.create_dataset("train_dist", data=train_dist)
            h5file.create_dataset("valid_dist", data=valid_dist)
            h5file.create_dataset("test_dist", data=test_dist)

            h5file.create_dataset("train_tpts", data=train_tpts)
            h5file.create_dataset("valid_tpts", data=valid_tpts)
            h5file.create_dataset("test_tpts", data=test_tpts)

    def load_data(self, fpath):
        with h5py.File(fpath, "r") as h5file:

            # Load the data
            train_dist = to_tensor(h5file["train_dist"][()])
            valid_dist = to_tensor(h5file["valid_dist"][()])
            test_dist = to_tensor(h5file["test_dist"][()])

            train_tpts = to_tensor(h5file["train_tpts"][()])
            valid_tpts = to_tensor(h5file["valid_tpts"][()])
            test_tpts = to_tensor(h5file["test_tpts"][()])

        train_data = (train_dist, train_tpts)
        valid_data = (valid_dist, valid_tpts)
        test_data = (test_dist, test_tpts)

        return train_data, valid_data, test_data
    
    def setup(self, stage=None):
        hps = self.hparams
        # Load data arrays from file
        if hps.make_data:
            timepoints, trajectory = self.model.make_trajectory(
                n=hps.n_samples * hps.n_timesteps,
                resample=True,
                pts_per_period=hps.pts_per_period,
                return_times = True, #False,
                noise=hps.noise,
            )
            trajectory = trajectory.reshape(hps.n_samples, hps.n_timesteps, -1)
            timepoints = timepoints.reshape(hps.n_samples, hps.n_timesteps, -1)

            inds = np.arange(hps.n_samples)
            train_inds, test_inds = train_test_split(
                inds, test_size=0.2, random_state=hps.seed
            )
            train_inds, valid_inds = train_test_split(
                train_inds, test_size=0.2, random_state=hps.seed
            )

            train_dist = trajectory[train_inds]
            valid_dist = trajectory[valid_inds] 
            test_dist = trajectory[test_inds]

            train_tpts = timepoints[train_inds]
            valid_tpts = timepoints[valid_inds] 
            test_tpts = timepoints[test_inds]

            train_data = (train_dist, train_tpts)
            valid_data = (valid_dist, valid_tpts)
            test_data = (test_dist, test_tpts)

            if hps.save_data:
                self.save_data(self.fpath, train_data, valid_data, test_data)
        else:
            train_data, valid_data, test_data = self.load_data(self.fpath)

        # Store datasets
        train_dist, train_tpts = train_data
        valid_dist, valid_tpts = valid_data
        test_dist, test_tpts = test_data

        if (~torch.is_tensor(train_dist) == False) or (~torch.is_tensor(train_tpts) == False):
            train_dist = to_tensor(train_dist)
            train_tpts = to_tensor(train_tpts)
        if (~torch.is_tensor(valid_dist) == False) or (~torch.is_tensor(valid_tpts) == False):
            valid_dist = to_tensor(valid_dist)
            valid_tpts = to_tensor(valid_tpts)
        if (~torch.is_tensor(test_dist) == False) or (~torch.is_tensor(test_tpts) == False):
            test_dist = to_tensor(test_dist)
            test_tpts = to_tensor(test_tpts)

        self.train_ds = TensorDataset(train_dist, train_tpts)
        self.valid_ds = TensorDataset(valid_dist, valid_tpts)
        self.test_ds = TensorDataset(test_dist, test_tpts)
    
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