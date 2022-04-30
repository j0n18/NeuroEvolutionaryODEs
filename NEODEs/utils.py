import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from neat.phenotype.feed_forward import FeedForwardNet
import torch
from data import NeuralODEDataModule
from sklearn.metrics import r2_score
import numpy as np

def instantiate_dataloader(datamodule, phase_flag):

    #Prepare the datamodule:
    datamodule.prepare_data()
    datamodule.setup()

    if phase_flag == "train":
        return datamodule.train_dataloader()
    elif phase_flag == "val":
        return datamodule.val_dataloader()



def NEODE_fwd(model, datamodule, phase_flag = "val"):

    #import pdb; pdb.set_trace();

    dataloader = instantiate_dataloader(datamodule, phase_flag)

    for (batch_idx, batch) in enumerate(dataloader):

        x, t = batch

        batch_size, _, _ = x.size()
        x_hat = torch.zeros_like(x)
        t_spans = torch.zeros(batch_size, 2)

        for bat in range(batch_size):
            batch_times = t[bat].squeeze()
            t_spans[bat,0] = batch_times[0]
            t_spans[bat,1] = batch_times[-1]

            _, x_out  = model(x[bat].squeeze(), t_spans[bat])

            x_hat[bat] = x_out[-1]

    return x_hat, t_spans


def get_r2_score(valid_data, node_data):

    valid_data = valid_data.detach().cpu().numpy()
    node_data = node_data.detach().cpu().numpy()
    valid_data = np.concatenate([*valid_data])
    node_data = np.concatenate([*node_data])
    r2 = r2_score(valid_data, node_data)

    return r2

