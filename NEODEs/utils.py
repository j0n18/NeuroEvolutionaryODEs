import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

def instantiate_dataloader(datamodule, phase_flag):

    #Prepare the datamodule:
    datamodule.prepare_data()
    datamodule.setup()

    if phase_flag == "train":
        return datamodule.train_dataloader()
    elif phase_flag == "val":
        return datamodule.val_dataloader()
    elif phase_flag == "test":
        return datamodule.test_dataloader()



def NEODE_fwd(model, datamodule, phase_flag = "val"):

    dataloader = instantiate_dataloader(datamodule, phase_flag)

    if phase_flag == "val":
        n_samples, *_ = datamodule.valid_ds.tensors[0].size()
        num_batches = n_samples // datamodule.hparams.batch_size
        x_node = torch.zeros_like(datamodule.valid_ds.tensors[0])
        t_node = torch.zeros((n_samples, 2))

    elif phase_flag == "train":
        n_samples, *_ = datamodule.valid_ds.tensors[0].size()
        num_batches = n_samples // datamodule.hparams.batch_size
        x_node = torch.zeros_like(datamodule.train_ds.tensors[0])
        t_node = torch.zeros((n_samples, 2))

    elif phase_flag == "test":
        n_samples, *_ = datamodule.test_ds.tensors[0].size()
        num_batches = n_samples // datamodule.hparams.batch_size
        x_node = torch.zeros_like(datamodule.test_ds.tensors[0])
        t_node = torch.zeros((n_samples, 2))

    
    for B in range(1,num_batches + 1):
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

            x_node[(B-1) * batch_size : B * batch_size] = x_hat
            t_node[(B-1) * batch_size : B * batch_size] = t_spans

    return x_node, t_node


def get_similarity_score(valid_data, node_data):
    '''
    Similarity is measured using the average cosine similarity
    evaluated at each point along the trajectory.
    '''

    valid_data = valid_data.detach()
    node_data = node_data.detach()

    valid_data = valid_data.reshape(-1, valid_data.shape[-1])
    node_data = node_data.reshape(-1, node_data.shape[-1])

    mse = torch.mean((node_data - valid_data)**2)

    return mse #F.cosine_similarity(valid_data,node_data).mean()


