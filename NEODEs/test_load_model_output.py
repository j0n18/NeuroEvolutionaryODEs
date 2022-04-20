import logging
import matplotlib.pyplot as plt
from torchdyn.datasets import *
from torchdyn.utils import *
from plotting import plot_3D_trajectories
from data import NeuralODEDataModule

from learners import TrajectoryLearner
from models.Vanilla_NeuralODE import VanillaNeuralODE

from example_params import (datamodule_params,
                            model_params)


log = logging.getLogger(__name__)

#import pdb; pdb.set_trace()

chkpt = "C:\\Users\\jonat\\OneDrive\\Desktop\\Deep Learning\\Project\\NeuroEvolutionaryODEs\\NEODEs\\chkpts\\epoch=59-step=600.ckpt"

node_datamodule = NeuralODEDataModule(**datamodule_params)
node_datamodule.setup()

model = VanillaNeuralODE(**model_params)
trained_learn = TrajectoryLearner.load_from_checkpoint(chkpt, model = model, datamodule = node_datamodule)
trained_learn.freeze()

val_dataloader = node_datamodule.val_dataloader()

valid_data, valid_times = node_datamodule.valid_ds.tensors

node_data, node_times = trained_learn.forward(valid_data, valid_times)

import pdb; pdb.set_trace();

plot_3D_trajectories(valid_data, node_data)