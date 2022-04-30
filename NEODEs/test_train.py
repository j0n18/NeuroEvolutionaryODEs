import logging
import matplotlib.pyplot as plt
from torchdyn.datasets import *
from torchdyn.utils import *
from data import NeuralODEDataModule
from pytorch_lightning import Trainer
from utils import NEODE_fwd, get_similarity_score
from plotting import plot_3D_trajectories

from learners import TrajectoryLearner
from models.Vanilla_NeuralODE import VanillaNeuralODE

from example_params import (datamodule_params,
                            model_params,
                            callbacks, 
                            loggers, 
                            trainer_params)

#much of this comes from this tutorial notebook:
#https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb
#This is just a script for testing models to make sure they train properly.

log = logging.getLogger(__name__)

#import pdb; pdb.set_trace()

#Instantiate datamodule:
log.info("Instantiating datamodule...")
node_datamodule = NeuralODEDataModule(**datamodule_params)
node_datamodule.setup()

#Instantiate model:
log.info("Instantiating model...")
model = VanillaNeuralODE(**model_params)
learn = TrajectoryLearner(node_datamodule, model)

# instantiate the callbacks: (already done by the import)
log.info("Instantiating callbacks...")
callbacks = callbacks

# instantiate the loggers: (already done by the import)
log.info("Instantiating loggers...")
loggers = loggers

# instantiate trainer:
log.info("Instantiating Trainer...")
trainer = Trainer(
    **trainer_params,
    callbacks=callbacks,
    #logger=loggers
)

# Fit the trainer using the model and datamodules:
log.info("Starting training.")


import time
startTime = time.time()

trainer.fit(model = learn, datamodule=node_datamodule)

executionTime = (time.time() - startTime)
print('Execution time for Vanilla Neural ODE in seconds: ' + str(executionTime))

valid_data, valid_times = node_datamodule.valid_ds.tensors
node_data, node_times = NEODE_fwd(model, node_datamodule)

plot_3D_trajectories(valid_data, node_data, model_name = "NEAT Neural ODE")

r2 = get_similarity_score(valid_data, node_data)

print('Similarity Score:', r2)

#import pdb; pdb.set_trace();