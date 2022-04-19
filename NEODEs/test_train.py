import logging
import matplotlib.pyplot as plt
from torchdyn.datasets import *
from torchdyn.utils import *
from data import NeuralODEDataModule
from pytorch_lightning import Trainer

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
    logger=loggers
)

# Fit the trainer using the model and datamodules:
log.info("Starting training.")
trainer.fit(model = learn, datamodule=node_datamodule)