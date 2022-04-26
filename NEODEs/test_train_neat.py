import logging
import matplotlib.pyplot as plt
from torchdyn.datasets import *
from torchdyn.utils import *
from data import NeuralODEDataModule
from pytorch_lightning import Trainer

from learners import TrajectoryLearner
from models.Vanilla_NeuralODE import VanillaNeuralODE
from models.config import config as c
from pytorch_neat.neat.visualize import draw_net
import pytorch_neat.neat.population as pop
from tqdm import tqdm

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
num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000

c_neat = c.NeuralODEConfig()
c_neat.add_data(node_datamodule)

neat = pop.Population(c_neat)
solution, generation = neat.run()

if solution is not None:
    avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
    min_num_generations = min(generation, min_num_generations)

    num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
    avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)
    min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
    max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)
    if num_hidden_nodes == 1:
        found_minimal_solution += 1

    num_of_solutions += 1
    draw_net(solution, view=True, filename='./images/solution-' + str(num_of_solutions), show_disabled=True)

model = NEAT_NeuralODE(solution, c_neat)
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