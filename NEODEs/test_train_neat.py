import logging
import matplotlib.pyplot as plt
from torchdyn.datasets import *
from torchdyn.utils import *
from data import NeuralODEDataModule
from models.NEAT_NeuralODE import NEAT_NeuralODE
from models.config import config as c
from neat.visualize import draw_net
import neat.population as pop
from tqdm import tqdm
from utils import NEODE_fwd, get_similarity_score
from plotting import plot_3D_trajectories
from learners import TrajectoryLearner
from pytorch_lightning import Trainer

from example_params import (datamodule_params,
                            model_params,
                            callbacks, 
                            loggers, 
                            trainer_params)

#much of this comes from this tutorial notebook:
#https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb
#This is just a script for testing models to make sure they train properly.

log = logging.getLogger(__name__)

#import pdb; pdb.set_trace();
#Instantiate datamodule:
log.info("Instantiating datamodule...")
node_datamodule = NeuralODEDataModule(**datamodule_params)
node_datamodule.setup()

#Instantiate model:
log.info("Instantiating model...")

# Do a run with evolutionary training to get the evolved FFN:
num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000

c_neat = c.NeuralODEConfig()
#c_neat.add_data(node_datamodule) #should now be handled within NeuralODEConfig()

neat = pop.Population(c_neat)

#time the evolution:
import time
startTime = time.time()

solution, generation = neat.run()

executionTime = (time.time() - startTime)

#import pdb; pdb.set_trace();

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


#do a forward pass of the model on the validation dataset:
model = NEAT_NeuralODE(solution, c_neat)

valid_data, valid_times = node_datamodule.valid_ds.tensors
node_data, node_times = NEODE_fwd(model, node_datamodule)

csim = get_similarity_score(valid_data, node_data)



node_data = node_data.detach().numpy()

#plot neural ODE trajectories after just doing architectural optimization:
plot_3D_trajectories(valid_data, node_data, model_name = "NEAT Neural ODE")

print('Similarity Score:', csim)
print('Execution time for NEAT Neural ODE evolution in seconds: ' + str(executionTime))

#Pass the evolved FFN solution into a Neural ODE wrapper:
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

#import pdb; pdb.set_trace();

# Fit the trainer using the model and datamodules:
log.info("Starting training.")

#time the evolution:
import time
startTime = time.time()

trainer.fit(model = learn, datamodule=node_datamodule)

executionTime = (time.time() - startTime)
print('Execution time for NEAT Neural ODE weight training in seconds: ' + str(executionTime))