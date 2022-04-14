import torch
import matplotlib.pyplot as plt
from torchdyn.datasets import *
from torchdyn.utils import *

from learners import BaseLearner
from models.Vanilla_NeuralODE import VanillaNeuralODE
import pytorch_lightning as pl

#much of this comes from this tutorial notebook:
#https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb
#This is just a script for testing models to make sure they train properly.

import pdb; pdb.set_trace()

#generate a toy dataset to test training the model:
d = ToyDataset()
X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.1)

t_span = t_span = torch.linspace(0, 1, 2)

#Plot the data:
colors = ['orange', 'blue'] 
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], s=1, color=colors[yn[i].int()])
plt.tight_layout()
#plt.show()

#instantiate model and learner:
model = VanillaNeuralODE(2, 64, 2)
learn = BaseLearner(X, yn, t_span, model)

#Train the model:
trainer = pl.Trainer(min_epochs=200, max_epochs=250, progress_bar_refresh_rate=1)
trainer.fit(learn)

#plot the results of a trained model:
X_train = torch.Tensor(X)
y_train = torch.LongTensor(yn.long())
t_span = torch.linspace(0,1,100)

t_eval, trajectory = model.forward(X_train, t_span)
trajectory = trajectory.detach()
plot_2D_depth_trajectory(t_span, trajectory, yn, len(X))
plt.show()
