import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from NEODEs.pytorch_neat.neat.phenotype.feed_forward import FeedForwardNet
import torch

class NEAT_NeuralODE(pl.LightningModule):


    def __init__(self, genome, config):

        super().__init__()

        #instantiate the FFN we want to use in the Neural ODE:
        #Note: NeuralODE() returns t_vals, sol by default
        #https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/core/neuralde.py#L33

        self.ffn = FeedForwardNet(genome, config)
        self.model = NeuralODE(self.ffn, 
                                sensitivity='adjoint', 
                                solver='tsit5', 
                                #interpolator=None, 
                                atol=1e-3, rtol=1e-3)

        self.save_hyperparameters()

    def forward(self, x, t_span):
        #import pdb; pdb.set_trace();

        return self.model(x,t_span) #.trajectory(x,t_span) #[-1]