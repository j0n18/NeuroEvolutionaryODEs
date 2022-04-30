import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from neat.phenotype.feed_forward import FeedForwardNet
import torch
import torch.nn as nn

class NEATVectorField(nn.Module):

    def __init__(self, ffn):
        super().__init__()
        self.ffn = ffn

    def forward(self, t, x):

        n_timesteps, dim = x.size()

        output = torch.zeros_like(x)

        for t in range(n_timesteps): 
            output[t] = self.ffn(x[t].unsqueeze(0))

        return output

    def __call__(self, t, x):
        
        return self.forward(t,x)



class NEAT_NeuralODE(pl.LightningModule):

    def __init__(self, genome, config):

        super().__init__()

        self.ffn = FeedForwardNet(genome, config)
        self.vf = NEATVectorField(self.ffn)

        self.model = NeuralODE(self.vf, 
                                sensitivity='adjoint', 
                                solver='tsit5', 
                                interpolator=None, 
                                atol=1e-3, rtol=1e-3)

        self.save_hyperparameters()

    def forward(self, x, t_span):

        return self.model(x,t_span) 