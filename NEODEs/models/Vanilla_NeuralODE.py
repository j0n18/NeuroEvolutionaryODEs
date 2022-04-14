import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from models.modules.nets import MLP

class VanillaNeuralODE(pl.LightningModule):
    '''
    Implements the basic implementation of a Neural ODE
    as used for the Vanilla Neural ODE (Depth-Invariant)
    section of this notebook:
    https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb

    Note: After instantiation, takes the data and timesteps as input.
    '''

    def __init__(self, input_dim, hidden_size, output_dim):

        super().__init__()

        #instantiate the FFN we want to use in the Neural ODE:
        self.ffn = MLP(input_dim, hidden_size, output_dim)
        self.model = NeuralODE(self.ffn, 
                                sensitivity='adjoint', 
                                solver='tsit5', 
                                interpolator=None, 
                                atol=1e-3, rtol=1e-3)

        self.save_hyperparameters()

    def forward(self, x, t_span):
        return self.model(x, t_span)
