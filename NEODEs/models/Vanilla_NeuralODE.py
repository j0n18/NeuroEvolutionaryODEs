import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from models.modules.nets import MLP
import torch

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
        #Note: NeuralODE() returns t_vals, sol by default
        #https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/core/neuralde.py#L33

        self.ffn = MLP(input_dim, hidden_size, output_dim)
        self.model = NeuralODE(self.ffn, 
                                sensitivity='adjoint', 
                                solver='tsit5', 
                                #interpolator=None, 
                                atol=1e-3, rtol=1e-3)

        self.save_hyperparameters()

    def forward(self, x, t_span):
        import pdb; pdb.set_trace();

        return self.model.trajectory(x,t_span) #[-1]

        
        #^why is this output the wrong size in the first dimension?
        #the size change is within this function:
        #https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/odeint.py
        #https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/odeint.py#L88
        #https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/odeint.py#L385
        #^It is stacking iterations of solutions and returning that. Why?
        #First condition of this must be satisfied, because the other is False by default:
        #https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/odeint.py#L384
        #https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/odeint.py#L387

        #it gives a full n_timesteps evaluation from every point inside t_span back.
        #What do we do with that? (this sounds familiar)
        
        #for shapes to make sense with TrajectoryLearner, we somehow need to return something with the same
        #shape as an element of x[batch] : n_timesteps x dim
