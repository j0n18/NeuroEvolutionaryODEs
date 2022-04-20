import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl

class ClassificationLearner(pl.LightningModule):
    def __init__(self, 
                X_data: torch.Tensor, 
                y_data: torch.Tensor,
                t_span:torch.Tensor, 
                model:nn.Module):

        super().__init__()
        self.model, self.t_span = model, t_span
        self.X_data = X_data
        self.y_data = y_data
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        import pdb; pdb.set_trace();
        x, y = batch      
        t_eval, y_hat = self.model(x, self.t_span)
        y_hat = y_hat[-1] # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return {'loss': loss}   
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):

        X = self.X_data
        yn = self.y_data

        X_train = torch.Tensor(X)
        y_train = torch.LongTensor(yn.long())
        
        train = data.TensorDataset(X_train, y_train)
        trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

        return trainloader


class TrajectoryLearner(pl.LightningModule):
    def __init__(self, 
                datamodule: pl.LightningDataModule,
                model:pl.LightningModule):

        super().__init__()

        self.model = model
        self.datamodule = datamodule

        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x, t):

        batch_size, _, _ = x.size()
        x_hat = torch.zeros_like(x)
        t_spans = torch.zeros(batch_size, 2)

        for bat in range(batch_size):
            batch_times = t[bat].squeeze()
            t_spans[bat,0] = batch_times[0]
            t_spans[bat,1] = batch_times[-1]

            #Assuming the NeurlODE class is using model(x,t) and not model.trajectory(x,t):
            _, x_out  = self.model(x[bat].squeeze(), t_spans[bat])

            x_hat[bat] = x_out[-1]

        return x_hat, t_spans
    
    def training_step(self, batch, batch_idx):
        x, t = batch

        #NeuralODE model handles single sample operations,
        #so batch the data in here:

        x_hat, _ = self.forward(x,t)

        #loss: difference between true trajectory
        #and the evolved Neural ODE trajectory
        mse = nn.MSELoss()
        loss = mse(x_hat, x)

        self.log_dict(
            {
                "train_loss": loss,
            }
        )

        return loss  

    def validation_step(self, batch, batch_idx):

        x, t = batch 
        
        #NeurlODE model handles single sample operations,
        #so batch the data in here:

        x_hat, _ = self.forward(x,t)

        #import pdb; pdb.set_trace();

        #loss: difference between true trajectory
        #and the evolved Neural ODE trajectory
        mse = nn.MSELoss()
        loss = mse(x_hat, x)

        self.log_dict(
            {
                "val_loss": loss,
            }
        )

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        return optimizer

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()