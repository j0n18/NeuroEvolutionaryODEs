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
        #import pdb; pdb.set_trace();
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
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, t = batch   

        #NeurlODE model handles single sample operations,
        #so batch the data in here:

        batch_size, _, _ = x.size()
        x_hat = torch.zeros_like(x)

        for bat in range(batch_size):
            x_hat[bat] = self.model(x[bat].squeeze(), t[bat].squeeze())

        #loss: difference between true trajectory
        #and the evolved Neural ODE trajectory
        mse = nn.MSELoss()
        loss = mse(x_hat, x)

        return {'loss': loss}   

    def validation_step(self, batch, batch_idx):

        import pdb; pdb.set_trace();

        x, t = batch 
        
        #NeurlODE model handles single sample operations,
        #so batch the data in here:

        batch_size, _, _ = x.size()
        x_hat = torch.zeros_like(x)

        for bat in range(batch_size):
            x_hat[bat] = self.model(x[bat].squeeze(), t[bat].squeeze())

        #loss: difference between true trajectory
        #and the evolved Neural ODE trajectory
        mse = nn.MSELoss()
        loss = mse(x_hat, x)

        return {'loss': loss} 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()