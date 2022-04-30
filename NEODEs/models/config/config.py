import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
from torchdyn.core import NeuralODE
from data import NeuralODEDataModule
from example_params import datamodule_params

class NeuralODEConfig:

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 3 #2
    NUM_OUTPUTS = 3 # 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 200 #3.9

    POPULATION_SIZE = 50 #150 #150 #for testing
    NUMBER_OF_GENERATIONS = 100 #50 #150 # for testing
    SPECIATION_THRESHOLD = 200 #3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.10 #0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.4 #0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30

    def get_dataloader(self, datamodule, phase_flag):

        #Prepare the datamodule:
        datamodule.prepare_data()
        datamodule.setup()

        if phase_flag == "train":
            return datamodule.train_dataloader()
        elif phase_flag == "val":
            return datamodule.val_dataloader()

    def get_data(self, datamodule):    
        train_ds = datamodule.train_ds.tensors
        val_ds = datamodule.val_ds.tensors
        test_ds = datamodule.test_ds.tensors

        return train_ds, val_ds, test_ds

    def fitness_fn(self, genome):
        fitness = 1000.0 * 150 #this is arbitrary and will need to be tuned to a more appropriate value 
                        #4.0  # Max fitness for XOR

        phenotype = FeedForwardNet(genome, self)
        model = NeuralODE(phenotype, 
                                sensitivity='adjoint', 
                                solver='tsit5', 
                                #interpolator=None, 
                                atol=1e-3, rtol=1e-3)

        phenotype.to(self.DEVICE)
        #train_data, val_data, test_data = self.get_data(datamodule)

        #Instantiate the dataloader:
        dataloader = self.get_dataloader(NeuralODEDataModule(**datamodule_params), "train")
        #should we have two different datasets: one for evolutionary training, and one for weight training?

        for (batch_idx, batch) in enumerate(dataloader):

            x, t = batch

            batch_size, _, _ = x.size()
            x_hat = torch.zeros_like(x)
            t_spans = torch.zeros(batch_size, 2)

            for bat in range(batch_size):
                batch_times = t[bat].squeeze()
                t_spans[bat,0] = batch_times[0]
                t_spans[bat,1] = batch_times[-1]

            #these should be inside, but its faster without and performance seems fine
            _, x_out  = model(x[bat].squeeze(), t_spans[bat])

            x_hat[bat] = x_out[-1]

            #print(x_hat)

            mse = nn.MSELoss()
            loss = mse(x_hat, x)

            #import pdb; pdb.set_trace();

            fitness -= loss


            #create a lower bound for the fitness:
            if fitness <= 0:
                fitness = 0

        return fitness.detach().numpy()

    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels