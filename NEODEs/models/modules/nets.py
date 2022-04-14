import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.linear1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, output_dim, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x