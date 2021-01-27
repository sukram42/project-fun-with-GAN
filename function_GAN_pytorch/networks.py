import torch
from torch import nn, optim
from torch.nn import functional as F


class SimpleDescriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, lr_disc=0.001):
        super(SimpleDescriminator, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dim)

        self.optimizer = optim.SGD(self.parameters(), lr=lr_disc)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return torch.sigmoid(self.output(x))


class SimpleGenerator(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim,lr_gen=0.001):
        super(SimpleGenerator, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_gen)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        # x = F.leaky_relu(self.linear2(x))
        return torch.tanh(self.output(x))