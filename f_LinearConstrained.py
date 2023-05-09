import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# state equation dynamics mapping ([x_k, u_k, d_k] -> x_{k+1})
# many to one RNN

class f_LinCons(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(f_LinCons, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.A_prime = nn.Parameter(torch.rand(self.input_dim, self.output_dim))
        self.M_prime = nn.Parameter(torch.rand(self.input_dim, self.output_dim))

    def effective_A(self):

        epsilon = 0.1
        M = 1 - epsilon * torch.sigmoid(self.M_prime)
        A_tilde = torch.nn.functional.softmax(self.A_prime, dim=1) * M
        return A_tilde.T
    
    def forward(self, x):
        # multiplying the input by the constrained A matrix
        return torch.mm(x, self.effective_A())