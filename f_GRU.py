'''
Code inspired by work in:

J. Drgona, A. Tuor, and D. Vrabie, “Constrained physics-informed deep
learning for stable system identification and control of unknown linear
systems,” arXiv preprint arXiv:2004.11184, 2020.

and is based off of code in:
https://github.com/pnnl/deps_arXiv2020

and cited as:
J. Drgona, A. Tuor, and D. Vrabie, “Learning constrained adaptive
differentiable predictive control policies with guarantees,” 2022

'''

import torch
import torch.nn as nn
from torch.autograd import Variable

# state equation dynamics mapping ([x_k, u_k, d_k] -> x_{k+1})
# many to one GRU RNN

class f_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(f_GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim, bias=False)   
        
    def forward(self, x, U, D):
    
        X = []
        Y = []
        for u, d in zip(U, D):
            x = self.cell(torch.cat([u, d], dim=1), x)
            out = x
            X.append(x)
            Y.append(out)
        
        return torch.stack(X), torch.stack(Y)
    
