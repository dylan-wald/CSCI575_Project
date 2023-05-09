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
from f_LinearConstrained import f_LinCons
# state equation dynamics mapping ([x_k, u_k, d_k] -> x_{k+1})
# many to one RNN

class f_NSSM(nn.Module):
    def __init__(self, state_dim, input_dim, disturbance_dim): 
        super(f_NSSM, self).__init__()

        self.A = f_LinCons(state_dim, state_dim)
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.E = nn.Linear(disturbance_dim, state_dim, bias=False)
    
    def forward(self, x, U, D):
    
        X = []
        Y = []
        for u, d in zip(U, D):
            x = self.A(x) + self.B(u) + self.E(d)
            y = x

            X.append(x)
            Y.append(y)
        
        return torch.stack(X), torch.stack(Y)