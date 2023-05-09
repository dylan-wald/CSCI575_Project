
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


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Variable 
import torch
from torch.utils.data import TensorDataset, DataLoader

from f_GRU import f_GRU
from f_RNN import f_RNN
from f_NSSM import f_NSSM

from data_loader import load_data
from plotting import plotting
from plot_all import plot_all

# Load the data
data = load_data(file="sys_id_data.csv", num_days=20)

# split data into train and test sets
X_train, y_train, X_test, y_test = data.data_split(day_train=10)
print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

# choose which normalization scheme to use
norm_all = True
norm = False
no_norm = False
if norm_all:
    X_train = load_data.min_max_norm_all(X_train, 0, 1, X_train.shape[-1])
    X_test = load_data.min_max_norm_all(X_test, 0, 1, X_test.shape[-1])
if norm:
    X_train = load_data.min_max_norm(X_train, 0, 1)
    X_test = load_data.min_max_norm(X_test, 0, 1)
if no_norm:
    None

input_dim = X_train.shape[2]
output_dim = y_train.shape[2]

state_dim = 1
disturbance_dim = 3

# placeholders
Loss_all_plot = []
Y_pred_all_plot = []
Y_pred_test_all = []
A_all_plot = []
B_all_plot = []
E_all_plot = []

# individual parameter for each model
model_list = ["RNN_model", "NSSM_model", "GRU_model"]
for mod_name in model_list:
    model = 0
    if mod_name == "RNN_model":
        input_dim = 5
        learn_rate = 0.0005
        num_epochs = 15000
        model = f_RNN(input_dim, state_dim)
    if mod_name == "GRU_model":
        input_dim = 5
        learn_rate = 0.0006
        num_epochs = 25000
        model = f_GRU(input_dim, state_dim)
    if mod_name == "NSSM_model":
        input_dim = 2
        learn_rate = 0.0005
        num_epochs = 15000
        model = f_NSSM(state_dim, input_dim, disturbance_dim)

    cost = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    # initial condtition to the model
    x0 = y_train[0, :, :]
    # inputs to each cell
    U = X_train[:, :, :2]
    # inputs to each cell
    D = X_train[:, :, 2:]

    # shift the output temperatures by one timestep (track the next timesteps' Tz)
    Y_true = y_train
    
    # train the model
    loss_all = []
    for epoch in range(num_epochs):
        model.zero_grad()
        X_pred, Y_pred = model(x0, U, D)
        loss = cost(Y_pred, Y_true)
        loss.backward()
        optimizer.step()

        loss_all.append(loss.detach().numpy())

        if epoch % 100 == 0:
            print("Epock {} loss: {}".format(epoch, loss))
    Loss_all_plot.append(loss_all)

    # evaluate the training data
    X_out, Y_out = model(x0, U, D)
    Y_pred_all_plot.append(Y_out)

    # evaluate the testing data
    x0_test = y_test[0, :, :]
    U_test = X_test[:, :, :2]
    D_test = X_test[:, :, 2:]
    X_out_test, Y_out_test = model(x0_test, U_test, D_test)
    Y_pred_test_all.append(Y_out_test)

    if mod_name == "RNN_model":
        BE = model.cell.weight_ih.squeeze(0).detach().numpy().reshape((1,5))
        B = BE[0][:2].reshape(1,2)
        E = BE[0][2:].reshape(1,3)
        A = model.cell.weight_hh.squeeze(0).detach().numpy()
        print(np.shape(A))
        print("A:",np.diag(A))
        print(np.shape(B))
        print("B:", B)
        print(np.shape(E))
        print("E:", E)
    elif mod_name == "NSSM_model":
        A = model.A.effective_A().squeeze(0).detach().numpy().reshape((1,1))
        B = model.B.weight.squeeze(0).detach().numpy().reshape((1,2))
        E = model.E.weight.squeeze(0).detach().numpy().reshape((1,3))
        print(np.shape(A))
        print("A:",np.diag(A))
        print(np.shape(B))
        print("B:", B)
        print(np.shape(E))
        print("E:", E)
    else:
        A = None
        B = None
        E = None
    A_all_plot.append(A)
    B_all_plot.append(B)
    E_all_plot.append(E)

# Runs the plotting script to plot all results
plot_all(X_train, y_train, Loss_all_plot, Y_pred_all_plot, Y_true, Y_pred_test_all, y_test, 
        model_list, A_all=A_all_plot, B_all=B_all_plot, E_all=E_all_plot)