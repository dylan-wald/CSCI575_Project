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

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

class load_data():

    def __init__(self, file, num_days):
        self.file = file
        self.num_days = num_days

        self.load()

    def load(self):
        df = pd.read_csv(self.file)
        print("Data loaded")

        # self.x = np.array(df.T_room_rom)
        # self.u = np.array(df.Q_hvac)
        # Q_solar = np.array(df.Q_solar)
        # T_oa = np.array(df.T_outside)
        # Q_int = np.array(df.Q_internal)
        # self.d = np.hstack((T_oa, Q_solar, Q_int))

        # states
        self.x = np.array(df["T_z [C]"].iloc[0:288*self.num_days])

        # inputs
        self.T_sa = np.array(df["T_sa [C]"].iloc[0:288*self.num_days])
        self.ms_dot = np.array(df["ms_dot [kg/s]"].loc[0:288*self.num_days])

        self.Q_solar = np.array(df["Q_solar [J]"].iloc[0:288*self.num_days])/(5*60)    # convert to kW (J/sec in 5 min/1000)
        self.T_oa = np.array(df["T_oa [C]"].iloc[0:288*self.num_days])
        self.Q_int = np.array(df["Q_int [W]"].iloc[0:288*self.num_days])               # convert to kW (W/1000)

        # removing weekends from the dataset (irregular temperature schedules)
        rm = np.linspace(4*288, 6*288-1, 2*288).astype(int)
        for i in range(1,3):
            rm = np.concatenate((rm, np.linspace(4*288+(i*7*288), 6*288+(i*7*288)-1, 2*288).astype(int)))
            

        self.x = np.delete(self.x, rm)

        self.T_sa = np.delete(self.T_sa, rm)
        self.ms_dot = np.delete(self.ms_dot, rm)

        self.Q_solar = np.delete(self.Q_solar, rm)
        self.T_oa = np.delete(self.T_oa, rm)
        self.Q_int = np.delete(self.Q_int, rm)
        self.d = np.hstack((self.T_oa, self.Q_solar, self.Q_int))


    def data_split(self, day_train):

        '''
        splitting the data into train and test sets
        '''

        X = np.hstack((
            self.T_sa.reshape(len(self.T_sa),1),
            self.ms_dot[:-1].reshape(len(self.ms_dot)-1,1),
            self.T_oa.reshape(len(self.T_oa),1),
            self.Q_solar.reshape(len(self.Q_solar),1),
            self.Q_int.reshape(len(self.Q_int),1)
            ))

        y = self.x.reshape(len(self.x),1)

        X_train = X[:day_train*288, :]
        y_train = y[:day_train*288, :]
        X_test = X[day_train*288:, :]
        y_test = y[day_train*288:, :]

        day_len = 288
        day_test = int(len(y_test)/day_len)
        horizon = 16

        X_train_reshape = np.zeros((horizon, 180, 5))
        y_train_reshape = np.zeros((horizon, 180, 1))
        X_test_reshape = np.zeros((horizon, 72, 5))
        y_test_reshape = np.zeros((horizon, 72, 1))
        for i in range(int(day_len/horizon)):
            for j in range(horizon):
                y_train_reshape[j, i*day_train:(i+1)*day_train, :] = y_train[np.arange((16*i)+j, len(y_train), day_len)]
                X_train_reshape[j, i*day_train:(i+1)*day_train, :] = X_train[np.arange((16*i)+j, len(y_train), day_len)]
                y_test_reshape[j, i*day_test:(i+1)*day_test, :] = y_test[np.arange((16*i)+j, len(y_test), day_len)]
                X_test_reshape[j, i*day_test:(i+1)*day_test, :] = X_test[np.arange((16*i)+j, len(y_test), day_len)]

        X_train_tensors = torch.tensor(X_train_reshape, dtype=torch.float32)
        y_train_tensors = torch.tensor(y_train_reshape, dtype=torch.float32)
        X_test_tensors = torch.tensor(X_test_reshape, dtype=torch.float32)
        y_test_tensors = torch.tensor(y_test_reshape, dtype=torch.float32)

        return X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors
    
    def min_max_norm(data, new_min, new_max):

        '''
        normalizing based on max input value
        '''
        
        x_min = data.min()
        x_max = data.max()
        x_norm = (data - x_min)/(x_max - x_min)*(new_max - new_min) + new_min

        return x_norm

    def min_max_norm_all(data, new_min, new_max, dim):

        '''
        normalizing all inputs individually
        '''
        
        for i in range(dim):
            X = data[:, :, i]
            x_min = X.min()
            x_max = X.max()
            data[:, :, i] = (X - x_min)/(x_max - x_min)*(new_max - new_min) + new_min

        return data