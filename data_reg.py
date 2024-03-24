import torch
from torch.utils.data import Dataset
from sklearn import datasets
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random
FONTSIZE = 13


class SinDataset(Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.data_x = torch.rand(1,num_samples) * 0.8 * 3.1416
        self.noise = 0.5*(torch.rand(num_samples)-0.5)
        self.data_y = torch.sin(2*torch.pow(self.data_x,3)).reshape(-1) + self.noise
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num

        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        x_train = self.data_x[:, self.train_id]
        x_train = x_train.reshape(-1)
        sorted, index = torch.sort(x_train)
        self.train_id = self.train_id[index]
        # index2 = [1, 15, 25, 39, 43, 46, 55, 60, 63, 66, 69, 72, 74, 78]
        index2 = [1, 25, 39, 55, 60, 69, 72, 74, 78]

        self.sv_id = self.train_id[index2]
        # idx = list(range(0, int(np.ceil(num_samples / 2)), 10)) + list(range(int(np.ceil(num_samples / 2)), self.train_num, 5))
        # self.sv_id = self.train_id[idx]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)
        xx = torch.linspace(0, 0.8 * 3.14156, 1000)
        yy =  torch.sin(2*torch.pow(xx, 3))
        # x_train = self.data_x[:,self.train_id]
        # y_train = self.data_y[self.train_id]
        # x_sv = self.data_x[:,self.sv_id]
        # y_sv = self.data_y[self.sv_id]
        plt.figure(5)
        ax=plt.subplot(2,2,1)
        plt.plot(xx,yy,'k:', label='Ground Truth')
        plt.plot(self.data_x[:, self.sv_id].reshape(-1), self.data_y[self.sv_id].reshape(-1), 'k+',
                 markersize=10)
        plt.plot(self.data_x[:, self.val_id].reshape(-1), self.data_y[self.val_id].reshape(-1), 'k+',
                 label='Sample Data (noised)', markersize=10)
        plt.legend(loc="lower left", fontsize=FONTSIZE)
        ax.set_title('(a) Ground Truth', fontsize=FONTSIZE)
        # plt.subplot(2, 2, 2)
        # plt.scatter(x_train, y_train, c='black', marker='+')
        # plt.scatter(x_sv,  y_sv, marker='o',facecolor='white',edgecolors='black')
        # plt.subplot(2, 2, 3)
        # plt.scatter(x_train, y_train, c='black', marker='+')
        # plt.scatter(x_sv,  y_sv, marker='o',facecolor='white',edgecolors='black')
        # plt.subplot(2, 2, 4)
        # plt.scatter(x_train, y_train, c='black', marker='+')
        # plt.scatter(x_sv,  y_sv, marker='o',facecolor='white',edgecolors='black')

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]
