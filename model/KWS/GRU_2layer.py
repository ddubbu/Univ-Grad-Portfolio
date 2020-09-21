## Multi-label GRU model 구현
# GRU layer
''' import libraries '''

#%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms  # 1 batch = (1, 784)
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
import numpy as np
#from torchsummary import summary

n_mfcc = 24
Tx = 1723

class KWS_2GRU(nn.Module):
    def __init__(self, p=0.0):
        super(KWS_2GRU, self).__init__()
        self.num_feature = n_mfcc
        self.hidden_size = 128
        self.conv = nn.Conv1d(in_channels=n_mfcc, out_channels=self.num_feature,
                              kernel_size=3, stride=1, padding=1)  # Tx = 1723 = Ty 유지
        self.bn_num_feature = nn.BatchNorm1d(self.num_feature)  # [N,C,L], what is channel?
        self.bn_Tx = nn.BatchNorm1d(Tx)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

        self.GRU1 = nn.GRU(self.num_feature, self.hidden_size,
                           num_layers=1, batch_first=True)
        self.GRU2 = nn.GRU(self.hidden_size, self.hidden_size,
                           num_layers=1, batch_first=True)
        self.linear = nn.Linear(Tx * self.hidden_size, Tx)


    def forward(self, x):

        x = torch.transpose(x, 1, 2)  # [batch_size, Tx, n_mfcc] -> [batch_size, n_mfcc, Tx]
        x = self.conv(x)  # -> [batch_size, self.num_feature, Tx]
        output = nn.Sequential(
                                self.bn_num_feature, self.relu, self.dropout,
                                )(x)
        output = torch.transpose(output, 1, 2)  # -> [b, Tx, self.num_feature]


        output, hidden = self.GRU1(output)
        output = nn.Sequential(self.dropout, self.bn_Tx)(output)
        output, hidden = self.GRU2(output, hidden)
        output = nn.Sequential(self.dropout, self.bn_Tx, self.dropout)(output)
        output = output.view(-1, Tx * self.hidden_size)
        output = self.linear(output)

        return output