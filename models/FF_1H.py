import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class SimpleNN(nn.Module):
    # in the future, implement one with configuration files for ease
    def __init__(self, input_size, output_size, hidden_size , act_func, dropout = 0) -> None:
        super(SimpleNN, self).__init__()
        self.name = 'FF'
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_func(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.regressor(x)

class NN2(nn.Module):
    # Two-layers
    def __init__(self, input_size, output_size, hid1_size, hid2_size, act_func,dropout = 0) -> None:
        super(NN2, self).__init__()
        self.name = 'FF'
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hid1_size),
            act_func(),
            nn.Dropout(p=dropout),
            nn.Linear(hid1_size, hid2_size),
            act_func(),
            nn.Dropout(p=dropout),
            nn.Linear(hid2_size, output_size)
        )

    def forward(self, x):
        return self.regressor(x)
    
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size,1)
    
    def forward(self, x):
        out = self.linear(x)
        return out