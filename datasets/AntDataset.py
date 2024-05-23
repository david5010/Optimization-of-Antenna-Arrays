from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os

class AntDataset(Dataset):
    def __init__(self, data, shuffle = False, seed = 123):
        if not isinstance(data,str):
            self.data = data.astype(np.float32)
        elif os.path.splitext(data)[-1] == '.csv':
            self.data = pd.read_csv(data, header = None).values.astype(np.float32)
        elif os.path.splitext(data)[-1] == '.npz':
            self.data = np.load(data)['data'].astype(np.float32)
        else:
            self.data = np.load(data)['data'].astype(np.float32)

        self.shuffle = shuffle

        if self.shuffle:
            # Shuffle the antennas around
            num_ant = (self.data.shape[1]-1)//2
            index_order = np.random.RandomState(seed=seed).permutation(num_ant)
            X, Y, label = self.data[:, :num_ant], self.data[:, num_ant:-1], self.data[:,-1].reshape(-1,1)
            self.data = np.hstack((X[:, index_order], Y[:, index_order], label[:, -1].reshape(-1, 1)))



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = self.data[index, :-1]
        label = self.data[index, -1]
        return torch.from_numpy(features), torch.from_numpy(np.array(label))
    
class AntDataset2D(Dataset):
    def __init__(self, data, shuffle = False, seed = 123):
        if os.path.splitext(data)[-1] == '.csv':
            self.data = pd.read_csv(data, header= None).values.astype(np.float32)
        elif os.path.splitext(data)[-1] == '.npz':
            self.data = np.load(data)['data'].astype(np.float32)
        
        self.shuffle = shuffle
        if self.shuffle:
            # Shuffle the antennas around
            num_ant = (self.data.shape[1]-1)//2
            index_order = np.random.RandomState(seed=seed).permutation(num_ant)
            X, Y, label = self.data[:, :num_ant], self.data[:, num_ant:-1], self.data[:,-1].reshape(-1,1)
            self.data = np.hstack((X[:, index_order], Y[:, index_order], label[:, -1].reshape(-1, 1)))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the array pattern and target from the CSV data
        array_pattern = self.data[idx, :-1]
        target = self.data[idx, -1]
        
        # Reshape the array pattern and convert to tensor
        sequence_length = 1024
        input_dim = 2
        array_pattern = array_pattern.reshape(input_dim, sequence_length).T
        array_pattern = torch.from_numpy(array_pattern)
        
        # Convert the target to tensor
        target = torch.from_numpy(np.array(target))
        
        return array_pattern, target