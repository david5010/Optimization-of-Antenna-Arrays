import torch
import math
import numpy as np

class ScaleCost:
    """
    Scale Only the cost
    """
    def __init__(self, mean = None, std = None) -> None:
        self.mean = mean
        self.scale = std

    def fit(self, data):
        self.mean = np.mean(data)
        self.scale = np.std(data)
    
    def transform(self, data):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted")
        return (data - self.mean) / self.scale

    def inverse_transform(self, scaled_data):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted")
        return (scaled_data * self.scale) + self.mean

    def save_scaler(self, path):
        """
        Saves the fitted parameters
        """
        scaler_attributes = {
            'mean': self.mean,
            'scale': self.scale
        }
        torch.save(scaler_attributes, path)

    def load_scaler(self, path):
        scaler_attributes = torch.load(path)
        self.mean = scaler_attributes['mean']
        self.scale = scaler_attributes['scale']

class ScaleYZCost:
    """
    Scale All
    """
    def __init__(self, y_mean = None, z_mean = None,
                 y_scale = None, z_scale = None,
                 cost_mean = None, cost_scale = None) -> None:
        self.y_mean = y_mean
        self.z_mean = z_mean
        self.y_scale = y_scale
        self.z_scale = z_scale
        self.cost_mean = cost_mean
        self.cost_scale = cost_scale

    def fit(self, y_data, z_data, cost_data):
        """
        Don't count the 0
        """
        y_mask = (y_data != 0)
        non_padded_y = y_data[y_mask]
        self.y_mean = non_padded_y.mean()
        self.y_scale = non_padded_y.std()

        z_mask = (z_data != 0)
        non_padded_z = z_data[z_mask]
        self.z_mean = non_padded_z.mean()
        self.z_scale = non_padded_z.std()

        self.cost_mean = np.mean(cost_data)
        self.cost_scale = np.std(cost_data)
    
    def transform(self, y_data, z_data, cost_data):
        scaled_y = np.zeros_like(y_data)
        scaled_z = np.zeros_like(z_data)
        scaled_cost = (cost_data - self.cost_mean) / self.cost_scale
        
        y_mask = (y_data != 0)
        z_mask = (z_data != 0)
        
        scaled_y[y_mask] = (y_data[y_mask] - self.y_mean) / self.y_scale
        scaled_z[z_mask] = (z_data[z_mask] - self.z_mean) / self.z_scale
        
        scaled_cost = scaled_cost.reshape(-1, 1)
        combined = np.concatenate((scaled_y, scaled_z, scaled_cost), axis=1)
        
        return combined


    def inverse_transform(self, scaled_y, scaled_z, scaled_cost):
        y = (scaled_y * self.y_scale) + self.y_mean
        z = (scaled_z * self.z_scale) + self.z_mean
        cost = (scaled_cost * self.cost_scale) + self.cost_mean

        return np.concatenate((y,z,cost), axis = 1)
    
    def get_stats(self):
        stats = {
            'y_mean': self.y_mean,
            'z_mean': self.z_mean,
            'cost_mean': self.cost_mean,
            'y_scale': self.y_scale,
            'z_scale': self.z_scale,
            'cost_scale': self.cost_scale
        }
        return stats

    def save_scaler(self, path):
        """
        Saves the fitted parameters
        """
        scaler_attributes = {
            'mean': self.mean,
            'scale': self.scale
        }
        torch.save(scaler_attributes, path)

    def load_scaler(self, path):
        scaler_attributes = torch.load(path)
        self.mean = scaler_attributes['mean']
        self.scale = scaler_attributes['scale']

