import torch
import matplotlib.pyplot as plt
import numpy as np
import os
def save_figure(figure, title, directory='figures'):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    figure_path = os.path.join(directory, f"{title}.png")
    figure.savefig(figure_path)  # Save with reduced white space
    plt.close(figure)  # Close the figure to free up memory

def show_arrays(inp, title='Antenna Array Pattern', directory='figures'):
    fig, ax = plt.subplots()  # Use subplots for better control over the figure
    data = inp.cpu().detach().numpy()
    reshaped_data = np.reshape(data, (2, 1024))
    x, y = reshaped_data
    
    nonzero_ind = np.nonzero((x != 0) | (y != 0))
    x, y = x[nonzero_ind], y[nonzero_ind]
    
    ax.scatter(x, y, s=0.75)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title(title)
    
    save_figure(fig, title, directory)  # Pass the directory to save_figure


def calculate_min_distance(ants, unit = 1):
    """
    Checks the smallest Euclidean distance, if it violates then return True
    """
    reshaped_tensor = torch.reshape(ants, (2, 1024))

# Transpose the reshaped tensor
    reshaped_tensor = torch.transpose(reshaped_tensor, 0, 1)
    nonzero_rows = torch.any(reshaped_tensor != 0, dim=1)
    filtered_tensor = reshaped_tensor[nonzero_rows]
    distances = torch.cdist(filtered_tensor, filtered_tensor)
    distances.fill_diagonal_(float('inf'))
    closest_neighbor_distances, _ = torch.min(distances, dim=1)
    smallest = torch.min(closest_neighbor_distances)
    return smallest.item()
    

def reset_padding(tensor, padding_mask):
    # tensor: the tensor to be adjusted
    # padding_mask: a Boolean tensor of the same shape as tensor, where True indicates padding that should be reset to zero
    with torch.no_grad():
        tensor[padding_mask] = 0
        
def optimize_ants(model, ants_input, optimizer_class, iters, lr=0.001):
    ants = ants_input.clone().detach().requires_grad_(True)  # Ensure ants is a clone and tracks gradients
    padding_mask = (ants == 0)
    opt = optimizer_class([ants], lr=lr)  # Initialize the optimizer correctly
    
    for it in range(iters):
        last = ants.clone()
        opt.zero_grad()
        model_output = model(ants)
        model_output.backward()  # Make sure the loss function is called correctly
        opt.step()  # Call step on the correct optimizer instance
        reset_padding(ants, padding_mask)  # Apply padding reset correctly
        if calculate_min_distance(ants) < 0.5:
            print(f'Early Stop at {it}')
            return last
    return ants  # Return the optimized tensor

def optimize_ants_pen(model, ants_input, optimizer_class, iters, lr=0.001, loss_fn = lambda x,y: x):
    ants = ants_input.clone().detach().requires_grad_(True)  # Ensure ants is a clone and tracks gradients
    padding_mask = (ants == 0)
    opt = optimizer_class([ants], lr=lr)  # Initialize the optimizer correctly
    
    for it in range(iters):
        last = ants.clone()
        opt.zero_grad()
        model_output = model(ants)
        loss = loss_fn(model_output, ants) # default only returns the model_output
        loss.backward()
        # model_output.backward()  # Make sure the loss function is called correctly
        opt.step()  # Call step on the correct optimizer instance
        reset_padding(ants, padding_mask)  # Apply padding reset correctly
        if calculate_min_distance(ants) < 0.5:
            print(f'Early Stop at {it}')
            return last
    return ants  # Return the optimized tensor

def yz_split(tensor):
    """
    Splits a 1D tensor into two parts, yant and zant, each taking half of the tensor.
    Then removes all trailing zeros from both parts.
    
    Args:
    - tensor (torch.Tensor): A 1D tensor with 2048 elements.
    
    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: The two parts of the tensor with trailing zeros removed.
    """    
    # Split the tensor into yant and zant
    yant, zant = tensor.split(1024)
    
    # Function to remove trailing zeros from a tensor
    def remove_trailing_zeros(tensor):
        # Find the indices of non-zero elements and get the maximum index
        non_zero_indices = tensor.nonzero(as_tuple=True)[0]
        if len(non_zero_indices) == 0:  # If tensor is all zeros
            return tensor[:0]  # Return an empty tensor
        else:
            max_index = non_zero_indices.max()
            return tensor[:max_index + 1]
    
    # Remove trailing zeros from yant and zant
    yant = remove_trailing_zeros(yant)
    zant = remove_trailing_zeros(zant)
    
    return yant, zant