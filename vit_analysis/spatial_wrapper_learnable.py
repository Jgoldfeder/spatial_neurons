import torch
import torch.nn as nn

import os
import torch
import math

import torch.nn.functional as F

import torch

def compute_tensor_stats(list_of_lists):
    """
    Computes min, max, mean, and standard deviation for a list of lists of tensors.
    
    Args:
        list_of_lists (List[List[torch.Tensor]]): A list of lists of tensors.
        
    Returns:
        tuple: (min_value, max_value, mean_value, std_value)
    """
    # Flatten each tensor in each sublist to a 1D tensor.
    flattened_tensors = [tensor.view(-1) for sublist in list_of_lists for tensor in sublist]
    
    # Concatenate all flattened tensors into one long tensor.
    all_values = torch.cat(flattened_tensors, dim=0)
    
    # Compute the statistics.
    min_val = all_values.min().item()
    max_val = all_values.max().item()
    mean_val = all_values.mean().item()
    std_val = all_values.std().item()
    
    return min_val, max_val, mean_val, std_val




def collision_penalty(x_in, y_in, x_out, y_out, threshold):
    """
    Computes a repulsive (collision) penalty for all neurons combined
    (both input and output) using torch.cdist for efficiency.

    Args:
        x_in, y_in (torch.Tensor): 1D tensors for input neuron x and y positions.
        x_out, y_out (torch.Tensor): 1D tensors for output neuron x and y positions.
        threshold (float): The minimum allowed distance between neurons.
        lambda_factor (float): Scaling factor for the penalty.

    Returns:
        torch.Tensor: A scalar penalty value.
    """
    # Concatenate input and output positions to treat them as one group.
    x_all = torch.cat((x_in, x_out), dim=0)
    y_all = torch.cat((y_in, y_out), dim=0)
    
    # Stack into a tensor of shape [N_total, 2]
    positions = torch.stack((x_all, y_all), dim=1)
    N_total = positions.size(0)
    
    # Compute the pairwise Euclidean distance matrix using torch.cdist
    dists = torch.cdist(positions, positions, p=2)
    
    # Set the diagonal to infinity to avoid self-collision penalty.
    mask = torch.eye(N_total, device=positions.device, dtype=torch.bool)
    dists = dists.masked_fill(mask, float('inf'))

    
    # Compute the repulsive penalty for distances below the threshold.
    penalty = F.relu(threshold - dists) ** 2
    
    # Each pair is counted twice in the symmetric distance matrix, so divide the sum by 2.
    total_penalty = penalty.sum() / 2.0
    return total_penalty


# Function to compute distance matrix for a given linear layer
def compute_distance_matrix(N, M, A, B, D,cache_dir="cache"):
    x_in = torch.linspace(-A / 2, A / 2, N)
    y_in = torch.full((N,), -D / 2)
    x_out = torch.linspace(-B / 2, B / 2, M)
    y_out = torch.full((M,), D / 2)

    # Convert to learnable parameters
    x_in = nn.Parameter(x_in)
    y_in = nn.Parameter(y_in)
    x_out = nn.Parameter(x_out)
    y_out = nn.Parameter(y_out)


    return nn.ParameterList( [x_in,y_in,x_out,y_out])
def compute_distance_matrix_cdist(o_X, o_Y, i_X, i_Y):
    """
    Uses torch.cdist to compute the pairwise Euclidean distance matrix.
    """
    inputs = torch.stack((i_X, i_Y), dim=1)
    outputs = torch.stack((o_X, o_Y), dim=1)
    return torch.cdist(inputs, outputs)

class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1,device="cuda"):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.value_networks = []
        self.value_distance_matrices = nn.ModuleList( [])
        self.linear_distance_matrices = nn.ModuleList( [])
        self.A = A
        self.B = B
        self.D = D
        self.spatial_cost_scale = spatial_cost_scale  # Scaling factor for spatial cost
        self.device=device
        self._extract_layers(model)

    def _extract_layers(self, module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                N = layer.in_features
                M = layer.out_features
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D)
                self.linear_distance_matrices.append(distance_matrix)
            elif isinstance(layer, nn.MultiheadAttention):
                in_proj_weight = layer.in_proj_weight
                in_proj_bias = layer.in_proj_bias
                qkv_size = in_proj_weight.size(0) // 3
                value_proj_weight = in_proj_weight[2 * qkv_size:, :]
                value_proj_bias = in_proj_bias[2 * qkv_size:]
                self.value_networks.append((value_proj_weight, value_proj_bias))

                N = value_proj_weight.size(1)
                M = value_proj_weight.size(0)
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D)
                self.value_distance_matrices.append(distance_matrix)
            else:
                self._extract_layers( layer)

    def get_cost(self,quadratic=False):
        total_cost = 0.0
        total_params = 0

        collision_cost = 0
        collision_threshold = self.D

        # Compute cost for linear layers
        for layer, dist_coords in zip(self.linear_layers, self.linear_distance_matrices):
            collision_cost+=collision_penalty(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3], collision_threshold)
            dist_matrix = compute_distance_matrix_cdist(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3])
            weight_abs = torch.abs(layer.weight)
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device)*dist_matrix.to(self.device))
            else:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        # Compute cost for value projection layers
        for value_proj, dist_coords in zip(self.value_networks, self.value_distance_matrices):
            collision_cost+=collision_penalty(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3], collision_threshold)

            dist_matrix = compute_distance_matrix_cdist(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3])
            weight_abs = torch.abs(value_proj[0])
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix*dist_matrix)
            else:
                total_cost += torch.sum(weight_abs * dist_matrix)
            total_params += weight_abs.numel()

        # Apply the scaling factor to the spatial cost
        #print(self.spatial_cost_scale * total_cost / total_params,collision_cost/total_params)
        return self.spatial_cost_scale * total_cost / total_params + self.spatial_cost_scale * collision_cost / total_params

    def get_stats(self):
        return compute_tensor_stats(self.linear_distance_matrices+self.value_distance_matrices )
    
    def forward(self, *args):
        return self.model(*args)
