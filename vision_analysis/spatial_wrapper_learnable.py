import torch
import torch.nn as nn
import os
import math
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

def cartesian_to_polar(
    x: torch.Tensor,
    y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        x: 1D (or any-shape) tensor of x-coordinates
        y: Tensor of the same shape as x, y-coordinates

    Returns:
        r:     Tensor of radii, same shape as x
        theta: Tensor of angles in radians, same shape as x (range: -π to π)
    """
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta


def polar_to_cartesian(
    r: torch.Tensor,
    theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert polar coordinates to Cartesian coordinates.

    Args:
        r:     Tensor of radii
        theta: Tensor of angles in radians

    Returns:
        x: Tensor of x-coordinates
        y: Tensor of y-coordinates
    """
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y

def optimize_coordinates(W, x_in, y_in, x_out, y_out, max_iter=100, tol=1e-6, verbose=False,polar=False):
    """
    Finds permutations of input and output coordinates (x_in, y_in, x_out, y_out)
    to minimize mean(W * cdist(input_coords, output_coords)).

    Returns the permuted coordinate tensors.
    """
    # Ensure CPU numpy arrays for optimization
    W = W.detach().cpu()
    x_in_np = x_in.detach().cpu()
    y_in_np = y_in.detach().cpu()
    x_out_np = x_out.detach().cpu()
    y_out_np = y_out.detach().cpu()

    C = compute_distance_matrix_cdist(x_in_np, y_in_np, x_out_np, y_out_np,polar=polar)


    m, n = W.shape
    row_perm = np.arange(m)
    col_perm = np.arange(n)

    def compute_objective(row_perm, col_perm):
        C_perm = C[np.ix_(row_perm, col_perm)]
        return torch.mean(W * C_perm)

    obj_prev = compute_objective(row_perm, col_perm)
    if verbose:
        print("Initial objective:", obj_prev)

    for iteration in range(max_iter):
        # Step 1: Optimize row permutation
        cost_rows = np.dot(W, C[:, col_perm].T)
        _, new_row_perm = linear_sum_assignment(cost_rows)
        row_perm = new_row_perm

        # Step 2: Optimize column permutation
        cost_cols = np.dot(W.T, C[row_perm, :])
        _, new_col_perm = linear_sum_assignment(cost_cols)
        col_perm = new_col_perm

        # Check convergence
        obj_current = compute_objective(row_perm, col_perm)
        if verbose:
            print(f"Iteration {iteration+1}: objective = {obj_current:.6f}", flush=True)
        if abs(obj_prev - obj_current) < tol:
            break
        obj_prev = obj_current

    # Permute coordinates accordingly
    x_in_perm = x_in_np[col_perm]
    y_in_perm = y_in_np[col_perm]
    x_out_perm = x_out_np[row_perm]
    y_out_perm = y_out_np[row_perm]

    return x_in_perm, y_in_perm, x_out_perm, y_out_perm


def l1_non_linear_weights(model, l1_lambda=1):
    l1_loss = 0.0
    total_elements = 0

    # Step 1: collect all linear weight parameter *identities*
    linear_weight_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                linear_weight_ids.add(id(module.weight))

    # Step 2: apply L1 to all other learnable params
    for param in model.parameters():
        if param.requires_grad and id(param) not in linear_weight_ids:
            l1_loss += param.abs().sum()
            total_elements += param.numel()

    if total_elements == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    return l1_lambda * (l1_loss / total_elements)

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
# if we wanna use euclidean space constraints, we can pass in the prior layer locations
def compute_distance_matrix(N, M, A, B, D,cache_dir="cache",polar=False,prev=None):
    x_in = torch.linspace(-A / 2, A / 2, N)
    y_in = torch.full((N,), -D / 2)
    x_out = torch.linspace(-B / 2, B / 2, M)
    y_out = torch.full((M,), D / 2)

    # if polar, convert to polar
    if polar:
        x_in,y_in=cartesian_to_polar(x_in,y_in)
        x_out,y_out=cartesian_to_polar(x_out,y_out)
    # Convert to learnable parameters
    x_in = nn.Parameter(x_in)
    y_in = nn.Parameter(y_in)
    x_out = nn.Parameter(x_out)
    y_out = nn.Parameter(y_out)

    if prev is not None:
        return nn.ParameterList( [prev[0],prev[1],x_out,y_out])
    return nn.ParameterList( [x_in,y_in,x_out,y_out])

def compute_distance_matrix_cdist(o_X, o_Y, i_X, i_Y,polar=False):
    """
    Uses torch.cdist to compute the pairwise Euclidean distance matrix.
    """
    # if polar, first convert to cartesian
    if polar:
        o_X,o_Y = polar_to_cartesian(o_X,o_Y)
        i_X,i_Y = polar_to_cartesian(i_X,i_Y)
    inputs = torch.stack((i_X, i_Y), dim=1)
    outputs = torch.stack((o_X, o_Y), dim=1)
    return torch.cdist(inputs, outputs)

class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1,device="cuda",use_polar=False,euclidean=False):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.value_networks = []
        self.conv_layers = [] 

        self.value_distance_matrices = nn.ModuleList( [])
        self.linear_distance_matrices = nn.ModuleList( [])
        self.conv_distance_matrices = nn.ModuleList([])

        self.A = A
        self.B = B
        self.D = D
        self.spatial_cost_scale = spatial_cost_scale  # Scaling factor for spatial cost
        self.device=device
        self.use_polar=use_polar
        self.euclidean=euclidean
        self._extract_layers(model)

    def _extract_layers(self, module):
        prev=None
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                N = layer.in_features
                M = layer.out_features
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D,polar=self.use_polar,prev=prev)
                self.linear_distance_matrices.append(distance_matrix)
                if self.euclidean:
                    prev = (distance_matrix[2],distance_matrix[3])
            elif isinstance(layer, nn.Conv2d):
                self.conv_layers.append(layer)
                N = layer.in_channels
                M = layer.out_channels
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D,polar=self.use_polar,prev=prev)
                self.conv_distance_matrices.append(distance_matrix)
                if self.euclidean:
                    prev = (distance_matrix[2],distance_matrix[3])
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
            dist_matrix = compute_distance_matrix_cdist(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3],polar=self.use_polar)
            weight_abs = torch.abs(layer.weight)
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device)*dist_matrix.to(self.device))
            else:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        # Cost for convolutional layers.
        for layer, dist_coords in zip(self.conv_layers, self.conv_distance_matrices):
            collision_cost += collision_penalty(dist_coords[0], dist_coords[1],
                                                 dist_coords[2], dist_coords[3],
                                                 collision_threshold)
            dist_matrix = compute_distance_matrix_cdist(dist_coords[0], dist_coords[1],
                                                        dist_coords[2], dist_coords[3],polar=self.use_polar)
            # Average over the spatial kernel dimensions.
            weight_abs = torch.mean(torch.abs(layer.weight), dim=(2, 3))
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device) *
                                          dist_matrix.to(self.device))
            else:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        # Compute cost for value projection layers
        for value_proj, dist_coords in zip(self.value_networks, self.value_distance_matrices):
            collision_cost+=collision_penalty(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3], collision_threshold)

            dist_matrix = compute_distance_matrix_cdist(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3],polar=self.use_polar)
            weight_abs = torch.abs(value_proj[0])
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix*dist_matrix)
            else:
                total_cost += torch.sum(weight_abs * dist_matrix)
            total_params += weight_abs.numel()

        # Apply the scaling factor to the spatial cost
        #print(self.spatial_cost_scale * total_cost / total_params,collision_cost/total_params)
        return self.spatial_cost_scale * total_cost / total_params + self.spatial_cost_scale * collision_cost / total_params + l1_non_linear_weights(self.model)

    def get_stats(self):
        return compute_tensor_stats(self.linear_distance_matrices+self.value_distance_matrices )
    
    def forward(self, *args):
        return self.model(*args)

    def optimize(self):
        print("init",self.get_cost(),flush=True)
        # Compute cost for linear layers
        total=len(self.linear_distance_matrices)+len(self.value_distance_matrices),
        for layer, dist_coords in zip(self.linear_layers, self.linear_distance_matrices):
            weight_abs = torch.abs(layer.weight)
            x_in_perm, y_in_perm, x_out_perm, y_out_perm = optimize_coordinates(weight_abs,dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3],polar=self.use_polar)
            with torch.no_grad():
                dist_coords[0].copy_(x_in_perm)
                dist_coords[1].copy_(y_in_perm)
                dist_coords[2].copy_(x_out_perm)
                dist_coords[3].copy_(y_out_perm)
            
        # Compute cost for value projection layers
        for value_proj, dist_coords in zip(self.value_networks, self.value_distance_matrices):
            weight_abs = torch.abs(value_proj[0])
            x_in_perm, y_in_perm, x_out_perm, y_out_perm = optimize_coordinates(weight_abs,dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3],polar=self.use_polar)
        
            with torch.no_grad():
                dist_coords[0].copy_(x_in_perm)
                dist_coords[1].copy_(y_in_perm)
                dist_coords[2].copy_(x_out_perm)
                dist_coords[3].copy_(y_out_perm)

        # Optimize convolutional layers:
        new_conv_dist_matrices = []

        for layer, dist_coords in zip(self.conv_layers, self.conv_distance_matrices):
            weight_abs = torch.mean(torch.abs(layer.weight), dim=(2,3))
            x_in_perm, y_in_perm, x_out_perm, y_out_perm = optimize_coordinates(weight_abs,dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3],polar=self.use_polar)

            with torch.no_grad():
                dist_coords[0].copy_(x_in_perm)
                dist_coords[1].copy_(y_in_perm)
                dist_coords[2].copy_(x_out_perm)
                dist_coords[3].copy_(y_out_perm)

        print("final",self.get_cost(),flush=True)
