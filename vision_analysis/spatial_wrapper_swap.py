
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import os
import math

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

def create_clustered_connectivity(i, o, k):
    """
    Create a connectivity matrix with shape (o, i) based on k contiguous clusters.
    
    Parameters:
      i (int): number of input neurons.
      o (int): number of output neurons.
      k (int): number of clusters.
      
    Returns:
      np.ndarray: A binary connectivity matrix of shape (o, i) where each cluster of input neurons 
                  is fully connected (all ones) to the corresponding cluster of output neurons.
                  
    The neurons are divided as evenly as possible into k contiguous clusters.
    For example, if i=10, o=20 and k=5, the inputs are divided into groups of 2 neurons
    and the outputs into groups of 4 neurons. Each group of 2 inputs connects to one group of 4 outputs.
    """
    # Create an o x i matrix initialized with zeros.
    matrix = np.ones((o, i), dtype=int)
    
    # Determine cluster sizes for the inputs.
    base_in = i // k
    remainder_in = i % k
    input_cluster_sizes = [base_in + 1 if idx < remainder_in else base_in for idx in range(k)]
    
    # Determine cluster sizes for the outputs.
    base_out = o // k
    remainder_out = o % k
    output_cluster_sizes = [base_out + 1 if idx < remainder_out else base_out for idx in range(k)]
    
    # Define start indices for input and output clusters.
    in_start = 0
    out_start = 0
    
    # Fill the connectivity matrix block-by-block.
    for cluster in range(k):
        in_end = in_start + input_cluster_sizes[cluster]
        out_end = out_start + output_cluster_sizes[cluster]
        # Connect the entire block corresponding to the current cluster.
        matrix[out_start:out_end, in_start:in_end] = 0
        
        # Update indices for the next cluster.
        in_start = in_end
        out_start = out_end
        
    return torch.tensor(matrix)

def alternative_hungarian_optimization(W, C, max_iter=100, tol=1e-6, verbose=False):
    """
    Finds row and column permutations for matrix C to minimize mean(W * C_perm),
    where * is the elementwise product. Returns the new permuted C matrix.
    
    Parameters:
      W: 2D numpy array of shape (m, n) with nonnegative entries.
      C: 2D numpy array of shape (m, n) with nonnegative entries.
      max_iter: maximum number of alternating iterations.
      tol: tolerance for convergence (based on change in objective).
      verbose: if True, prints progress.
      
    Returns:
      row_perm: permutation of row indices for C (as a numpy array).
      col_perm: permutation of column indices for C (as a numpy array).
      obj: final objective value.
      C_new: the permuted version of C, i.e., C[np.ix_(row_perm, col_perm)].
    """
    W=W.detach().cpu().numpy()
    C=C.detach().cpu().numpy() 
    m, n = W.shape
    
    # Initialize with the identity permutation.
    row_perm = np.arange(m)
    col_perm = np.arange(n)
    
    def compute_objective(row_perm, col_perm):
        # Compute mean of elementwise product after applying the permutations.
        C_perm = C[np.ix_(row_perm, col_perm)]
        return np.mean(W * C_perm)
    
    obj_prev = compute_objective(row_perm, col_perm)
    if verbose:
        print("Initial objective: ", obj_prev)
    
    for iteration in range(max_iter):
        # --- Step 1: Optimize row permutation (with fixed col_perm) ---
        # Vectorized computation of cost for rows:
        # cost_rows[i, k] = dot(W[i, :], C[k, col_perm])
        cost_rows = np.dot(W, C[:, col_perm].T)
        _, new_row_perm = linear_sum_assignment(cost_rows)
        row_perm = new_row_perm  # new_row_perm gives the row from C assigned to row i in W
        
        # --- Step 2: Optimize column permutation (with fixed row_perm) ---
        # Vectorized computation of cost for columns:
        # cost_cols[j, l] = dot(W[:, j], C[row_perm, l])
        cost_cols = np.dot(W.T, C[row_perm, :])
        _, new_col_perm = linear_sum_assignment(cost_cols)
        col_perm = new_col_perm
        
        # --- Check convergence ---
        obj_current = compute_objective(row_perm, col_perm)
        if verbose:
            print(f"Iteration {iteration+1}: objective = {obj_current}",flush=True)
        
        if abs(obj_prev - obj_current) < tol:
            break
        obj_prev = obj_current
        
    # Create the new permuted version of C.
    C_new = C[np.ix_(row_perm, col_perm)]
    return torch.tensor(C_new)



def compute_distance_matrix_circle(N, M, A, B, D, cache_dir="cache"):
    """
    Computes an MxN matrix where each element (i, j) is the Euclidean distance 
    between the i-th outgoing neuron and j-th incoming neuron.
    
    Incoming neurons are arranged uniformly along a circle of radius A in the xy-plane (z=0).
    Outgoing neurons are arranged uniformly along a circle of radius B in a parallel plane (z=D).
    
    Args:
        N (int): Number of incoming neurons.
        M (int): Number of outgoing neurons.
        A (float): Radius of the circle for incoming neurons.
        B (float): Radius of the circle for outgoing neurons.
        D (float): Distance between the two parallel planes.
        cache_dir (str): Directory to cache the computed distance matrix.
    
    Returns:
        torch.Tensor: An MxN distance matrix.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"circle_dist_N{N}_M{M}_A{A:.6f}_B{B:.6f}_D{D:.6f}.pt"
    cache_path = os.path.join(cache_dir, fname)
    if os.path.isfile(cache_path):
        return torch.load(cache_path)
    
    # Compute angles for uniform distribution along each circle.
    angles_in = torch.arange(N) * (2 * math.pi / N)
    angles_out = torch.arange(M) * (2 * math.pi / M)
    
    # Incoming neuron coordinates (circle of radius A at z=0).
    x_in = A * torch.cos(angles_in)
    y_in = A * torch.sin(angles_in)
    z_in = torch.zeros(N)
    
    # Outgoing neuron coordinates (circle of radius B at z=D).
    x_out = B * torch.cos(angles_out)
    y_out = B * torch.sin(angles_out)
    z_out = torch.full((M,), D)
    
    # Compute differences in the x and y dimensions using broadcasting.
    dx = x_out[:, None] - x_in[None, :]
    dy = y_out[:, None] - y_in[None, :]
    # The difference in z is D for every pair (since z_in is 0 and z_out is D).
    dz = D
    
    # Compute the Euclidean distance matrix.
    distance_matrix = torch.sqrt(dx**2 + dy**2 + dz**2)
    
    torch.save(distance_matrix, cache_path)
    return distance_matrix

# Function to compute distance matrix for a given linear layer
def compute_distance_matrix(N, M, A, B, D,cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"dist_N{N}_M{M}_A{A:.6f}_B{B:.6f}_D{D:.6f}.pt"
    cache_path = os.path.join(cache_dir, fname)
    if os.path.isfile(cache_path):
        return torch.load(cache_path)
    
    x_in = torch.linspace(-A / 2, A / 2, N)
    y_in = torch.full((N,), -D / 2)
    x_out = torch.linspace(-B / 2, B / 2, M)
    y_out = torch.full((M,), D / 2)

    distance_matrix = torch.empty(M, N)
    for j in range(M):
        for i in range(N):
            distance_matrix[j, i] = torch.sqrt((x_out[j] - x_in[i])**2 + (y_out[j] - y_in[i])**2)
    torch.save(distance_matrix, cache_path)
    return distance_matrix

class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1,device="cuda",circle=False,cluster=-1,distribution="spatial"):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.value_networks = []
        self.value_distance_matrices = []
        self.linear_distance_matrices = []
        self.conv_layers = []
        self.conv_distance_matrices = []
        self.A = A
        self.B = B
        self.D = D
        self.circle = circle
        self.cluster=cluster
        self.spatial_cost_scale = spatial_cost_scale  # Scaling factor for spatial cost
        self.device=device
        self.distribution=distribution
        if self.distribution not in ['spatial','cluster','uniform','gaussian']:
            raise ValueError("Illegal Distribuion!")
        self._extract_layers(model)

    def _extract_layers(self, module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                N = layer.in_features
                M = layer.out_features
                if self.cluster > 0:
                    distance_matrix = create_clustered_connectivity(N,M,self.cluster)
                elif self.circle:
                    distance_matrix = compute_distance_matrix_circle(N, M, self.A, self.B, self.D)
                elif self.distribution == 'uniform':
                    distance_matrix = torch.rand(M, N)
                elif self.distribution == 'gaussian':
                    distance_matrix = torch.randn(M, N).abs()
                else:
                    distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D)
                self.linear_distance_matrices.append(distance_matrix)
            elif isinstance(layer, nn.Conv2d):
                    # For conv layers, we may treat channel connectivity similar to linear layers.
                    # Note: Conv2d weights are of shape (out_channels, in_channels, kernel_h, kernel_w).
                    # One option is to average over the spatial kernel dimensions.
                    self.conv_layers.append(layer)
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels
                    if self.cluster > 0:
                        distance_matrix = create_clustered_connectivity(in_channels, out_channels, self.cluster)
                    elif self.circle:
                        distance_matrix = compute_distance_matrix_circle(in_channels, out_channels, self.A, self.B, self.D)
                    else:
                        distance_matrix = compute_distance_matrix(in_channels, out_channels, self.A, self.B, self.D)
                    self.conv_distance_matrices.append(distance_matrix) 
            else:
                self._extract_layers( layer)

    def get_cost(self,quadratic=False):
        total_cost = 0.0
        total_params = 0

        # Compute cost for linear layers
        for layer, dist_matrix in zip(self.linear_layers, self.linear_distance_matrices):
            weight_abs = torch.abs(layer.weight)
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device)*dist_matrix.to(self.device))
            else:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        # Compute cost for value projection layers
        for value_proj, dist_matrix in zip(self.value_networks, self.value_distance_matrices):
            weight_abs = torch.abs(value_proj[0])
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix*dist_matrix)
            else:
                total_cost += torch.sum(weight_abs * dist_matrix)
            total_params += weight_abs.numel()

        # Cost for convolutional layers:
        for layer, dist_matrix in zip(self.conv_layers, self.conv_distance_matrices):
            # Average absolute weights over spatial dimensions:
            weight_abs = torch.mean(torch.abs(layer.weight), dim=(2,3))
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device) * dist_matrix.to(self.device))
            else:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        # Apply the scaling factor to the spatial cost
        return self.spatial_cost_scale * total_cost / total_params + l1_non_linear_weights(self)

    def optimize(self):
        print("init",self.get_cost(),flush=True)
        # Compute cost for linear layers
        new_dist_matrices= []
        i=0
        total=len(self.linear_distance_matrices)+len(self.value_distance_matrices),
        for layer, dist_matrix in zip(self.linear_layers, self.linear_distance_matrices):
            i+=1
            #print(i,total,dist_matrix.shape,flush=True)
            weight_abs = torch.abs(layer.weight)
            new_dist_matrices.append(alternative_hungarian_optimization(weight_abs, dist_matrix))
        self.linear_distance_matrices=new_dist_matrices
        new_dist_matrices= []
        # Compute cost for value projection layers
        for value_proj, dist_matrix in zip(self.value_networks, self.value_distance_matrices):
            i+=1
            #print(i,total,dist_matrix.shape,flush=True)

            weight_abs = torch.abs(value_proj[0])
            new_dist_matrices.append(alternative_hungarian_optimization(weight_abs, dist_matrix))
        self.value_distance_matrices=new_dist_matrices

        # Optimize convolutional layers:
        new_conv_dist_matrices = []

        for layer, dist_matrix in zip(self.conv_layers, self.conv_distance_matrices):
            # Use the channel-wise averaged weights as before:
            i+=1
            #print(i,total,dist_matrix.shape,flush=True)
            weight_abs = torch.mean(torch.abs(layer.weight), dim=(2,3))
            new_conv_dist_matrices.append(alternative_hungarian_optimization(weight_abs, dist_matrix))
        self.conv_distance_matrices = new_conv_dist_matrices

        print("final",self.get_cost(),flush=True)


    def forward(self, *args):
        return self.model(*args)
