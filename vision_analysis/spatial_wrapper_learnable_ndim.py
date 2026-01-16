import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

def optimize_coordinates(W, coords_in, coords_out, max_iter=100, tol=1e-6, verbose=False):
    """
    Finds permutations of input and output coordinates to minimize mean(W * distance).
    """
    W_np = W.detach().cpu().numpy()
    coords_in_np = coords_in.detach().cpu().numpy()
    coords_out_np = coords_out.detach().cpu().numpy()
    # Compute distance matrix C of shape (m, n)
    C = np.linalg.norm(coords_out_np[:, None, :] - coords_in_np[None, :, :], axis=2)
    m, n = W_np.shape
    row_perm = np.arange(m)
    col_perm = np.arange(n)

    def objective(rp, cp):
        return np.mean(W_np * C[rp][:, cp])

    obj_prev = objective(row_perm, col_perm)
    if verbose:
        print('Initial objective:', obj_prev)

    for i in range(max_iter):
        # Optimize row permutation
        cost_rows = W_np @ C[:, col_perm].T
        _, row_perm = linear_sum_assignment(cost_rows)
        # Optimize column permutation
        cost_cols = W_np.T @ C[row_perm, :]
        _, col_perm = linear_sum_assignment(cost_cols)
        obj_current = objective(row_perm, col_perm)
        if verbose:
            print('Iteration', i+1, 'objective =', obj_current)
        if abs(obj_prev - obj_current) < tol:
            break
        obj_prev = obj_current

    coords_in_perm = coords_in_np[col_perm]
    coords_out_perm = coords_out_np[row_perm]
    return (torch.tensor(coords_in_perm, device=coords_in.device),
            torch.tensor(coords_out_perm, device=coords_out.device))

def collision_penalty(coords_in, coords_out, threshold):
    """
    Computes a repulsive penalty for all neurons combined in N dims.
    """
    positions = torch.cat([coords_in, coords_out], dim=0)
    dists = torch.cdist(positions, positions, p=2)
    mask = torch.eye(dists.size(0), device=dists.device, dtype=torch.bool)
    dists = dists.masked_fill(mask, float('inf'))
    penalty = F.relu(threshold - dists) ** 2
    return penalty.sum() / 2.0

def compute_distance_matrix(N, M, A, B, D, n_dims, prev=None):
    """
    Initializes coords for N input neurons and M output neurons in n_dims.
    Variation along dim 0, separation along dim 1, other dims = 0.
    """
    if n_dims < 2:
        raise ValueError('n_dims must be >= 2')
    coords_in = torch.zeros((N, n_dims))
    coords_out = torch.zeros((M, n_dims))
    # variation along axis 0
    coords_in[:, 0] = torch.linspace(-A/2, A/2, N)
    coords_out[:, 0] = torch.linspace(-B/2, B/2, M)
    # separation along axis 1
    coords_in[:, 1] = -D / 2
    coords_out[:, 1] = D / 2
    coords_in = nn.Parameter(coords_in)
    coords_out = nn.Parameter(coords_out)
    if prev is not None:
        return nn.ParameterList([coords_in, coords_out, prev[0], prev[1]])
    return nn.ParameterList([coords_in, coords_out])

def compute_distance_matrix_cdist(coords_in, coords_out):
    """
    Computes pairwise distance matrix C of shape (m, n).
    """
    return torch.cdist(coords_out, coords_in, p=2)

def l1_non_linear_weights(model, l1_lambda=1):
    """
    L1 on all learnable params except linear-layer weights.
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total = 0
    linear_ids = {
        id(p) for m in model.modules() if isinstance(m, nn.Linear)
        for p in ([m.weight] if m.weight.requires_grad else [])
    }
    for p in model.parameters():
        if p.requires_grad and id(p) not in linear_ids:
            l1_loss += p.abs().sum()
            total += p.numel()
    return l1_loss * (l1_lambda / total) if total > 0 else l1_loss

def compute_tensor_stats(list_of_lists):
    """
    Computes (min, max, mean, std) over lists of tensor lists.
    """
    flat = [t.view(-1) for sub in list_of_lists for t in sub]
    all_vals = torch.cat(flat)
    return (all_vals.min().item(), all_vals.max().item(),
            all_vals.mean().item(), all_vals.std().item())

class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1.0, n_dims=2,device='cuda'):
        super().__init__()
        self.model = model.to(device)
        self.n_dims = n_dims
        self.A, self.B, self.D = A, B, D
        self.spatial_cost_scale = spatial_cost_scale
        self.device = device

        self.linear_layers = []
        self.linear_distance_matrices = nn.ModuleList()
        self.conv_layers = []
        self.conv_distance_matrices = nn.ModuleList()
        self.value_distance_matrices = nn.ModuleList()

        self._extract_layers(self.model)

    def _extract_layers(self, module, prev=None, prefix=''):
        for name, layer in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            # Skip attention layers
            if 'attn' in full_name:
                self._extract_layers(layer, prev, full_name)
                continue
            # Skip final classification layer (head/fc/classifier)
            if name in ('head', 'fc', 'classifier'):
                continue

            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                N, M = layer.in_features, layer.out_features
                dist = compute_distance_matrix(N, M, self.A, self.B, self.D, self.n_dims, prev)
                self.linear_distance_matrices.append(dist)
                prev = (dist[0], dist[1])
            elif isinstance(layer, nn.Conv2d):
                self.conv_layers.append(layer)
                N, M = layer.in_channels, layer.out_channels
                dist = compute_distance_matrix(N, M, self.A, self.B, self.D, self.n_dims, prev)
                self.conv_distance_matrices.append(dist)
                prev = (dist[0], dist[1])
            else:
                self._extract_layers(layer, prev, full_name)

    def get_cost(self, quadratic=False):
        total_cost, total_params, collision_cost = 0.0, 0, 0.0
        # linear layers
        for layer, dist in zip(self.linear_layers, self.linear_distance_matrices):
            ci, co = dist[0].to(self.device), dist[1].to(self.device)
            collision_cost += collision_penalty(ci, co, self.D)
            C = compute_distance_matrix_cdist(ci, co)
            W = layer.weight.abs()
            total_cost += ((W * C * C) if quadratic else (W * C)).sum()
            total_params += W.numel()
        # conv layers
        for layer, dist in zip(self.conv_layers, self.conv_distance_matrices):
            ci, co = dist[0].to(self.device), dist[1].to(self.device)
            collision_cost += collision_penalty(ci, co, self.D)
            C = compute_distance_matrix_cdist(ci, co)
            W = layer.weight.abs().mean(dim=(2,3))
            total_cost += ((W * C * C) if quadratic else (W * C)).sum()
            total_params += W.numel()
        return (self.spatial_cost_scale * total_cost / total_params
                + self.spatial_cost_scale * collision_cost / total_params
                + l1_non_linear_weights(self.model))

    def optimize(self, max_iter=100, tol=1e-6, verbose=False):
        print('Initial cost', self.get_cost())
        # linear
        for layer, dist in zip(self.linear_layers, self.linear_distance_matrices):
            ci, co = dist[0], dist[1]
            W = layer.weight.abs()
            new_ci, new_co = optimize_coordinates(W, ci, co, max_iter, tol, verbose)
            with torch.no_grad():
                ci.copy_(new_ci)
                co.copy_(new_co)
        # conv
        for layer, dist in zip(self.conv_layers, self.conv_distance_matrices):
            ci, co = dist[0], dist[1]
            W = layer.weight.abs().mean(dim=(2,3))
            new_ci, new_co = optimize_coordinates(W, ci, co, max_iter, tol, verbose)
            with torch.no_grad():
                ci.copy_(new_ci)
                co.copy_(new_co)
        print('Final cost', self.get_cost())

    def get_stats(self):
        return compute_tensor_stats(list(self.linear_distance_matrices)
                                    + list(self.conv_distance_matrices))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
