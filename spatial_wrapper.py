import torch
import torch.nn as nn


# Function to compute distance matrix for a given linear layer
def compute_distance_matrix(N, M, A, B, D):
    x_in = torch.linspace(-A / 2, A / 2, N)
    y_in = torch.full((N,), -D / 2)
    x_out = torch.linspace(-B / 2, B / 2, M)
    y_out = torch.full((M,), D / 2)
    
    x_in = x_in.view(N, 1)
    y_in = y_in.view(N, 1)
    x_out = x_out.view(1, M)
    y_out = y_out.view(1, M)
    
    distance_matrix = torch.sqrt((x_out - x_in)**2 + (y_out - y_in)**2)
    return distance_matrix.T

class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1e-4, device="cuda"):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.value_networks = []
        self.value_distance_matrices = []
        self.linear_distance_matrices = []
        self.A = A
        self.B = B
        self.D = D
        self.spatial_cost_scale = spatial_cost_scale
        self.device = device
        
        self._extract_layers(model)
        self.linear_distance_matrices = [m.to(device) for m in self.linear_distance_matrices]
        self.value_distance_matrices = [m.to(device) for m in self.value_distance_matrices]

    def _extract_layers(self, module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                N, M = layer.in_features, layer.out_features
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D)
                self.linear_distance_matrices.append(distance_matrix)
            elif isinstance(layer, nn.MultiheadAttention):
                in_proj_weight = layer.in_proj_weight
                qkv_size = in_proj_weight.size(0) // 3
                value_proj_weight = in_proj_weight[2 * qkv_size:, :]
                value_proj_bias = layer.in_proj_bias[2 * qkv_size:] if layer.in_proj_bias is not None else None
                self.value_networks.append((value_proj_weight, value_proj_bias))
                N, M = value_proj_weight.size(1), value_proj_weight.size(0)
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D)
                self.value_distance_matrices.append(distance_matrix)
            else:
                self._extract_layers(layer)

    def get_cost(self):
        total_cost = 0.0
        total_params = 0
        
        costs = []
        param_counts = []
        
        # Compute cost for linear layers
        for layer, dist_matrix in zip(self.linear_layers, self.linear_distance_matrices):
            weight_abs = torch.abs(layer.weight)
            costs.append(torch.sum(weight_abs * dist_matrix))
            param_counts.append(weight_abs.numel())
        
        # Compute cost for value projection layers
        for (value_proj_weight, _), dist_matrix in zip(self.value_networks, self.value_distance_matrices):
            weight_abs = torch.abs(value_proj_weight)
            costs.append(torch.sum(weight_abs * dist_matrix))
            param_counts.append(weight_abs.numel())
        
        if costs:  # Check if we have any costs
            total_cost = torch.stack(costs).sum()
            total_params = sum(param_counts)
            
        # Apply the scaling factor to the spatial cost
        return self.spatial_cost_scale * total_cost / total_params if total_params > 0 else torch.tensor(0.0).to(self.device)

    def forward(self, x):
        return self.model(x)
    