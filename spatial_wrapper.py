import torch
import torch.nn as nn


# Function to compute distance matrix for a given linear layer
def compute_distance_matrix(N, M, A, B, D):
    x_in = torch.linspace(-A / 2, A / 2, N)
    y_in = torch.full((N,), -D / 2)
    x_out = torch.linspace(-B / 2, B / 2, M)
    y_out = torch.full((M,), D / 2)

    distance_matrix = torch.empty(M, N)
    for j in range(M):
        for i in range(N):
            distance_matrix[j, i] = torch.sqrt((x_out[j] - x_in[i])**2 + (y_out[j] - y_in[i])**2)

    return distance_matrix

class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1e-4,device="cuda"):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.value_networks = []
        self.value_distance_matrices = []
        self.linear_distance_matrices = []
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

    def get_cost(self):
        total_cost = 0.0
        total_params = 0

        # Compute cost for linear layers
        for layer, dist_matrix in zip(self.linear_layers, self.linear_distance_matrices):
            weight_abs = torch.abs(layer.weight)
            total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        # Compute cost for value projection layers
        for value_proj, dist_matrix in zip(self.value_networks, self.value_distance_matrices):
            weight_abs = torch.abs(value_proj[0])
            total_cost += torch.sum(weight_abs * dist_matrix)
            total_params += weight_abs.numel()

        # Apply the scaling factor to the spatial cost
        return self.spatial_cost_scale * total_cost / total_params

    def forward(self, x):
        return self.model(x)
    