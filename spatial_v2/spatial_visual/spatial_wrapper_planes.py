"""
Variant of spatial_wrapper_learnable.py with structural layer separation.

Geometry per layer:
  - input neurons:  learnable (x, y) on the plane z = 0
  - output neurons: learnable (x, y) on the plane z = D (z is fixed, not learnable)
  - wire length     = sqrt(dx^2 + dy^2 + D^2)  -> every connection costs at least D
  - collision penalty relu(D - d)^2 only between neurons on the SAME plane
    (cross-plane pairs are separated by D by construction).

Each plane initializes as an even 2D grid: a line init (all points at the same
y) is a symmetry saddle — every pairwise dy is zero, so the y coordinates
would receive exactly zero gradient and never move.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


def optimize_coordinates(W, x_in, y_in, x_out, y_out, D, max_iter=100, tol=1e-6, verbose=False):
    """
    Finds permutations of input and output coordinates to minimize
    mean(W * dist(input_coords, output_coords)). Returns permuted coordinates.
    """
    W = W.detach().cpu()
    x_in_np = x_in.detach().cpu()
    y_in_np = y_in.detach().cpu()
    x_out_np = x_out.detach().cpu()
    y_out_np = y_out.detach().cpu()

    C = compute_distance_matrix_cdist(x_in_np, y_in_np, x_out_np, y_out_np, D)

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
        cost_rows = np.dot(W, C[:, col_perm].T)
        _, new_row_perm = linear_sum_assignment(cost_rows)
        row_perm = new_row_perm

        cost_cols = np.dot(W.T, C[row_perm, :])
        _, new_col_perm = linear_sum_assignment(cost_cols)
        col_perm = new_col_perm

        obj_current = compute_objective(row_perm, col_perm)
        if verbose:
            print(f"Iteration {iteration+1}: objective = {obj_current:.6f}", flush=True)
        if abs(obj_prev - obj_current) < tol:
            break
        obj_prev = obj_current

    x_in_perm = x_in_np[col_perm]
    y_in_perm = y_in_np[col_perm]
    x_out_perm = x_out_np[row_perm]
    y_out_perm = y_out_np[row_perm]

    return x_in_perm, y_in_perm, x_out_perm, y_out_perm


def compute_tensor_stats(list_of_lists):
    flattened_tensors = [tensor.view(-1) for sublist in list_of_lists for tensor in sublist]
    all_values = torch.cat(flattened_tensors, dim=0)
    return (all_values.min().item(), all_values.max().item(),
            all_values.mean().item(), all_values.std().item())


def collision_penalty(x, y, threshold):
    """
    Repulsive penalty among neurons of a SINGLE plane: pairs closer than
    `threshold` (in-plane 2D distance) are penalized quadratically.
    """
    positions = torch.stack((x, y), dim=1)
    n = positions.size(0)
    dists = torch.cdist(positions, positions, p=2)
    mask = torch.eye(n, device=positions.device, dtype=torch.bool)
    dists = dists.masked_fill(mask, float('inf'))
    penalty = F.relu(threshold - dists) ** 2
    return penalty.sum() / 2.0


def select_diverse_neurons(signatures, k):
    """
    Greedy farthest-point selection of K neurons that are both IMPORTANT (large
    connectivity norm) and MUTUALLY NON-OVERLAPPING (low |W|-cosine similarity).
    `signatures` is (n, feat), the neuron's absolute connectivity vector. Returns
    the chosen neuron indices. These become spread anchors: a diverse scaffold
    that spans the functional space so the layout cannot collapse to a point.
    """
    n = signatures.shape[0]
    k = min(k, n)
    imp = signatures.norm(dim=1) + 1e-9
    s = F.normalize(signatures, dim=1)
    chosen = [int(torch.argmax(imp))]
    maxsim = (s @ s[chosen[-1]]).clamp(-1, 1)
    for _ in range(k - 1):
        score = imp * (1 - maxsim)          # important AND far from everything chosen
        score[torch.tensor(chosen, device=score.device)] = -1e9
        nxt = int(torch.argmax(score))
        chosen.append(nxt)
        maxsim = torch.maximum(maxsim, (s @ s[nxt]).clamp(-1, 1))
    return torch.tensor(chosen, device=signatures.device)


def _grid_init(n, side):
    """
    Evenly spread n points on a 2D grid covering a side x side square, then
    SHUFFLE the assignment of neurons to grid sites (deterministically, seeded
    by n). Without the shuffle, neuron index correlates with position — fatal
    when indices are semantically ordered (e.g. output logits 0-9 = task A,
    10-19 = task B would start spatially pre-separated, faking any
    "the wiring cost sorted them" result).
    """
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    xs = torch.linspace(-side / 2, side / 2, cols)
    ys = torch.linspace(-side / 2, side / 2, rows)
    gx, gy = torch.meshgrid(xs, ys, indexing='xy')
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(n))
    return gx.flatten()[:n][perm].clone(), gy.flatten()[:n][perm].clone()


def compute_distance_matrix(N, M, A, B, prev=None):
    """
    Initial in-plane coordinates: each plane starts as an even 2D grid
    (the layer separation lives entirely in the fixed z gap).
    If `prev` is given (euclidean chaining), the input coordinates are shared
    with the previous layer's outputs.
    """
    gx_in, gy_in = _grid_init(N, A)
    gx_out, gy_out = _grid_init(M, B)
    x_in = nn.Parameter(gx_in)
    y_in = nn.Parameter(gy_in)
    x_out = nn.Parameter(gx_out)
    y_out = nn.Parameter(gy_out)

    if prev is not None:
        return nn.ParameterList([prev[0], prev[1], x_out, y_out])
    return nn.ParameterList([x_in, y_in, x_out, y_out])


def compute_distance_matrix_cdist(x_in, y_in, x_out, y_out, D):
    """
    (M, N) matrix of wire lengths between the two planes:
    sqrt(in-plane distance^2 + D^2). Matches the weight shape (out, in).
    """
    inputs = torch.stack((x_in, y_in), dim=1)     # (N, 2)
    outputs = torch.stack((x_out, y_out), dim=1)  # (M, 2)
    planar = torch.cdist(outputs, inputs)         # (M, N)
    return torch.sqrt(planar ** 2 + D ** 2)


class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1, device="cuda", euclidean=False,
                 collision_radius=None, coupled=False, detach_weights=False,
                 k_anchors=0, anchor_strength=10.0, anchor_radius=None):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.conv_layers = []

        self.linear_distance_matrices = nn.ModuleList([])
        self.conv_distance_matrices = nn.ModuleList([])

        self.A = A
        self.B = B
        self.D = D
        # within-plane repulsion range, decoupled from the plane gap D (defaults to D)
        self.collision_radius = D if collision_radius is None else collision_radius
        self.spatial_cost_scale = spatial_cost_scale
        self.device = device
        # coupled: layer n's input coords ARE layer n-1's output coords (one physical
        # neuron, one position). Collapses the per-layer plane pairs into a single stack
        # of L+1 planes that can be plotted as a whole network. (`euclidean` is the
        # original name for the same mechanism.)
        self.euclidean = euclidean or coupled
        # detach_weights: the wiring cost uses |W|.detach(), so get_cost moves ONLY the
        # positions (a passive embedding of the weight graph) and never pushes the weights.
        # Use when a separate penalty (e.g. L1) is the thing regularizing the weights.
        self.detach_weights = detach_weights
        # anchors: K diverse+important neurons per plane, repelled strongly (radius
        # anchor_radius, weight anchor_strength) ON TOP OF the ordinary collision term,
        # so they spread wide and scaffold the layout against collapse. update_anchors()
        # must be called (after a warmup) to select them; k_anchors=0 disables.
        self.k_anchors = k_anchors
        self.anchor_strength = anchor_strength
        self.anchor_radius = A if anchor_radius is None else anchor_radius
        self.anchor_idx = None
        self._extract_layers(model)

    def _extract_layers(self, module, prefix=''):
        prev = None
        for name, layer in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            # Skip attention layers
            if 'attn' in full_name:
                self._extract_layers(layer, full_name)
                continue
            # Skip final classification layer (head/fc/classifier)
            if name in ('head', 'fc', 'classifier'):
                continue

            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                N = layer.in_features
                M = layer.out_features
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, prev=prev)
                self.linear_distance_matrices.append(distance_matrix)
                if self.euclidean:
                    prev = (distance_matrix[2], distance_matrix[3])
            elif isinstance(layer, nn.Conv2d):
                self.conv_layers.append(layer)
                N = layer.in_channels
                M = layer.out_channels
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, prev=prev)
                self.conv_distance_matrices.append(distance_matrix)
                if self.euclidean:
                    prev = (distance_matrix[2], distance_matrix[3])
            else:
                self._extract_layers(layer, full_name)

    def get_cost(self, quadratic=False):
        total_cost = 0.0
        total_params = 0

        collision_cost = 0
        collision_threshold = self.collision_radius

        all_layers = (list(zip(self.linear_layers, self.linear_distance_matrices))
                      + list(zip(self.conv_layers, self.conv_distance_matrices)))

        for layer, dist_coords in all_layers:
            # collision only within each plane
            collision_cost += collision_penalty(dist_coords[0], dist_coords[1], collision_threshold)
            collision_cost += collision_penalty(dist_coords[2], dist_coords[3], collision_threshold)

            dist_matrix = compute_distance_matrix_cdist(dist_coords[0], dist_coords[1],
                                                        dist_coords[2], dist_coords[3], self.D)
            if isinstance(layer, nn.Conv2d):
                weight_abs = torch.mean(torch.abs(layer.weight), dim=(2, 3))
            else:
                weight_abs = torch.abs(layer.weight)
            if self.detach_weights:
                weight_abs = weight_abs.detach()
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device) * dist_matrix.to(self.device))
            else:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        cost = (self.spatial_cost_scale * total_cost / total_params
                + self.spatial_cost_scale * collision_cost / total_params)

        # strong extra repulsion among the diverse anchor neurons (radius anchor_radius),
        # on top of the ordinary collision above -> they spread wide and scaffold the layout
        if self.k_anchors > 0 and self.anchor_idx is not None:
            anchor_cost = 0.0
            n_terms = 0
            for (x, y), idx in zip(self._plane_coords(), self.anchor_idx):
                if idx.numel() >= 2:
                    idx = idx.to(x.device)
                    pairs = idx.numel() * (idx.numel() - 1) / 2.0
                    anchor_cost = anchor_cost + collision_penalty(x[idx], y[idx], self.anchor_radius) / pairs
                    n_terms += 1
            if n_terms > 0:
                cost = cost + self.spatial_cost_scale * self.anchor_strength * anchor_cost / n_terms
        return cost

    def get_stats(self):
        return compute_tensor_stats(self.linear_distance_matrices)

    def coupled_planes(self):
        """
        For a coupled network: the unique (x, y) planes in z-order, one per layer
        boundary — length len(linear_layers) + 1. Plane i sits at z = i * D and
        holds the neurons between layer i-1 and layer i. Because layer i's input
        coords are the same Parameters as layer i-1's output coords, taking the
        output side of every layer (plus the very first input plane) enumerates
        each physical plane exactly once. Meaningful only when coupled=True.
        """
        dmats = self.linear_distance_matrices
        planes = [(dmats[0][0], dmats[0][1])]
        for dm in dmats:
            planes.append((dm[2], dm[3]))
        return planes

    def _plane_coords(self):
        """List of (x, y) parameter pairs, one per plane. Coupled: L+1 shared planes.
        Decoupled: each layer's in-plane then out-plane."""
        if self.euclidean:
            return self.coupled_planes()
        out = []
        for dm in self.linear_distance_matrices:
            out.append((dm[0], dm[1])); out.append((dm[2], dm[3]))
        return out

    def _anchor_signatures(self):
        """Per-plane absolute connectivity signature (n, feat), aligned with
        _plane_coords(): incoming (rows) concatenated with outgoing (cols)."""
        W = [torch.abs(l.weight).detach() for l in self.linear_layers]
        sigs = []
        if self.euclidean:
            n_layers = len(W)
            for p in range(n_layers + 1):
                parts = []
                if p > 0:          parts.append(W[p - 1])        # incoming: row per plane-p neuron
                if p < n_layers:   parts.append(W[p].t())        # outgoing: col per plane-p neuron
                sigs.append(torch.cat(parts, dim=1))
        else:
            for w in W:
                sigs.append(w.t())   # in-plane: outgoing signature
                sigs.append(w)       # out-plane: incoming signature
        return sigs

    def update_anchors(self):
        """(Re)select the K diverse+important anchor neurons per plane. Call after a
        warmup so the weights are meaningful, and periodically to refresh."""
        if self.k_anchors <= 0:
            self.anchor_idx = None
            return
        self.anchor_idx = [select_diverse_neurons(sig, self.k_anchors)
                           for sig in self._anchor_signatures()]

    def forward(self, *args):
        return self.model(*args)

    def optimize(self):
        print("init", self.get_cost(), flush=True)
        for layer, dist_coords in zip(self.linear_layers, self.linear_distance_matrices):
            weight_abs = torch.abs(layer.weight)
            x_in_perm, y_in_perm, x_out_perm, y_out_perm = optimize_coordinates(
                weight_abs, dist_coords[0], dist_coords[1], dist_coords[2], dist_coords[3], self.D)
            with torch.no_grad():
                dist_coords[0].copy_(x_in_perm)
                dist_coords[1].copy_(y_in_perm)
                dist_coords[2].copy_(x_out_perm)
                dist_coords[3].copy_(y_out_perm)

        for layer, dist_coords in zip(self.conv_layers, self.conv_distance_matrices):
            weight_abs = torch.mean(torch.abs(layer.weight), dim=(2, 3))
            x_in_perm, y_in_perm, x_out_perm, y_out_perm = optimize_coordinates(
                weight_abs, dist_coords[0], dist_coords[1], dist_coords[2], dist_coords[3], self.D)
            with torch.no_grad():
                dist_coords[0].copy_(x_in_perm)
                dist_coords[1].copy_(y_in_perm)
                dist_coords[2].copy_(x_out_perm)
                dist_coords[3].copy_(y_out_perm)

        print("final", self.get_cost(), flush=True)
