"""
Per-layer spatial regularizer: EVERY forward projection is spatialized as its own little
MLP, in its OWN 2D space (decoupled -- no cross-layer position consistency).

  - Linear (out, in): 'in' input neurons on plane z = 0, 'out' output neurons on z = D.
  - Conv2d: the kernel (C_out, C_in, kH, kW) is reshaped to a matrix (C_out, C_in*kH*kW)
    and treated as a dense layer. 'in' neurons = the C_in*kH*kW PATCH-SLOTS (one per
    (in_channel, dy, dx)); 'out' neurons = the C_out output channels. Every kernel entry
    is one weight = one connection = one length. Weight sharing NEVER enters the cost:
    the kernel matrix is spatialized once, exactly like an MLP layer.

Cost per layer = mean(|W| * wire_length) + within-plane repulsion,
with wire_length = sqrt(dx^2 + dy^2 + D^2). Planes are independent across layers.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def _assign_factored(A, Bm, block=0, rounds=5):
    """Assign rows of the cost matrix cost = A @ Bm.T (cost[i,k] = A[i] . Bm[k])
    to columns, minimizing sum(cost[i, perm[i]]). A, Bm are (K, d).

    Exact Hungarian if K<=block or block==0. Otherwise BLOCK coordinate descent:
    re-permute a random subset of neurons optimally among the sites they currently
    occupy -- an exact small Hungarian on a block-local cost that is computed DIRECTLY
    (A[B] @ Bm[slots].T), so the full K*K cost matrix is never materialized. That is
    the key to tractability for wide layers: cost per round is O(K*block*d), not O(K^2*d)."""
    K = A.shape[0]
    if block == 0 or K <= block:
        _, perm = linear_sum_assignment(A @ Bm.T)
        return perm
    perm = np.arange(K)
    for _ in range(rounds):
        order = np.random.permutation(K)
        for st in range(0, K, block):
            B = order[st:st + block]
            slots = perm[B]
            sub = A[B] @ Bm[slots].T          # block x block, built directly
            _, c = linear_sum_assignment(sub)
            perm[B] = slots[c]
    return perm


def optimize_coordinates(W, xin, yin, xout, yout, D, max_iter=100, tol=1e-6, block=0):
    """Discrete SWAP: reassign neurons to grid positions to minimize mean(|W|*dist),
    dist = sqrt(dx^2+dy^2+D^2). Alternating assignment (row=out, col=in). If block>0,
    layers larger than `block` use blocked coordinate-descent assignment instead of a
    full Hungarian (tractable for wide conv layers). Returns permuted positions."""
    inpos = np.stack([xin, yin], 1)        # (in, 2)
    outpos = np.stack([xout, yout], 1)     # (out, 2)
    d2 = ((outpos[:, None, :] - inpos[None, :, :]) ** 2).sum(-1)   # (out, in)
    C = np.sqrt(d2 + D ** 2)
    m, n = W.shape
    row_perm = np.arange(m); col_perm = np.arange(n)

    if block and (m > block or n > block):
        # blocked path: fixed iters, never build the full m*m or n*n cost matrix
        for _ in range(max_iter):
            row_perm = _assign_factored(W, C[:, col_perm], block, rounds=3)      # output neurons
            col_perm = _assign_factored(W.T, C[row_perm, :].T, block, rounds=3)  # input neurons
        return xin[col_perm], yin[col_perm], xout[row_perm], yout[row_perm]

    def obj(rp, cp):
        return (W * C[np.ix_(rp, cp)]).mean()

    prev = obj(row_perm, col_perm)
    for _ in range(max_iter):
        row_perm = _assign_factored(W, C[:, col_perm])     # assign output neurons
        col_perm = _assign_factored(W.T, C[row_perm, :].T)  # assign input neurons
        cur = obj(row_perm, col_perm)
        if abs(prev - cur) < tol:
            break
        prev = cur
    return xin[col_perm], yin[col_perm], xout[row_perm], yout[row_perm]


def _grid_init(n, side, seed):
    """Even 2D grid over a side x side square, neuron->site assignment shuffled so any
    final structure is learned, not baked in."""
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    xs = torch.linspace(-side / 2, side / 2, cols)
    ys = torch.linspace(-side / 2, side / 2, rows)
    gx, gy = torch.meshgrid(xs, ys, indexing='xy')
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return gx.flatten()[:n][perm].clone(), gy.flatten()[:n][perm].clone()


def _grid_init_sites(n, side, B, seed):
    """Coarse-grid init for BLOCK mode: place ceil(n/B) discrete SITES on a grid and
    give each site capacity B, so the length-n position array has B identical coords per
    site. A permutation of that array (what swap does) moves neurons between sites while
    keeping exactly B per site -> 'swap' = switching which group a neuron belongs to.
    If n < B the whole dimension collapses to a single group (degenerate conv dims)."""
    n_sites = max(1, int(math.ceil(n / B)))
    rows = int(math.ceil(math.sqrt(n_sites)))
    cols = int(math.ceil(n_sites / rows))
    xs = torch.linspace(-side / 2, side / 2, cols)
    ys = torch.linspace(-side / 2, side / 2, rows)
    gx, gy = torch.meshgrid(xs, ys, indexing='xy')
    sx = gx.flatten()[:n_sites]; sy = gy.flatten()[:n_sites]
    slot_site = torch.clamp(torch.arange(n) // B, max=n_sites - 1)   # B slots per site
    px = sx[slot_site].clone(); py = sy[slot_site].clone()
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return px[perm].clone(), py[perm].clone()


class SpatialCNN(nn.Module):
    def __init__(self, model, D=1.0, side=20.0, gamma=1.0,
                 collision_radius=1.0, repulsion='exp', fr_eps=0.1, device='cuda',
                 exclude=(), block_size=0):
        super().__init__()
        self.model = model
        self.D = D
        self.side = side
        self.gamma = gamma
        self.collision_radius = collision_radius
        self.repulsion = repulsion
        self.fr_eps = fr_eps
        self.device = device
        # BLOCK mode: neurons share capacity-`block_size` sites (groups). Cost is
        # UNCHANGED (Sum |w|*dist); positions are frozen and repulsion is off so the
        # groups stay intact and swap only re-assigns group membership.
        self.block_size = block_size

        self.layers = []          # the nn.Linear / nn.Conv2d modules, in order
        self.names = []           # display label per layer
        self.meta = []            # (n_in, n_out, kind, extra) per layer
        self.pos = nn.ParameterList()   # 4 params per layer: xin, yin, xout, yout

        seed = 0
        for name, m in model.named_modules():
            if any(e in name for e in exclude):   # skip regularizing these (e.g. Q,K)
                continue
            if isinstance(m, nn.Linear):
                n_in, n_out, kind, extra = m.in_features, m.out_features, 'linear', None
            elif isinstance(m, nn.Conv2d):
                kh, kw = m.kernel_size
                n_in, n_out, kind, extra = (m.in_channels // m.groups) * kh * kw, m.out_channels, 'conv', (m.in_channels, kh, kw)
            else:
                continue
            self.layers.append(m)
            self.names.append(f'{name or kind} [{kind}]  {n_in}->{n_out}')
            self.meta.append((n_in, n_out, kind, extra))
            if block_size > 0:
                xi, yi = _grid_init_sites(n_in, side, block_size, seed); seed += 1
                xo, yo = _grid_init_sites(n_out, side, block_size, seed); seed += 1
            else:
                xi, yi = _grid_init(n_in, side, seed); seed += 1
                xo, yo = _grid_init(n_out, side, seed); seed += 1
            # BLOCK mode freezes positions (requires_grad=False): only swap moves neurons.
            rg = block_size == 0
            self.pos += [nn.Parameter(xi, requires_grad=rg), nn.Parameter(yi, requires_grad=rg),
                         nn.Parameter(xo, requires_grad=rg), nn.Parameter(yo, requires_grad=rg)]

    def weight_matrix(self, l):
        """The layer's weights as a 2D (out, in) matrix; convs are reshaped to
        (C_out, C_in*kH*kW)."""
        layer = self.layers[l]
        if isinstance(layer, nn.Conv2d):
            return layer.weight.view(layer.out_channels, -1)
        return layer.weight

    def planes(self, l):
        """(xin, yin, xout, yout) parameters for layer l."""
        return self.pos[4 * l], self.pos[4 * l + 1], self.pos[4 * l + 2], self.pos[4 * l + 3]

    def position_parameters(self):
        return list(self.pos)

    def _repel(self, x, y):
        n = x.numel()
        if n < 2:
            return x.new_zeros(())
        pos = torch.stack((x, y), dim=1)
        d = torch.cdist(pos, pos)
        iu = torch.triu_indices(n, n, offset=1, device=d.device)
        dd = d[iu[0], iu[1]]
        if self.repulsion == 'exp':
            return torch.exp(-dd).mean()
        if self.repulsion == 'fr':
            k2 = (self.side * self.side) / n
            return (k2 / torch.sqrt(dd * dd + self.fr_eps ** 2)).mean()
        return (torch.relu(self.collision_radius - dd) ** 2).mean()   # collision

    @torch.no_grad()
    def swap(self, block=0):
        """Relocate neurons to grid positions minimizing wiring cost, per layer
        (discrete assignment). Positions stay on the fixed grid; only the
        neuron->position assignment changes. Call periodically during training.
        block>0 uses blocked coordinate-descent assignment for layers wider than
        `block` (makes wide conv layers tractable)."""
        for l in range(len(self.layers)):
            xin, yin, xout, yout = self.planes(l)
            W = self.weight_matrix(l).abs().detach().cpu().numpy()
            xi, yi, xo, yo = optimize_coordinates(
                W,
                xin.detach().cpu().numpy(), yin.detach().cpu().numpy(),
                xout.detach().cpu().numpy(), yout.detach().cpu().numpy(), self.D,
                block=block, max_iter=(3 if block else 100))
            dev = xin.device
            xin.copy_(torch.from_numpy(xi).to(dev)); yin.copy_(torch.from_numpy(yi).to(dev))
            xout.copy_(torch.from_numpy(xo).to(dev)); yout.copy_(torch.from_numpy(yo).to(dev))

    def get_cost(self):
        wire = 0.0; nparam = 0; rep = 0.0; nrep = 0
        for l in range(len(self.layers)):
            xin, yin, xout, yout = self.planes(l)
            W = self.weight_matrix(l).abs()                    # (out, in)
            dx = xout[:, None] - xin[None, :]
            dy = yout[:, None] - yin[None, :]
            dist = torch.sqrt(dx * dx + dy * dy + self.D ** 2)  # (out, in)
            wire = wire + (W * dist).sum(); nparam += W.numel()
            if self.block_size == 0:      # repulsion off in block mode (sites are shared on purpose)
                rep = rep + self._repel(xin, yin) + self._repel(xout, yout); nrep += 2
        return self.gamma * (wire / nparam + (rep / nrep if nrep else 0.0))

    def forward(self, *args):
        return self.model(*args)
