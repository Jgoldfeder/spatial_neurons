"""
Knob-free variant of spatial_wrapper_planes.py (subclasses it; the original is
untouched). Removes the anchor mechanism (K_ANCHORS / strength / radius) by
replacing its job -- long-range spreading pressure -- with two principled sources:

1. FIXED BOUNDARY PLANES (fixed_boundary=True, coupled mode only): the input
   plane is pinned to the TRUE input geometry (e.g. the 28x28 pixel grid) and
   the output plane to a fixed circle of classes; neither is trainable. Hidden
   neurons hang between two fixed, spread scaffolds: wiring attraction toward
   spread endpoints both prevents collapse (the job the anchors were doing) and
   seeds functional geography directly. The _grid_init shuffle rationale does
   not apply to pinned planes: their positions are task-defined, not learned,
   so nothing about the HIDDEN layout is pre-separated.

2. SCALE-FREE REPULSION (fr_repulsion=True): the hard-shell collision
   relu(r - d)^2 -- exactly zero beyond r, so it provides no long-range
   spreading and jams once packed -- is replaced on free (hidden) planes by a
   smooth all-pairs Coulomb-style term  k^2 / sqrt(d^2 + eps^2)  with
   k = sqrt(A*B / n), the natural spacing of n points in an A x B plane.
   k is computed from the geometry, not tuned. Long-range and smooth, so
   functional groups can slide past each other instead of jamming
   (Fruchterman-Reingold-style attraction/repulsion balance).

Linear layers only (this variant is for the MLP experiments).
"""

import math

import torch
import torch.nn as nn

import spatial_wrapper_planes as swp


class SpatialNetV2(swp.SpatialNet):
    def __init__(self, model, A, B, D, device="cuda",
                 coupled=True, fixed_boundary=True, input_shape=None,
                 output_layout='circle', pin_inputs=True, pin_outputs=True,
                 repulsion='fr', fr_repulsion=None, repulsion_strength=1.0,
                 fr_eps=0.1, **kw):
        kw.setdefault('k_anchors', 0)
        self._prev_out = None   # coupled-chain cursor, used by _extract_layers (runs in super().__init__)
        super().__init__(model, A, B, D, device=device, coupled=coupled, **kw)
        assert not self.conv_layers, "SpatialNetV2 spatializes Linear layers only (Conv2d is a plain front-end)"
        self.fixed_boundary = fixed_boundary
        # repulsion mode on FREE planes:
        #   'fr'        : k^2 / sqrt(d^2 + eps^2)  -- scale-free, LONG-range (unbounded;
        #                 inflates against the linear wiring attraction, esp. small planes)
        #   'exp'       : exp(-d)                  -- Wolczyk et al. 2019 density term;
        #                 SHORT-range and self-limiting (no runaway, weaker spreading)
        #   'collision' : relu(r - d)^2            -- original hard-shell collision
        # fr_repulsion (bool) is kept for back-compat: True->'fr', False->'collision'.
        if fr_repulsion is not None:
            repulsion = 'fr' if fr_repulsion else 'collision'
        assert repulsion in ('fr', 'exp', 'collision'), repulsion
        self.repulsion = repulsion
        self.fr_repulsion = (repulsion == 'fr')
        self.repulsion_strength = repulsion_strength
        self.fr_eps = fr_eps
        if fixed_boundary:
            assert self.euclidean, "fixed_boundary requires coupled=True"
            # boundary GEOMETRY (pixel-grid inputs, circle outputs) is always set;
            # pin_inputs / pin_outputs control whether those planes may then MOVE.
            # Unpinned planes are warm-started at the task geometry and train like
            # hidden planes (and receive the repulsion term).
            self._pin_boundary(input_shape, output_layout, pin_inputs, pin_outputs)

    def _extract_layers(self, module, prefix=''):
        """Like the parent, but Conv2d layers are SKIPPED (not spatialized) -- they act
        as a plain, unregularized feature front-end. Only Linear layers get positions,
        so the coupled plane stack is built purely from the MLP; the conv's output
        feature map becomes fc1's input, which _pin_boundary pins as the retina."""
        for name, layer in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            if name in ('head', 'fc', 'classifier'):
                continue
            if isinstance(layer, nn.Conv2d):
                continue                      # unspatialized front-end: ignore
            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                dm = swp.compute_distance_matrix(layer.in_features, layer.out_features,
                                                 self.A, self.B, prev=self._prev_out)
                self.linear_distance_matrices.append(dm)
                if self.euclidean:
                    self._prev_out = (dm[2], dm[3])
            else:
                self._extract_layers(layer, full_name)

    def _pin_boundary(self, input_shape, output_layout, pin_inputs=True, pin_outputs=True):
        dmats = self.linear_distance_matrices

        # input plane -> the true input geometry (pixel grid), spanning A x B.
        # input_shape may be (H, W) grayscale or (C, H, W) multi-channel; channels are
        # tiled side by side (a C*W-wide retina), so each channel is a contiguous block
        # and no two inputs share a position. Flatten order is channel-major, matching
        # torch's x.flatten(1) on a (C, H, W) image.
        x_in, y_in = dmats[0][0], dmats[0][1]
        n_in = x_in.numel()
        if input_shape is not None:
            if len(input_shape) == 3:
                C, H, W = input_shape
            else:
                C, (H, W) = 1, input_shape
            assert C * H * W == n_in, f"input_shape {input_shape} != {n_in} inputs"
        else:
            C = 1; H = W = int(math.ceil(math.sqrt(n_in)))
        idx = torch.arange(n_in)
        ch = idx // (H * W)
        rem = idx % (H * W)
        row = (rem // W).float()
        col = (rem % W).float()
        total_cols = C * W
        gx = ((ch.float() * W + col) / max(total_cols - 1, 1) - 0.5) * self.A
        gy = (0.5 - row / max(H - 1, 1)) * self.B      # image row 0 at the top
        with torch.no_grad():
            x_in.copy_(gx); y_in.copy_(gy)
        if pin_inputs:
            x_in.requires_grad_(False); y_in.requires_grad_(False)

        # output plane -> fixed layout (circle: class k at angle 2*pi*k/M)
        x_out, y_out = dmats[-1][2], dmats[-1][3]
        m = x_out.numel()
        if output_layout == 'circle':
            ang = torch.arange(m).float() / m * 2 * math.pi
            ox = 0.35 * self.A * torch.cos(ang)
            oy = 0.35 * self.B * torch.sin(ang)
        elif output_layout == 'line':
            ox = (torch.arange(m).float() / max(m - 1, 1) - 0.5) * self.A
            oy = torch.zeros(m)
        else:
            raise ValueError(f"unknown output_layout {output_layout!r}")
        with torch.no_grad():
            x_out.copy_(ox); y_out.copy_(oy)
        if pin_outputs:
            x_out.requires_grad_(False); y_out.requires_grad_(False)

    def free_position_parameters(self):
        """Trainable position parameters only (pinned boundary planes excluded),
        deduplicated -- coupled planes share Parameters between adjacent layers.
        Use this for the optimizer's position param group."""
        seen = {}
        for dm in self.linear_distance_matrices:
            for p in dm:
                if p.requires_grad:
                    seen[id(p)] = p
        return list(seen.values())

    def get_cost(self, quadratic=False):
        # wiring term, identical to the parent
        total_cost, total_params = 0.0, 0
        for layer, dc in zip(self.linear_layers, self.linear_distance_matrices):
            dist = swp.compute_distance_matrix_cdist(dc[0], dc[1], dc[2], dc[3], self.D)
            w = torch.abs(layer.weight)
            if self.detach_weights:
                w = w.detach()
            d = dist * dist if quadratic else dist
            total_cost += torch.sum(w * d.to(self.device))
            total_params += w.numel()
        cost = self.spatial_cost_scale * total_cost / total_params

        # repulsion on FREE planes only (pinned planes are spread by construction)
        rep, n_free = 0.0, 0
        for (x, y) in self._plane_coords():
            if not (x.requires_grad or y.requires_grad):
                continue
            n = x.numel()
            if n < 2:
                continue
            if self.repulsion == 'fr':
                k2 = (self.A * self.B) / n                     # natural spacing^2
                pos = torch.stack((x, y), dim=1)
                dmat = torch.cdist(pos, pos)
                iu = torch.triu_indices(n, n, offset=1, device=dmat.device)
                d = dmat[iu[0], iu[1]]
                rep = rep + (k2 / torch.sqrt(d * d + self.fr_eps ** 2)).mean()
            elif self.repulsion == 'exp':
                # Wolczyk et al. 2019 density cost: mean_{i<j} exp(-||p_i - p_j||)
                pos = torch.stack((x, y), dim=1)
                dmat = torch.cdist(pos, pos)
                iu = torch.triu_indices(n, n, offset=1, device=dmat.device)
                d = dmat[iu[0], iu[1]]
                rep = rep + torch.exp(-d).mean()
            else:
                pairs = n * (n - 1) / 2.0
                rep = rep + swp.collision_penalty(x, y, self.collision_radius) / pairs
            n_free += 1
        if n_free:
            cost = cost + self.spatial_cost_scale * self.repulsion_strength * rep / n_free
        return cost
