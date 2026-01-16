import torch
from typing import Literal, Optional

Mode = Literal["center_shift", "scale_match", "pad_square", "wrap", "binary"]

def block_distance_matrix(
    x: int,
    y: int,
    group: int,
    mode: Mode = "scale_match",
    *,
    pad_penalty: float = 1e6,     # used by pad_square
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Build an (x, y) matrix partitioned into group-sized blocks (last blocks may be smaller).
    Each block [a, b] gets a constant value depending on the chosen rectangular strategy.

    Modes:
      1) center_shift: original approach (shift by (nx - ny)//2).
      2) scale_match: scale the longer axis so "diagonal" covers the rectangle smoothly.
      3) pad_square: pad the short axis with dummy blocks to make it square, then crop.
      4) wrap: treat blocks on each axis as circular (torus distance), useful for periodicity.
      5) binary: uses wrap logic but returns binary costs (0.1 for diagonal, 1.0 for off-diagonal).

    Args:
      x, y: matrix dims
      group: block size (>=1)
      mode: which rectangular handling to use
      pad_penalty: penalty used for interactions with dummy blocks in pad_square
      dtype, device: tensor dtype/device

    Returns:
      Tensor (x, y) with block-constant values satisfying your |a-b|+1 rule
      (adapted per mode for rectangles).
    """
    if group < 1:
        raise ValueError("group must be >= 1")

    nx = (x + group - 1) // group  # number of x-blocks
    ny = (y + group - 1) // group  # number of y-blocks

    # Block index per row/col
    a = torch.div(torch.arange(x, device=device), group, rounding_mode="floor")  # [0..nx-1]
    b = torch.div(torch.arange(y, device=device), group, rounding_mode="floor")  # [0..ny-1]

    if mode == "center_shift":
        # Your original rule
        shift = (nx - ny) // 2
        dist = (a - shift).unsqueeze(1) - b.unsqueeze(0)
        M = dist.abs().to(dtype) + 1
        return M

    elif mode == "scale_match":
        # Idea: line up the diagonals by scaling the longer axis so that
        # a in [0..nx-1] maps to ~ b in [0..ny-1] via affine scaling.
        # This keeps the "1" band truly diagonal across the rectangle.
        # If nx > ny, compress a; if ny > nx, compress b.
        if nx == ny:
            dist = a.unsqueeze(1) - b.unsqueeze(0)
            M = dist.abs().to(dtype) + 1
            return M

        if nx > ny:
            # target b* = a * (ny-1)/(nx-1)  (use 0 if nx==1 to avoid div-by-zero)
            denom = max(nx - 1, 1)
            a_scaled = a.to(torch.float32) * (ny - 1) / denom
            dist = a_scaled.unsqueeze(1) - b.unsqueeze(0).to(torch.float32)
        else:
            # target a* = b * (nx-1)/(ny-1)
            denom = max(ny - 1, 1)
            b_scaled = b.to(torch.float32) * (nx - 1) / denom
            dist = a.unsqueeze(1).to(torch.float32) - b_scaled.unsqueeze(0)

        M = dist.abs().to(dtype) + 1
        return M

    elif mode == "pad_square":
        # Pad the shorter axis with dummy blocks to make it square in block-space,
        # apply the simple |a-b|+1 rule, then crop the padding.
        n = max(nx, ny)
        # Build block distance on the padded index grid
        A = torch.arange(n, device=device)
        B = torch.arange(n, device=device)
        base = (A.unsqueeze(1) - B.unsqueeze(0)).abs().to(dtype) + 1  # (n, n)

        # Map each row/col to its block index; rows: a in [0..nx-1], cols: b in [0..ny-1]
        # Extract the top-left (nx, ny) submatrix
        M_block = base[:nx, :ny]  # block-level matrix

        # Any interactions that would have used padded (dummy) blocks get a high penalty.
        # But since we crop to (nx, ny), only real blocks remain. If you want to *discourage*
        # blocks near the "missing" diagonal due to padding, you can optionally bias:
        # (Most use-cases do not need extra bias because we cropped already.)
        # Nothing else needed.

        # Inflate to cell-level (x, y) by assigning each position its block value.
        M = M_block[a][:, b].to(dtype)
        return M

    elif mode == "wrap":
        # Torus distance over blocks: distance is the min of direct vs wrap-around.
        # This makes sense when endpoints are conceptually adjacent.
        # We need a rectangular generalization: compare a in [0..nx-1] with b in [0..ny-1]
        # by first scaling to a common circle length (LCM-style). A simple approach:
        # scale both onto [0, 1) and use circular absolute difference.
        eps = 1e-12
        a_circ = (a.to(torch.float32) + 0.5) / max(nx, 1)  # center-of-block in [0,1)
        b_circ = (b.to(torch.float32) + 0.5) / max(ny, 1)

        # pairwise circular distance
        aa = a_circ.unsqueeze(1)  # (x,1)
        bb = b_circ.unsqueeze(0)  # (1,y)
        diff = (aa - bb).abs()
        circ_dist = torch.minimum(diff, 1.0 - diff + eps)

        # Convert to a block-like scale: multiply by max(nx, ny)-1 so that a perfect match ~0
        # then +1 to keep your "1 on the diagonal-band" convention.
        scaled = circ_dist * max(max(nx, ny) - 1, 1)
        M = scaled.to(dtype) + 1
        return M

    elif mode == "binary":
        # Binary mode: creates a "fat diagonal" that covers all blocks evenly.
        # For a 48x12 block matrix: each of 12 input blocks maps to 4 output blocks.
        # - 0.1 for diagonal blocks
        # - 1.0 for off-diagonal blocks

        if nx >= ny:
            # More output blocks than input blocks
            # Output block i is diagonal with input block floor(i * ny / nx)
            a_mapped = torch.div(a * ny, nx, rounding_mode='floor')
            is_diagonal = (a_mapped.unsqueeze(1) == b.unsqueeze(0))
        else:
            # More input blocks than output blocks
            # Input block j is diagonal with output block floor(j * nx / ny)
            b_mapped = torch.div(b * nx, ny, rounding_mode='floor')
            is_diagonal = (a.unsqueeze(1) == b_mapped.unsqueeze(0))

        M = torch.where(is_diagonal, torch.tensor(0.1, dtype=dtype, device=device),
                        torch.tensor(1.0, dtype=dtype, device=device))
        return M

    else:
        raise ValueError(f"Unknown mode: {mode}")
