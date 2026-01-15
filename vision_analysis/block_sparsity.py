import torch
import torch.nn as nn
import numpy as np
import igraph as ig
from util import should_regularize_layer, compute_pruning_threshold_cpu


def block_sparsity_score(model, block_size=32, threshold=0.0):
    """
    Evaluate block sparsity of a model's weight matrices.

    For each regularized layer (Linear and Conv2d, excluding attention),
    divides the weight matrix into block_size x block_size blocks and counts
    how many blocks are "sparse" (all weights below threshold).

    Parameters:
        model: PyTorch model
        block_size: Size of square blocks (e.g., 32 for 32x32 blocks)
        threshold: Weights with abs value below this are considered zero

    Returns:
        dict with:
            - 'total_blocks': total number of blocks across all layers
            - 'sparse_blocks': number of blocks where all weights < threshold
            - 'block_sparsity_ratio': sparse_blocks / total_blocks
            - 'per_layer': dict of per-layer stats
    """
    total_blocks = 0
    sparse_blocks = 0
    per_layer = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_regularize_layer(name) and hasattr(module, 'weight'):
                weight = module.weight.detach().abs().cpu()
                out_f, in_f = weight.shape

                layer_total = 0
                layer_sparse = 0

                # Iterate over blocks
                for i in range(0, out_f, block_size):
                    for j in range(0, in_f, block_size):
                        # Extract block (may be smaller at edges)
                        block = weight[i:i+block_size, j:j+block_size]
                        layer_total += 1
                        if (block < threshold).all():
                            layer_sparse += 1

                per_layer[name] = {
                    'total_blocks': layer_total,
                    'sparse_blocks': layer_sparse,
                    'ratio': layer_sparse / layer_total if layer_total > 0 else 0
                }
                total_blocks += layer_total
                sparse_blocks += layer_sparse

        elif isinstance(module, nn.Conv2d):
            if should_regularize_layer(name) and hasattr(module, 'weight'):
                # For conv: use channel dimensions (out_ch, in_ch)
                weight = module.weight.detach().abs().cpu()
                weight_2d = weight.mean(dim=(2, 3))  # Average over spatial dims
                out_ch, in_ch = weight_2d.shape

                layer_total = 0
                layer_sparse = 0

                for i in range(0, out_ch, block_size):
                    for j in range(0, in_ch, block_size):
                        block = weight_2d[i:i+block_size, j:j+block_size]
                        layer_total += 1
                        if (block < threshold).all():
                            layer_sparse += 1

                per_layer[name] = {
                    'total_blocks': layer_total,
                    'sparse_blocks': layer_sparse,
                    'ratio': layer_sparse / layer_total if layer_total > 0 else 0
                }
                total_blocks += layer_total
                sparse_blocks += layer_sparse

    return {
        'total_blocks': total_blocks,
        'sparse_blocks': sparse_blocks,
        'block_sparsity_ratio': sparse_blocks / total_blocks if total_blocks > 0 else 0,
        'per_layer': per_layer
    }


def block_diagonal_ratio(model, block_size=32):
    """
    Measure how much weight magnitude is concentrated in diagonal blocks vs off-diagonal.

    For each regularized layer, divides the weight matrix into blocks and compares
    the average magnitude of "diagonal" blocks (bi == bj) vs "off-diagonal" blocks.

    Parameters:
        model: PyTorch model
        block_size: Size of square blocks

    Returns:
        dict with:
            - 'diagonal_avg': average magnitude in diagonal blocks
            - 'off_diagonal_avg': average magnitude in off-diagonal blocks
            - 'ratio': diagonal_avg / off_diagonal_avg (higher = more block-diagonal)
            - 'per_layer': dict of per-layer stats
    """
    total_diag_sum = 0.0
    total_diag_count = 0
    total_off_sum = 0.0
    total_off_count = 0
    per_layer = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_regularize_layer(name) and hasattr(module, 'weight'):
                weight = module.weight.detach().abs().cpu()
                out_f, in_f = weight.shape

                n_blocks_out = (out_f + block_size - 1) // block_size
                n_blocks_in = (in_f + block_size - 1) // block_size

                layer_diag_sum = 0.0
                layer_diag_count = 0
                layer_off_sum = 0.0
                layer_off_count = 0

                for bi in range(n_blocks_out):
                    for bj in range(n_blocks_in):
                        # Extract block
                        i_start = bi * block_size
                        i_end = min(i_start + block_size, out_f)
                        j_start = bj * block_size
                        j_end = min(j_start + block_size, in_f)
                        block = weight[i_start:i_end, j_start:j_end]
                        block_sum = block.sum().item()
                        block_count = block.numel()

                        # Diagonal block: scale indices for rectangular matrices
                        # This matches how block_matrix.py scale_match mode works
                        if n_blocks_out == n_blocks_in:
                            is_diag = (bi == bj)
                        elif n_blocks_out > n_blocks_in:
                            # Scale bi to match bj range
                            bi_scaled = bi * (n_blocks_in - 1) / max(n_blocks_out - 1, 1)
                            is_diag = abs(bi_scaled - bj) < 0.5
                        else:
                            # Scale bj to match bi range
                            bj_scaled = bj * (n_blocks_out - 1) / max(n_blocks_in - 1, 1)
                            is_diag = abs(bi - bj_scaled) < 0.5

                        if is_diag:
                            layer_diag_sum += block_sum
                            layer_diag_count += block_count
                        else:
                            layer_off_sum += block_sum
                            layer_off_count += block_count

                layer_diag_avg = layer_diag_sum / layer_diag_count if layer_diag_count > 0 else 0
                layer_off_avg = layer_off_sum / layer_off_count if layer_off_count > 0 else 0

                per_layer[name] = {
                    'diagonal_avg': layer_diag_avg,
                    'off_diagonal_avg': layer_off_avg,
                    'ratio': layer_diag_avg / layer_off_avg if layer_off_avg > 0 else float('inf')
                }

                total_diag_sum += layer_diag_sum
                total_diag_count += layer_diag_count
                total_off_sum += layer_off_sum
                total_off_count += layer_off_count

        elif isinstance(module, nn.Conv2d):
            if should_regularize_layer(name) and hasattr(module, 'weight'):
                weight = module.weight.detach().abs().cpu()
                weight_2d = weight.mean(dim=(2, 3))
                out_ch, in_ch = weight_2d.shape

                n_blocks_out = (out_ch + block_size - 1) // block_size
                n_blocks_in = (in_ch + block_size - 1) // block_size

                layer_diag_sum = 0.0
                layer_diag_count = 0
                layer_off_sum = 0.0
                layer_off_count = 0

                for bi in range(n_blocks_out):
                    for bj in range(n_blocks_in):
                        i_start = bi * block_size
                        i_end = min(i_start + block_size, out_ch)
                        j_start = bj * block_size
                        j_end = min(j_start + block_size, in_ch)
                        block = weight_2d[i_start:i_end, j_start:j_end]
                        block_sum = block.sum().item()
                        block_count = block.numel()

                        # Diagonal block: scale indices for rectangular matrices
                        if n_blocks_out == n_blocks_in:
                            is_diag = (bi == bj)
                        elif n_blocks_out > n_blocks_in:
                            bi_scaled = bi * (n_blocks_in - 1) / max(n_blocks_out - 1, 1)
                            is_diag = abs(bi_scaled - bj) < 0.5
                        else:
                            bj_scaled = bj * (n_blocks_out - 1) / max(n_blocks_in - 1, 1)
                            is_diag = abs(bi - bj_scaled) < 0.5

                        if is_diag:
                            layer_diag_sum += block_sum
                            layer_diag_count += block_count
                        else:
                            layer_off_sum += block_sum
                            layer_off_count += block_count

                layer_diag_avg = layer_diag_sum / layer_diag_count if layer_diag_count > 0 else 0
                layer_off_avg = layer_off_sum / layer_off_count if layer_off_count > 0 else 0

                per_layer[name] = {
                    'diagonal_avg': layer_diag_avg,
                    'off_diagonal_avg': layer_off_avg,
                    'ratio': layer_diag_avg / layer_off_avg if layer_off_avg > 0 else float('inf')
                }

                total_diag_sum += layer_diag_sum
                total_diag_count += layer_diag_count
                total_off_sum += layer_off_sum
                total_off_count += layer_off_count

    diag_avg = total_diag_sum / total_diag_count if total_diag_count > 0 else 0
    off_avg = total_off_sum / total_off_count if total_off_count > 0 else 0

    return {
        'diagonal_avg': diag_avg,
        'off_diagonal_avg': off_avg,
        'ratio': diag_avg / off_avg if off_avg > 0 else float('inf'),
        'per_layer': per_layer
    }


def evaluate_block_sparsity_at_pruning_level(model, block_size=32, p_percent=50):
    """
    Evaluate block sparsity after pruning to a given sparsity level.

    First computes the threshold that leaves p_percent of weights remaining,
    then evaluates block sparsity at that threshold.

    Parameters:
        model: PyTorch model
        block_size: Size of square blocks
        p_percent: Percentage of weights to keep (e.g., 50 means prune 50%)

    Returns:
        dict with block sparsity stats and the threshold used
    """
    threshold = compute_pruning_threshold_cpu(model, p_percent)
    stats = block_sparsity_score(model, block_size=block_size, threshold=threshold)
    stats['threshold'] = threshold
    stats['p_percent'] = p_percent
    return stats


def compare_models_block_sparsity(model_paths, block_size=32, threshold=0.01, model_loader_fn=None):
    """
    Compare block sparsity across multiple saved models.

    Parameters:
        model_paths: dict of {name: path} for saved model state dicts
        block_size: Size of square blocks
        threshold: Threshold for considering weights as zero
        model_loader_fn: Function that takes a path and returns a loaded model.
                         If None, assumes paths are to state_dicts and you need
                         to provide a base model architecture.

    Returns:
        dict of {model_name: block_sparsity_stats}
    """
    results = {}
    for name, path in model_paths.items():
        if model_loader_fn is not None:
            model = model_loader_fn(path)
            stats = block_sparsity_score(model, block_size=block_size, threshold=threshold)
            results[name] = stats
    return results


def _reorder_by_community(W_abs: np.ndarray, comm_threshold: float = 0.0):
    """
    Reorder a weight matrix by Louvain community detection.

    Returns:
        W_reordered: The reordered weight matrix
        order_in: Permutation of input indices
        order_out: Permutation of output indices
        Q: Modularity score
        n_comms: Number of communities
    """
    O, I = W_abs.shape
    js, is_ = np.nonzero(W_abs > comm_threshold)

    if js.size == 0:
        return W_abs, np.arange(I), np.arange(O), 0.0, 0

    weights = W_abs[js, is_].astype(float)
    edges = [(int(i), int(I + j)) for i, j in zip(is_, js)]
    # Create graph with explicit number of vertices to handle isolated nodes
    g = ig.Graph(n=I + O, edges=edges, directed=False)
    g.es["weight"] = weights.tolist()
    comm = g.community_multilevel(weights="weight")
    Q = float(g.modularity(comm, weights="weight"))
    membership = np.asarray(comm.membership, dtype=int)
    n_comms = int(membership.max() + 1)

    memb_in, memb_out = membership[:I], membership[I:]

    # Within-community degree for nicer sorting
    deg_in = np.zeros(I, dtype=float)
    deg_out = np.zeros(O, dtype=float)
    np.add.at(deg_in, is_, weights)
    np.add.at(deg_out, js, weights)

    order_in = np.lexsort((-deg_in, memb_in))
    order_out = np.lexsort((-deg_out, memb_out))

    W_reordered = W_abs[np.array(order_out)[:, None], np.array(order_in)[None, :]]

    return W_reordered, order_in, order_out, Q, n_comms


def block_sparsity_after_reorder(model, block_size=16, sparsity_threshold=0.0, comm_threshold=0.0):
    """
    Measure block sparsity after reordering weights by Louvain communities.

    This first detects communities in each weight matrix using Louvain algorithm,
    reorders rows/columns to group neurons by community, then measures how many
    block_size x block_size blocks are sparse (all weights below threshold).

    Parameters:
        model: PyTorch model
        block_size: Size of square blocks to check for sparsity
        sparsity_threshold: Weights below this are considered zero
        comm_threshold: Threshold for building the community graph

    Returns:
        dict with:
            - 'total_blocks': total blocks across all layers
            - 'sparse_blocks': blocks where all weights < sparsity_threshold
            - 'block_sparsity_ratio': sparse_blocks / total_blocks
            - 'avg_modularity': average Q across layers
            - 'per_layer': detailed per-layer stats
    """
    total_blocks = 0
    sparse_blocks = 0
    total_Q = 0.0
    n_layers = 0
    per_layer = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_regularize_layer(name) and hasattr(module, 'weight'):
                W_abs = module.weight.detach().abs().cpu().numpy()
                O, I = W_abs.shape

                # Reorder by community
                W_re, order_in, order_out, Q, n_comms = _reorder_by_community(W_abs, comm_threshold)

                # Count sparse blocks in reordered matrix
                layer_total = 0
                layer_sparse = 0

                for bi in range(0, O, block_size):
                    for bj in range(0, I, block_size):
                        block = W_re[bi:bi+block_size, bj:bj+block_size]
                        layer_total += 1
                        if (block < sparsity_threshold).all():
                            layer_sparse += 1

                per_layer[name] = {
                    'total_blocks': layer_total,
                    'sparse_blocks': layer_sparse,
                    'ratio': layer_sparse / layer_total if layer_total > 0 else 0,
                    'modularity_Q': Q,
                    'n_communities': n_comms,
                    'shape': (O, I)
                }

                total_blocks += layer_total
                sparse_blocks += layer_sparse
                total_Q += Q
                n_layers += 1

    return {
        'total_blocks': total_blocks,
        'sparse_blocks': sparse_blocks,
        'block_sparsity_ratio': sparse_blocks / total_blocks if total_blocks > 0 else 0,
        'avg_modularity': total_Q / n_layers if n_layers > 0 else 0,
        'per_layer': per_layer
    }


def evaluate_block_sparsity_reordered_at_pruning_level(model, block_size=16, p_percent=50, comm_threshold=0.0):
    """
    Evaluate block sparsity (after community reordering) at a given pruning level.

    First computes the threshold that leaves p_percent of weights remaining,
    then reorders by communities and measures block sparsity.

    Parameters:
        model: PyTorch model
        block_size: Size of square blocks
        p_percent: Percentage of weights to keep (e.g., 40 means keep 40%, prune 60%)
        comm_threshold: Threshold for building community graph

    Returns:
        dict with block sparsity stats after reordering
    """
    threshold = compute_pruning_threshold_cpu(model, p_percent)
    stats = block_sparsity_after_reorder(
        model,
        block_size=block_size,
        sparsity_threshold=threshold,
        comm_threshold=comm_threshold
    )
    stats['pruning_threshold'] = threshold
    stats['p_percent'] = p_percent
    return stats


def _get_sparse_block_mask_from_reorder(W_abs: np.ndarray, order_in: np.ndarray, order_out: np.ndarray,
                                         block_size: int, sparsity_threshold: float):
    """
    Create a mask for the ORIGINAL weight matrix that zeros sparse blocks found after reordering.

    Parameters:
        W_abs: Original weight matrix (absolute values)
        order_in: Permutation of input indices from community detection
        order_out: Permutation of output indices from community detection
        block_size: Size of blocks to check
        sparsity_threshold: Threshold below which weights are considered zero

    Returns:
        mask: Boolean mask of same shape as W_abs, True = keep, False = zero
    """
    O, I = W_abs.shape

    # Reorder the matrix
    W_reordered = W_abs[np.ix_(order_out, order_in)]

    # Find which blocks are sparse in the reordered space
    sparse_blocks = []
    for bi in range(0, O, block_size):
        for bj in range(0, I, block_size):
            block = W_reordered[bi:bi+block_size, bj:bj+block_size]
            if (block < sparsity_threshold).all():
                # Record the block indices in reordered space
                sparse_blocks.append((bi, min(bi+block_size, O), bj, min(bj+block_size, I)))

    # Create mask in reordered space
    mask_reordered = np.ones((O, I), dtype=bool)
    for (bi_start, bi_end, bj_start, bj_end) in sparse_blocks:
        mask_reordered[bi_start:bi_end, bj_start:bj_end] = False

    # Convert mask back to original space using inverse permutation
    inv_order_out = np.argsort(order_out)
    inv_order_in = np.argsort(order_in)
    mask_original = mask_reordered[np.ix_(inv_order_out, inv_order_in)]

    return mask_original


def prune_sparse_blocks_and_evaluate(model, test_loader, block_size=16, p_percent=50,
                                      comm_threshold=0.0, device='cuda'):
    """
    Zero out sparse blocks (found via community reordering) and evaluate accuracy.

    This function:
    1. For each Linear layer, detects communities and reorders
    2. Identifies blocks that are sparse (all weights < threshold)
    3. Zeros those blocks in the ORIGINAL weight matrix
    4. Evaluates accuracy on the test set

    Parameters:
        model: PyTorch model
        test_loader: DataLoader for test set
        block_size: Size of blocks to check for sparsity
        p_percent: Percentage of weights to keep (determines threshold)
        comm_threshold: Threshold for community graph construction
        device: Device to run evaluation on

    Returns:
        dict with:
            - 'accuracy': Test accuracy after block pruning
            - 'total_weights_zeroed': Number of weights zeroed
            - 'total_weights': Total weights in pruned layers
            - 'actual_sparsity': Fraction of weights zeroed
            - 'block_sparsity_ratio': Fraction of blocks that were sparse
            - 'per_layer': Per-layer stats
    """
    # First compute threshold
    threshold = compute_pruning_threshold_cpu(model, p_percent)

    # Store original weights and masks
    original_weights = {}
    masks = {}
    per_layer = {}
    total_zeroed = 0
    total_weights = 0
    total_blocks = 0
    sparse_blocks = 0

    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_regularize_layer(name) and hasattr(module, 'weight'):
                W_abs = module.weight.detach().abs().cpu().numpy()
                O, I = W_abs.shape

                # Store original
                original_weights[name] = module.weight.data.clone()

                # Get community reordering
                _, order_in, order_out, Q, n_comms = _reorder_by_community(W_abs, comm_threshold)

                # Get mask for sparse blocks
                mask = _get_sparse_block_mask_from_reorder(
                    W_abs, order_in, order_out, block_size, threshold
                )
                masks[name] = mask

                # Count stats
                layer_zeroed = (~mask).sum()
                layer_total = mask.size
                total_zeroed += layer_zeroed
                total_weights += layer_total

                # Count blocks
                W_reordered = W_abs[np.ix_(order_out, order_in)]
                layer_total_blocks = 0
                layer_sparse_blocks = 0
                for bi in range(0, O, block_size):
                    for bj in range(0, I, block_size):
                        block = W_reordered[bi:bi+block_size, bj:bj+block_size]
                        layer_total_blocks += 1
                        if (block < threshold).all():
                            layer_sparse_blocks += 1

                total_blocks += layer_total_blocks
                sparse_blocks += layer_sparse_blocks

                per_layer[name] = {
                    'weights_zeroed': int(layer_zeroed),
                    'total_weights': int(layer_total),
                    'sparsity': float(layer_zeroed / layer_total),
                    'sparse_blocks': layer_sparse_blocks,
                    'total_blocks': layer_total_blocks,
                    'block_sparsity': layer_sparse_blocks / layer_total_blocks if layer_total_blocks > 0 else 0,
                    'modularity_Q': Q,
                    'n_communities': n_comms
                }

                # Apply mask to model weights
                mask_tensor = torch.from_numpy(mask).to(module.weight.device)
                module.weight.data *= mask_tensor.float()

    # Evaluate accuracy
    model = model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total

    # Restore original weights
    for name, module in model.named_modules():
        if name in original_weights:
            module.weight.data = original_weights[name]

    return {
        'accuracy': accuracy,
        'total_weights_zeroed': int(total_zeroed),
        'total_weights': int(total_weights),
        'actual_sparsity': float(total_zeroed / total_weights) if total_weights > 0 else 0,
        'block_sparsity_ratio': float(sparse_blocks / total_blocks) if total_blocks > 0 else 0,
        'sparse_blocks': sparse_blocks,
        'total_blocks': total_blocks,
        'pruning_threshold': threshold,
        'p_percent': p_percent,
        'per_layer': per_layer
    }


def magnitude_prune_and_evaluate(model, test_loader, p_percent=50, device='cuda'):
    """
    Traditional magnitude-based pruning: zero out smallest weights and evaluate accuracy.

    Parameters:
        model: PyTorch model
        test_loader: DataLoader for test set
        p_percent: Percentage of weights to KEEP (e.g., 10 means keep 10%, prune 90%)
        device: Device to run evaluation on

    Returns:
        dict with accuracy and sparsity stats
    """
    threshold = compute_pruning_threshold_cpu(model, p_percent)

    # Store original weights
    original_weights = {}
    total_zeroed = 0
    total_weights = 0

    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_regularize_layer(name) and hasattr(module, 'weight'):
                original_weights[name] = module.weight.data.clone()

                # Create mask for weights below threshold
                mask = module.weight.detach().abs() >= threshold
                layer_zeroed = (~mask).sum().item()
                layer_total = mask.numel()

                total_zeroed += layer_zeroed
                total_weights += layer_total

                # Apply mask
                module.weight.data *= mask.float()

    # Evaluate accuracy
    model = model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total

    # Restore original weights
    for name, module in model.named_modules():
        if name in original_weights:
            module.weight.data = original_weights[name]

    return {
        'accuracy': accuracy,
        'weights_zeroed': int(total_zeroed),
        'total_weights': int(total_weights),
        'actual_sparsity': float(total_zeroed / total_weights) if total_weights > 0 else 0,
        'pruning_threshold': threshold,
        'p_percent': p_percent
    }


def compare_pruning_methods(model, test_loader, p_percent=10, block_size=16, device='cuda'):
    """
    Compare block pruning vs magnitude pruning at the same threshold.

    Returns a dict with results from both methods for easy comparison.
    """
    block_result = prune_sparse_blocks_and_evaluate(
        model, test_loader, block_size=block_size, p_percent=p_percent, device=device
    )

    magnitude_result = magnitude_prune_and_evaluate(
        model, test_loader, p_percent=p_percent, device=device
    )

    return {
        'block_pruning': {
            'accuracy': block_result['accuracy'],
            'sparsity': block_result['actual_sparsity'],
            'block_sparsity_ratio': block_result['block_sparsity_ratio']
        },
        'magnitude_pruning': {
            'accuracy': magnitude_result['accuracy'],
            'sparsity': magnitude_result['actual_sparsity']
        },
        'p_percent': p_percent,
        'block_size': block_size
    }
