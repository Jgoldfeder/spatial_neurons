#!/usr/bin/env python3
"""
Clean training script for L1 and Spatial-Swap regularization experiments.

Usage:
    # L1 mode
    python train.py --mode l1 --gamma 1000 --epochs 10 --sparsity 90

    # Spatial-swap mode
    python train.py --mode spatial --gamma-spatial 100 --gamma-l1 500 --D 1.0 --epochs 10 --sparsity 90

    # Specify dataset and model
    python train.py --mode l1 --gamma 1000 --dataset cifar100 --model vit_tiny_patch16_224
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam

import classification_util
import spatial_wrapper_swap
import util


def parse_args():
    parser = argparse.ArgumentParser(description='Train with L1 or Spatial regularization')

    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['l1', 'spatial'],
                        help='Regularization mode: l1 or spatial')

    # L1 hyperparameters
    parser.add_argument('--gamma', type=float, default=1000,
                        help='L1 regularization strength (for l1 mode)')

    # Spatial hyperparameters
    parser.add_argument('--gamma-spatial', type=float, default=100,
                        help='Spatial regularization strength (for spatial mode)')
    parser.add_argument('--gamma-l1', type=float, default=0,
                        help='L1 regularization strength (for spatial mode, 0 = no L1)')
    parser.add_argument('--D', type=float, default=1.0,
                        help='Spatial distance parameter')
    parser.add_argument('--A', type=float, default=20.0,
                        help='Spatial A parameter (fixed)')
    parser.add_argument('--B', type=float, default=20.0,
                        help='Spatial B parameter (fixed)')

    # Training settings
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')

    # Evaluation settings
    parser.add_argument('--sparsity', type=float, default=90,
                        help='Target sparsity percentage for evaluation (e.g., 90 means 90%% sparse)')

    # Dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224',
                        help='Model architecture')

    # Output settings
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Base directory for logs')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (auto-generated if not specified)')

    return parser.parse_args()


def get_experiment_name(args):
    """Generate experiment name from hyperparameters."""
    if args.name:
        return args.name

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.mode == 'l1':
        return f"l1_g{args.gamma}_{args.dataset}_{timestamp}"
    else:
        return f"spatial_gs{args.gamma_spatial}_gl1{args.gamma_l1}_D{args.D}_{args.dataset}_{timestamp}"


def setup_logging(args, exp_name):
    """Create logging directories and return paths."""
    base_dir = os.path.join(args.log_dir, exp_name)

    metrics_dir = os.path.join(base_dir, 'metrics')
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return {
        'base': base_dir,
        'metrics': metrics_dir,
        'models': models_dir,
        'results': results_dir,
    }


class Logger:
    """Simple logger that writes to both stdout and file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w')

    def log(self, msg):
        print(msg)
        self.file.write(msg + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


def evaluate_accuracy(model, data_loader, device):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def compute_sparsity_threshold(model, target_sparsity_percent):
    """
    Find the threshold that achieves target_sparsity_percent sparsity
    on regularized layers only.
    """
    all_weights = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if util.should_regularize_layer(name) and hasattr(module, 'weight'):
                all_weights.append(module.weight.abs().detach().cpu().flatten())

    if not all_weights:
        return 0.0

    all_weights = torch.cat(all_weights)

    # Find threshold for target sparsity
    k = int(len(all_weights) * target_sparsity_percent / 100)
    if k == 0:
        return 0.0
    if k >= len(all_weights):
        return all_weights.max().item()

    threshold = torch.kthvalue(all_weights, k).values.item()
    return threshold


def prune_and_evaluate(model, threshold, test_loader, device):
    """Prune weights below threshold and evaluate accuracy."""
    # Get the underlying model if wrapped
    base_model = model.model if hasattr(model, 'model') else model

    # Prune weights
    pruned_count = 0
    total_count = 0

    with torch.no_grad():
        for name, module in base_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if util.should_regularize_layer(name) and hasattr(module, 'weight'):
                    mask = module.weight.abs() < threshold
                    pruned_count += mask.sum().item()
                    total_count += module.weight.numel()
                    module.weight[mask] = 0.0

    actual_sparsity = 100.0 * pruned_count / total_count if total_count > 0 else 0.0

    # Evaluate
    accuracy = evaluate_accuracy(model, test_loader, device)

    return accuracy, actual_sparsity


def train(args):
    # Setup
    exp_name = get_experiment_name(args)
    paths = setup_logging(args, exp_name)
    logger = Logger(os.path.join(paths['results'], 'training.txt'))

    logger.log(f"Experiment: {exp_name}")
    logger.log(f"Mode: {args.mode}")
    logger.log(f"Args: {vars(args)}")
    logger.log("")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Device: {device}")

    # Data
    train_loader, test_loader, num_classes = classification_util.get_data_loaders(
        args.dataset, batch_size=args.batch_size
    )
    logger.log(f"Dataset: {args.dataset}, Classes: {num_classes}")

    # Model
    model = classification_util.get_model(args.model, num_classes)

    # Wrap with spatial-circle-swap if needed
    if args.mode == 'spatial':
        model = spatial_wrapper_swap.SpatialNet(
            model, args.A, args.B, args.D,
            circle=True, cluster=-1, distribution="spatial"
        )
        logger.log(f"Wrapped model with SpatialNet circle-swap (A={args.A}, B={args.B}, D={args.D})")

    model = model.to(device)

    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    logger.log(f"Training for {args.epochs} epochs")
    logger.log("")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Perform Hungarian swap optimization for spatial mode
        if args.mode == 'spatial':
            model.optimize()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add regularization
            if args.mode == 'l1':
                l1_loss = util.l1_linear_and_conv(model)
                loss += args.gamma * l1_loss
            else:  # spatial
                # Spatial penalty
                spatial_loss = model.get_cost(quadratic=False)
                loss += args.gamma_spatial * spatial_loss

                # Optional L1 penalty
                if args.gamma_l1 > 0:
                    l1_loss = util.l1_linear_and_conv(model)
                    loss += args.gamma_l1 * l1_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Evaluate every 5 epochs or on final epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            test_acc = evaluate_accuracy(model, test_loader, device)
            logger.log(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        else:
            logger.log(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}%")

    logger.log("")

    # Save model before pruning
    base_model = model.model if hasattr(model, 'model') else model
    state_dict = base_model.state_dict()
    model_path = os.path.join(paths['models'], 'model.pt')
    torch.save(state_dict, model_path)
    logger.log(f"Saved model to {model_path}")

    # Final evaluation with pruning
    logger.log("")
    logger.log(f"=== Evaluation at {args.sparsity}% sparsity ===")

    # Accuracy before pruning
    acc_before = evaluate_accuracy(model, test_loader, device)
    logger.log(f"Accuracy before pruning: {acc_before:.2f}%")

    # Find threshold and prune
    threshold = compute_sparsity_threshold(model, args.sparsity)
    logger.log(f"Threshold for {args.sparsity}% sparsity: {threshold:.6f}")

    acc_after, actual_sparsity = prune_and_evaluate(model, threshold, test_loader, device)
    logger.log(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
    logger.log(f"Accuracy after pruning: {acc_after:.2f}%")

    # Save metrics
    metrics = {
        'experiment_name': exp_name,
        'mode': args.mode,
        'hyperparameters': {
            'gamma': args.gamma if args.mode == 'l1' else None,
            'gamma_spatial': args.gamma_spatial if args.mode == 'spatial' else None,
            'gamma_l1': args.gamma_l1 if args.mode == 'spatial' else None,
            'D': args.D if args.mode == 'spatial' else None,
        },
        'training': {
            'epochs': args.epochs,
            'lr': args.lr,
            'dataset': args.dataset,
            'model': args.model,
        },
        'evaluation': {
            'target_sparsity': args.sparsity,
            'actual_sparsity': actual_sparsity,
            'threshold': threshold,
            'accuracy_before_pruning': acc_before,
            'accuracy_after_pruning': acc_after,
        }
    }

    metrics_path = os.path.join(paths['metrics'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.log(f"Saved metrics to {metrics_path}")

    logger.close()

    print(f"\n=== Summary ===")
    print(f"Experiment: {exp_name}")
    print(f"Accuracy before pruning: {acc_before:.2f}%")
    print(f"Accuracy after {actual_sparsity:.1f}% pruning: {acc_after:.2f}%")
    print(f"Logs saved to: {paths['base']}")

    return metrics


if __name__ == '__main__':
    args = parse_args()
    train(args)
