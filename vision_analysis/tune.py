#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization for L1 and Spatial-Swap regularization.

Usage:
    # L1 mode - optimize gamma
    python tune.py --mode l1 --gamma-min 100 --gamma-max 10000 --sparsity 90 --trials 50

    # Spatial mode - optimize gamma-spatial, gamma-l1, and D
    python tune.py --mode spatial \
        --gamma-spatial-min 10 --gamma-spatial-max 1000 \
        --gamma-l1-min 0 --gamma-l1-max 5000 \
        --D-min 0.1 --D-max 10.0 \
        --sparsity 90 --trials 50

    # Use log scale for gamma search
    python tune.py --mode l1 --gamma-min 100 --gamma-max 10000 --log-scale --trials 50
"""

import argparse
import json
import os
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.optim import Adam

import classification_util
import spatial_wrapper_swap
import util


def parse_args():
    parser = argparse.ArgumentParser(description='Bayesian optimization for regularization hyperparameters')

    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['l1', 'spatial'],
                        help='Regularization mode: l1 or spatial')

    # L1 hyperparameter ranges
    parser.add_argument('--gamma-min', type=float, default=100,
                        help='Minimum gamma for L1 (l1 mode)')
    parser.add_argument('--gamma-max', type=float, default=10000,
                        help='Maximum gamma for L1 (l1 mode)')

    # Spatial hyperparameter ranges
    parser.add_argument('--gamma-spatial-min', type=float, default=10,
                        help='Minimum spatial gamma (spatial mode)')
    parser.add_argument('--gamma-spatial-max', type=float, default=1000,
                        help='Maximum spatial gamma (spatial mode)')
    parser.add_argument('--gamma-l1-min', type=float, default=0,
                        help='Minimum L1 gamma (spatial mode)')
    parser.add_argument('--gamma-l1-max', type=float, default=5000,
                        help='Maximum L1 gamma (spatial mode)')
    parser.add_argument('--D-min', type=float, default=0.1,
                        help='Minimum D parameter (spatial mode)')
    parser.add_argument('--D-max', type=float, default=10.0,
                        help='Maximum D parameter (spatial mode)')

    # Fixed spatial parameters
    parser.add_argument('--A', type=float, default=20.0,
                        help='Spatial A parameter (fixed)')
    parser.add_argument('--B', type=float, default=20.0,
                        help='Spatial B parameter (fixed)')

    # Search settings
    parser.add_argument('--log-scale', action='store_true',
                        help='Use log scale for gamma parameters')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of optimization trials')

    # Training settings
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs per trial')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')

    # Evaluation settings
    parser.add_argument('--sparsity', type=float, default=90,
                        help='Target sparsity percentage for evaluation')

    # Dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224',
                        help='Model architecture')

    # Output settings
    parser.add_argument('--log-dir', type=str, default='tune_logs',
                        help='Base directory for logs')
    parser.add_argument('--name', type=str, default=None,
                        help='Study name (auto-generated if not specified)')

    # Optuna settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--pruner', action='store_true',
                        help='Enable Optuna pruner for early stopping of bad trials')

    return parser.parse_args()


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
    """Find the threshold that achieves target_sparsity_percent sparsity."""
    all_weights = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if util.should_regularize_layer(name) and hasattr(module, 'weight'):
                all_weights.append(module.weight.abs().detach().cpu().flatten())

    if not all_weights:
        return 0.0

    all_weights = torch.cat(all_weights)

    k = int(len(all_weights) * target_sparsity_percent / 100)
    if k == 0:
        return 0.0
    if k >= len(all_weights):
        return all_weights.max().item()

    threshold = torch.kthvalue(all_weights, k).values.item()
    return threshold


def prune_and_evaluate(model, threshold, test_loader, device):
    """Prune weights below threshold and evaluate accuracy."""
    base_model = model.model if hasattr(model, 'model') else model

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
    accuracy = evaluate_accuracy(model, test_loader, device)

    return accuracy, actual_sparsity


def train_and_evaluate(args, hyperparams, train_loader, test_loader, num_classes, device, trial=None):
    """Train a model with given hyperparameters and return accuracy after pruning."""

    # Model
    model = classification_util.get_model(args.model, num_classes)

    # Wrap with spatial-circle-swap if needed
    if args.mode == 'spatial':
        model = spatial_wrapper_swap.SpatialNet(
            model, args.A, args.B, hyperparams['D'],
            circle=True, cluster=-1, distribution="spatial"
        )

    model = model.to(device)

    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add regularization
            if args.mode == 'l1':
                l1_loss = util.l1_linear_and_conv(model)
                loss += hyperparams['gamma'] * l1_loss
            else:  # spatial
                spatial_loss = model.get_cost(quadratic=False)
                loss += hyperparams['gamma_spatial'] * spatial_loss

                if hyperparams['gamma_l1'] > 0:
                    l1_loss = util.l1_linear_and_conv(model)
                    loss += hyperparams['gamma_l1'] * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        # Optional: report intermediate value for pruning
        if trial is not None and args.pruner and epoch % 2 == 0:
            # Quick evaluation for pruner
            test_acc = evaluate_accuracy(model, test_loader, device)
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Final evaluation with pruning
    acc_before = evaluate_accuracy(model, test_loader, device)
    threshold = compute_sparsity_threshold(model, args.sparsity)
    acc_after, actual_sparsity = prune_and_evaluate(model, threshold, test_loader, device)

    return acc_before, acc_after, actual_sparsity, threshold


def create_objective(args, train_loader, test_loader, num_classes, device):
    """Create the Optuna objective function."""

    def objective(trial):
        # Sample hyperparameters
        if args.mode == 'l1':
            if args.log_scale:
                gamma = trial.suggest_float('gamma', args.gamma_min, args.gamma_max, log=True)
            else:
                gamma = trial.suggest_float('gamma', args.gamma_min, args.gamma_max)
            hyperparams = {'gamma': gamma}
        else:  # spatial
            if args.log_scale:
                gamma_spatial = trial.suggest_float('gamma_spatial', args.gamma_spatial_min, args.gamma_spatial_max, log=True)
                gamma_l1 = trial.suggest_float('gamma_l1', max(args.gamma_l1_min, 1e-6), args.gamma_l1_max, log=True) if args.gamma_l1_max > 0 else 0
            else:
                gamma_spatial = trial.suggest_float('gamma_spatial', args.gamma_spatial_min, args.gamma_spatial_max)
                gamma_l1 = trial.suggest_float('gamma_l1', args.gamma_l1_min, args.gamma_l1_max)

            D = trial.suggest_float('D', args.D_min, args.D_max, log=args.log_scale)

            hyperparams = {
                'gamma_spatial': gamma_spatial,
                'gamma_l1': gamma_l1,
                'D': D,
            }

        # Train and evaluate
        acc_before, acc_after, actual_sparsity, threshold = train_and_evaluate(
            args, hyperparams, train_loader, test_loader, num_classes, device, trial
        )

        # Store additional info
        trial.set_user_attr('acc_before_pruning', acc_before)
        trial.set_user_attr('acc_after_pruning', acc_after)
        trial.set_user_attr('actual_sparsity', actual_sparsity)
        trial.set_user_attr('threshold', threshold)

        # Objective: maximize accuracy after pruning
        return acc_after

    return objective


def main():
    args = parse_args()

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_name = args.name or f"{args.mode}_tune_{args.dataset}_{timestamp}"

    log_dir = os.path.join(args.log_dir, study_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Study: {study_name}")
    print(f"Mode: {args.mode}")
    print(f"Target sparsity: {args.sparsity}%")
    print(f"Trials: {args.trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Log directory: {log_dir}")
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, test_loader, num_classes = classification_util.get_data_loaders(
        args.dataset, batch_size=args.batch_size
    )
    print(f"Dataset: {args.dataset}, Classes: {num_classes}")
    print()

    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner() if args.pruner else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # maximize accuracy
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    objective = create_objective(args, train_loader, test_loader, num_classes, device)

    print(f"Starting optimization with {args.trials} trials...")
    print("=" * 60)

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Results
    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print()

    best_trial = study.best_trial

    print(f"Best trial: #{best_trial.number}")
    print(f"Best accuracy after pruning: {best_trial.value:.2f}%")
    print(f"Accuracy before pruning: {best_trial.user_attrs['acc_before_pruning']:.2f}%")
    print(f"Actual sparsity: {best_trial.user_attrs['actual_sparsity']:.2f}%")
    print()
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value:.6f}")

    # Save results
    results = {
        'study_name': study_name,
        'mode': args.mode,
        'target_sparsity': args.sparsity,
        'trials': args.trials,
        'epochs_per_trial': args.epochs,
        'dataset': args.dataset,
        'model': args.model,
        'best_trial': {
            'number': best_trial.number,
            'accuracy_after_pruning': best_trial.value,
            'accuracy_before_pruning': best_trial.user_attrs['acc_before_pruning'],
            'actual_sparsity': best_trial.user_attrs['actual_sparsity'],
            'threshold': best_trial.user_attrs['threshold'],
            'hyperparameters': best_trial.params,
        },
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'user_attrs': t.user_attrs,
                'state': str(t.state),
            }
            for t in study.trials
        ],
        'search_ranges': {
            'gamma_min': args.gamma_min if args.mode == 'l1' else None,
            'gamma_max': args.gamma_max if args.mode == 'l1' else None,
            'gamma_spatial_min': args.gamma_spatial_min if args.mode == 'spatial' else None,
            'gamma_spatial_max': args.gamma_spatial_max if args.mode == 'spatial' else None,
            'gamma_l1_min': args.gamma_l1_min if args.mode == 'spatial' else None,
            'gamma_l1_max': args.gamma_l1_max if args.mode == 'spatial' else None,
            'D_min': args.D_min if args.mode == 'spatial' else None,
            'D_max': args.D_max if args.mode == 'spatial' else None,
            'log_scale': args.log_scale,
        }
    }

    results_path = os.path.join(log_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {results_path}")

    # Save Optuna study for later analysis
    study_path = os.path.join(log_dir, 'study.pkl')
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study saved to: {study_path}")

    print()
    print("To visualize results, you can load the study:")
    print(f"  import pickle")
    print(f"  study = pickle.load(open('{study_path}', 'rb'))")
    print(f"  optuna.visualization.plot_optimization_history(study)")


if __name__ == '__main__':
    main()
