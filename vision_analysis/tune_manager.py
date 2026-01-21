#!/usr/bin/env python3
"""
Central manager for multiple parallel hyperparameter tuning studies.

Workers connect to this manager, which assigns them to whichever study
has the fewest completed trials (equal allocation).

Usage:
    # Run with config files
    python tune_manager.py configs/l1_default.yaml configs/spatial_default.yaml

    # Override settings
    python tune_manager.py configs/l1_default.yaml --epochs 5 --trials 20

    # Workers connect as usual
    python tune_worker.py --server http://128.59.145.47:5000
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
from datetime import datetime

import yaml
import optuna
from optuna.samplers import TPESampler
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global state
studies = {}  # study_name -> {study, config, active_trials, completed, failed}
lock = threading.Lock()
manager_config = {}

# Timeouts
HEARTBEAT_TIMEOUT = 300  # 5 minutes without heartbeat = stale
TRIAL_TIMEOUT = 3600  # 1 hour max per trial


def parse_args():
    parser = argparse.ArgumentParser(
        description='Central manager for multiple parallel tuning studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run L1 and spatial studies in parallel
    python tune_manager.py configs/l1_default.yaml configs/spatial_default.yaml

    # Override epochs and trials for all studies
    python tune_manager.py configs/l1_default.yaml configs/spatial_default.yaml --epochs 5 --trials 20

    # Single study
    python tune_manager.py configs/l1_default.yaml
        """)

    parser.add_argument('configs', nargs='+', help='Config files for each study (YAML)')

    # Overrides (apply to all studies)
    parser.add_argument('--trials', type=int, default=None,
                        help='Override trials for all studies')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs for all studies')
    parser.add_argument('--sparsity', type=float, default=None,
                        help='Override sparsity for all studies')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset for all studies')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model for all studies')

    # Server settings
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port')
    parser.add_argument('--log-dir', type=str, default='tune_multi',
                        help='Base directory for logs')

    return parser.parse_args()


def load_config(config_path, overrides):
    """Load a study config from YAML file and apply overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'name' not in config:
        config['name'] = os.path.splitext(os.path.basename(config_path))[0]
    if 'mode' not in config:
        raise ValueError(f"Config {config_path} missing required 'mode' field")

    # Apply overrides
    if overrides.get('trials') is not None:
        config['trials'] = overrides['trials']
    if overrides.get('epochs') is not None:
        config['epochs'] = overrides['epochs']
    if overrides.get('sparsity') is not None:
        config['sparsity'] = overrides['sparsity']
    if overrides.get('dataset') is not None:
        config['dataset'] = overrides['dataset']
    if overrides.get('model') is not None:
        config['model'] = overrides['model']

    # Set defaults
    config.setdefault('trials', 50)
    config.setdefault('epochs', 10)
    config.setdefault('lr', 1e-4)
    config.setdefault('batch_size', 128)
    config.setdefault('sparsity', 90)
    config.setdefault('dataset', 'cifar100')
    config.setdefault('model', 'vit_tiny_patch16_224')
    config.setdefault('log_scale', False)

    # Mode-specific defaults
    if config['mode'] == 'l1':
        config.setdefault('gamma_min', 100)
        config.setdefault('gamma_max', 10000)
    elif config['mode'] == 'spatial':
        config.setdefault('gamma_spatial_min', 10)
        config.setdefault('gamma_spatial_max', 1000)
        config.setdefault('gamma_l1_min', 0)
        config.setdefault('gamma_l1_max', 5000)
        config.setdefault('D_min', 0.1)
        config.setdefault('D_max', 10.0)
        config.setdefault('A', 20.0)
        config.setdefault('B', 20.0)

    return config


def create_study(config, log_dir):
    """Create an Optuna study for a given configuration."""
    study_name = config['name']
    study_dir = os.path.join(log_dir, study_name)
    os.makedirs(study_dir, exist_ok=True)

    # Create subdirectories
    for subdir in ['metrics', 'models', 'results']:
        mode_dir = os.path.join(study_dir, subdir, config['mode'])
        os.makedirs(mode_dir, exist_ok=True)

    storage_path = os.path.join(study_dir, 'study.db')
    storage = f"sqlite:///{storage_path}"

    sampler = TPESampler(seed=42, constant_liar=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        load_if_exists=True,
    )

    # Save config
    config_path = os.path.join(study_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return {
        'study': study,
        'config': config,
        'dir': study_dir,
        'active_trials': {},  # trial_id -> {worker_id, start_time, last_heartbeat}
        'completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'failed': 0,
    }


def suggest_hyperparameters(study_info, trial):
    """Suggest hyperparameters for a trial."""
    config = study_info['config']
    mode = config['mode']
    log_scale = config.get('log_scale', False)

    if mode == 'l1':
        if log_scale:
            gamma = trial.suggest_float('gamma', config['gamma_min'], config['gamma_max'], log=True)
        else:
            gamma = trial.suggest_float('gamma', config['gamma_min'], config['gamma_max'])
        return {'gamma': gamma}

    else:  # spatial
        if log_scale:
            gamma_spatial = trial.suggest_float('gamma_spatial', config['gamma_spatial_min'], config['gamma_spatial_max'], log=True)
            if config['gamma_l1_max'] > 0:
                gamma_l1 = trial.suggest_float('gamma_l1', max(config['gamma_l1_min'], 1e-6), config['gamma_l1_max'], log=True)
            else:
                gamma_l1 = 0
        else:
            gamma_spatial = trial.suggest_float('gamma_spatial', config['gamma_spatial_min'], config['gamma_spatial_max'])
            gamma_l1 = trial.suggest_float('gamma_l1', config['gamma_l1_min'], config['gamma_l1_max'])

        D = trial.suggest_float('D', config['D_min'], config['D_max'], log=log_scale)

        return {
            'gamma_spatial': gamma_spatial,
            'gamma_l1': gamma_l1,
            'D': D,
            'A': config['A'],
            'B': config['B'],
        }


def get_training_config(config):
    """Extract training config from study config."""
    return {
        'epochs': config['epochs'],
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'sparsity': config['sparsity'],
        'dataset': config['dataset'],
        'model': config['model'],
    }


def get_study_with_fewest_trials():
    """Get the study with the fewest completed trials that still needs work."""
    with lock:
        eligible = []
        for name, info in studies.items():
            completed = len([t for t in info['study'].trials if t.state == optuna.trial.TrialState.COMPLETE])
            target = info['config']['trials']
            if completed < target:
                eligible.append((name, completed, info))

        if not eligible:
            return None

        # Sort by completed count, return the one with fewest
        eligible.sort(key=lambda x: x[1])
        return eligible[0][2]


def check_stale_trials():
    """Check for trials that have timed out and mark them as failed."""
    now = time.time()

    with lock:
        for study_name, info in studies.items():
            stale_trials = []

            for trial_id, trial_info in list(info['active_trials'].items()):
                time_since_heartbeat = now - trial_info['last_heartbeat']
                time_since_start = now - trial_info['start_time']

                if time_since_heartbeat > HEARTBEAT_TIMEOUT:
                    print(f"[{study_name}] Trial {trial_id} timed out (no heartbeat for {time_since_heartbeat:.0f}s)")
                    stale_trials.append(trial_id)
                elif time_since_start > TRIAL_TIMEOUT:
                    print(f"[{study_name}] Trial {trial_id} exceeded max duration ({time_since_start:.0f}s)")
                    stale_trials.append(trial_id)

            for trial_id in stale_trials:
                del info['active_trials'][trial_id]
                info['failed'] += 1
                try:
                    info['study'].tell(trial_id, state=optuna.trial.TrialState.FAIL)
                except:
                    pass


def print_status():
    """Print status of all studies."""
    with lock:
        print("\n" + "=" * 70)
        print(f"[Status] {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)

        total_completed = 0
        total_target = 0
        total_running = 0
        all_workers = set()

        for name, info in studies.items():
            completed = len([t for t in info['study'].trials if t.state == optuna.trial.TrialState.COMPLETE])
            target = info['config']['trials']
            running = len(info['active_trials'])
            workers = set(t['worker_id'] for t in info['active_trials'].values())

            total_completed += completed
            total_target += target
            total_running += running
            all_workers.update(workers)

            # Get best accuracy
            best_acc = None
            try:
                if info['study'].best_trial:
                    best_acc = info['study'].best_value
            except ValueError:
                pass

            status = f"  {name}: {completed}/{target}"
            if running > 0:
                status += f" (running: {running})"
            if best_acc is not None:
                status += f" | best: {best_acc:.2f}%"
            print(status)

        print("-" * 70)
        print(f"  Total: {total_completed}/{total_target} | Workers: {len(all_workers)} | Running: {total_running}")
        print("=" * 70 + "\n")


def status_printer():
    """Background thread to print status periodically."""
    while True:
        time.sleep(60)
        check_stale_trials()
        print_status()


@app.route('/get_trial', methods=['GET'])
def get_trial():
    """Worker requests a new trial - assign to study with fewest completed."""
    worker_id = request.args.get('worker_id', 'unknown')

    study_info = get_study_with_fewest_trials()

    if study_info is None:
        return jsonify({'status': 'done', 'message': 'All studies completed'})

    with lock:
        study = study_info['study']
        config = study_info['config']

        try:
            trial = study.ask()
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

        hyperparams = suggest_hyperparameters(study_info, trial)

        # Track active trial
        now = time.time()
        study_info['active_trials'][trial.number] = {
            'worker_id': worker_id,
            'start_time': now,
            'last_heartbeat': now,
        }

        print(f"[{config['name']}] Assigned trial {trial.number} to {worker_id}: {hyperparams}")

        return jsonify({
            'status': 'ok',
            'study_name': config['name'],
            'trial_id': trial.number,
            'mode': config['mode'],
            'hyperparams': hyperparams,
            'training_config': get_training_config(config),
        })


@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Worker sends heartbeat for an active trial."""
    data = request.json
    study_name = data.get('study_name')
    trial_id = data.get('trial_id')
    worker_id = data.get('worker_id')

    with lock:
        if study_name not in studies:
            return jsonify({'status': 'error', 'message': f'Unknown study: {study_name}'}), 400

        info = studies[study_name]
        if trial_id in info['active_trials']:
            info['active_trials'][trial_id]['last_heartbeat'] = time.time()
            return jsonify({'status': 'ok'})
        else:
            return jsonify({'status': 'warning', 'message': 'Trial not found in active trials'})


@app.route('/report', methods=['POST'])
def report():
    """Worker reports trial results."""
    data = request.json
    study_name = data.get('study_name')
    trial_id = data.get('trial_id')
    worker_id = data.get('worker_id')
    success = data.get('success')
    result = data.get('result', {})

    with lock:
        if study_name not in studies:
            return jsonify({'status': 'error', 'message': f'Unknown study: {study_name}'}), 400

        info = studies[study_name]
        config = info['config']
        study = info['study']

        # Remove from active trials
        if trial_id in info['active_trials']:
            del info['active_trials'][trial_id]

        if success:
            accuracy_after = result.get('accuracy_after_pruning', 0)

            try:
                study.tell(trial_id, accuracy_after)

                # Get trial params for naming
                trial = study.trials[trial_id]
                params = trial.params
                mode = config['mode']

                if mode == 'l1':
                    trial_name = f"trial{trial_id}_g{params['gamma']:.1f}"
                else:
                    trial_name = f"trial{trial_id}_gs{params['gamma_spatial']:.1f}_gl1{params.get('gamma_l1', 0):.1f}_D{params['D']:.2f}"

                # Save training log
                if 'training_log' in result:
                    results_dir = os.path.join(info['dir'], 'results', mode)
                    os.makedirs(results_dir, exist_ok=True)
                    with open(os.path.join(results_dir, f'{trial_name}.txt'), 'w') as f:
                        f.write(result['training_log'])

                # Save model
                if 'model_data' in result:
                    models_dir = os.path.join(info['dir'], 'models', mode)
                    os.makedirs(models_dir, exist_ok=True)
                    with open(os.path.join(models_dir, f'{trial_name}.pt'), 'wb') as f:
                        f.write(base64.b64decode(result['model_data']))

                # Save metrics
                metrics_dir = os.path.join(info['dir'], 'metrics', mode)
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_data = {k: v for k, v in result.items() if k not in ('training_log', 'model_data')}
                metrics_data['params'] = params
                with open(os.path.join(metrics_dir, f'{trial_name}.json'), 'w') as f:
                    json.dump(metrics_data, f, indent=2)

                info['completed'] += 1
                print(f"[{study_name}] Trial {trial_id} completed: acc={accuracy_after:.2f}% (worker: {worker_id})")

                # Save study results
                save_study_results(info)

            except Exception as e:
                print(f"[{study_name}] Error recording trial {trial_id}: {e}")
                info['failed'] += 1
                return jsonify({'status': 'error', 'message': str(e)}), 500
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"[{study_name}] Trial {trial_id} failed: {error_msg} (worker: {worker_id})")

            try:
                study.tell(trial_id, state=optuna.trial.TrialState.FAIL)
            except:
                pass

            info['failed'] += 1

        return jsonify({'status': 'ok'})


def save_study_results(info):
    """Save results for a study."""
    study = info['study']
    config = info['config']

    results = {
        'study_name': config['name'],
        'config': config,
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'failed_trials': info['failed'],
        'all_trials': [
            {
                'number': t.number,
                'state': str(t.state),
                'value': t.value,
                'params': t.params,
                'user_attrs': t.user_attrs,
            }
            for t in study.trials
        ],
    }

    try:
        if study.best_trial:
            results['best_trial'] = {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params,
            }
    except ValueError:
        pass

    results_path = os.path.join(info['dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


@app.route('/status', methods=['GET'])
def status():
    """Get status of all studies."""
    with lock:
        status_data = {
            'studies': {},
            'total_completed': 0,
            'total_target': 0,
            'total_running': 0,
            'all_workers': [],
        }

        all_workers = set()

        for name, info in studies.items():
            completed = len([t for t in info['study'].trials if t.state == optuna.trial.TrialState.COMPLETE])
            target = info['config']['trials']
            running = len(info['active_trials'])
            workers = list(set(t['worker_id'] for t in info['active_trials'].values()))

            best_value = None
            best_params = None
            try:
                if info['study'].best_trial:
                    best_value = info['study'].best_value
                    best_params = info['study'].best_params
            except ValueError:
                pass

            status_data['studies'][name] = {
                'mode': info['config']['mode'],
                'completed': completed,
                'target': target,
                'running': running,
                'failed': info['failed'],
                'workers': workers,
                'best_accuracy': best_value,
                'best_params': best_params,
            }

            status_data['total_completed'] += completed
            status_data['total_target'] += target
            status_data['total_running'] += running
            all_workers.update(workers)

        status_data['all_workers'] = list(all_workers)

        return jsonify(status_data)


@app.route('/results', methods=['GET'])
def results():
    """Get results from all studies."""
    with lock:
        all_results = {}

        for name, info in studies.items():
            study = info['study']

            trials_data = []
            for t in study.trials:
                trials_data.append({
                    'number': t.number,
                    'state': str(t.state),
                    'value': t.value,
                    'params': t.params,
                })

            best_trial = None
            try:
                if study.best_trial:
                    best_trial = {
                        'number': study.best_trial.number,
                        'value': study.best_value,
                        'params': study.best_params,
                    }
            except ValueError:
                pass

            all_results[name] = {
                'mode': info['config']['mode'],
                'trials': trials_data,
                'best_trial': best_trial,
            }

        return jsonify(all_results)


def main():
    global studies, manager_config

    args = parse_args()

    # Build overrides dict
    overrides = {
        'trials': args.trials,
        'epochs': args.epochs,
        'sparsity': args.sparsity,
        'dataset': args.dataset,
        'model': args.model,
    }

    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Load configs
    configs = []
    for config_path in args.configs:
        config = load_config(config_path, overrides)
        configs.append(config)

    print(f"\n{'=' * 60}")
    print("MULTI-STUDY HYPERPARAMETER TUNING MANAGER")
    print(f"{'=' * 60}")
    print(f"Log directory: {log_dir}")
    print(f"Studies: {len(configs)}")
    print()

    # Create studies
    for config in configs:
        study_info = create_study(config, log_dir)
        studies[config['name']] = study_info
        print(f"  [{config['name']}] mode={config['mode']}, trials={config['trials']}, epochs={config['epochs']}")
        if config['mode'] == 'l1':
            print(f"      gamma: [{config['gamma_min']}, {config['gamma_max']}]")
        else:
            print(f"      gamma_spatial: [{config['gamma_spatial_min']}, {config['gamma_spatial_max']}]")
            print(f"      D: [{config['D_min']}, {config['D_max']}]")

    # Save manager config
    manager_config = {
        'log_dir': log_dir,
        'timestamp': timestamp,
        'studies': [c['name'] for c in configs],
    }
    with open(os.path.join(log_dir, 'manager_config.json'), 'w') as f:
        json.dump(manager_config, f, indent=2)

    # Start background threads
    checker_thread = threading.Thread(target=status_printer, daemon=True)
    checker_thread.start()

    print()
    print(f"Server starting on {args.host}:{args.port}")
    print(f"Heartbeat timeout: {HEARTBEAT_TIMEOUT}s")
    print(f"Trial timeout: {TRIAL_TIMEOUT}s")
    print()
    print("Endpoints:")
    print("  GET  /get_trial  - Worker requests a trial (auto-assigned to study)")
    print("  POST /heartbeat  - Worker sends heartbeat")
    print("  POST /report     - Worker reports results")
    print("  GET  /status     - Get status of all studies")
    print("  GET  /results    - Get results from all studies")
    print()
    print("Workers will be assigned to the study with fewest completed trials.")
    print(f"{'=' * 60}\n")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
