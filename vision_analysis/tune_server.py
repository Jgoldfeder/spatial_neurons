#!/usr/bin/env python3
"""
Central server for distributed Bayesian hyperparameter optimization.

Usage:
    # Start server for L1 tuning
    python tune_server.py --mode l1 --gamma-min 100 --gamma-max 10000 \
        --sparsity 90 --trials 50 --epochs 10

    # Start server for spatial tuning
    python tune_server.py --mode spatial \
        --gamma-spatial-min 10 --gamma-spatial-max 1000 \
        --gamma-l1-min 0 --gamma-l1-max 5000 \
        --D-min 0.1 --D-max 10.0 \
        --sparsity 90 --trials 50 --epochs 10

    # Resume existing study
    python tune_server.py --resume <study_name>
"""

import argparse
import json
import os
import pickle
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify

import optuna
from optuna.samplers import TPESampler

app = Flask(__name__)

# Global state
study = None
study_config = None
active_trials = {}  # trial_id -> {worker_id, start_time, last_heartbeat}
completed_count = 0
failed_count = 0
lock = threading.Lock()

# Configuration
HEARTBEAT_TIMEOUT = 300  # 5 minutes without heartbeat = assume dead
TRIAL_TIMEOUT = 3600  # 1 hour max per trial


def parse_args():
    parser = argparse.ArgumentParser(description='Central server for distributed hyperparameter optimization')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['l1', 'spatial'],
                        help='Regularization mode: l1 or spatial')

    # L1 hyperparameter ranges
    parser.add_argument('--gamma-min', type=float, default=100)
    parser.add_argument('--gamma-max', type=float, default=10000)

    # Spatial hyperparameter ranges
    parser.add_argument('--gamma-spatial-min', type=float, default=10)
    parser.add_argument('--gamma-spatial-max', type=float, default=1000)
    parser.add_argument('--gamma-l1-min', type=float, default=0)
    parser.add_argument('--gamma-l1-max', type=float, default=5000)
    parser.add_argument('--D-min', type=float, default=0.1)
    parser.add_argument('--D-max', type=float, default=10.0)
    parser.add_argument('--A', type=float, default=20.0)
    parser.add_argument('--B', type=float, default=20.0)

    # Search settings
    parser.add_argument('--log-scale', action='store_true')
    parser.add_argument('--trials', type=int, default=50)

    # Training settings (passed to workers)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sparsity', type=float, default=90)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')

    # Server settings
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--log-dir', type=str, default='tune_distributed')
    parser.add_argument('--name', type=str, default=None)

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume existing study by name')

    # Timeout settings
    parser.add_argument('--heartbeat-timeout', type=int, default=300,
                        help='Seconds without heartbeat before assuming worker dead')
    parser.add_argument('--trial-timeout', type=int, default=3600,
                        help='Maximum seconds per trial')

    return parser.parse_args()


def create_study(args):
    """Create or load Optuna study with SQLite storage."""
    global study, study_config

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_name = args.name or f"{args.mode}_distributed_{args.dataset}_{timestamp}"

    log_dir = os.path.join(args.log_dir, study_name)
    os.makedirs(log_dir, exist_ok=True)

    # Use SQLite for persistence
    storage_path = os.path.join(log_dir, 'study.db')
    storage = f"sqlite:///{storage_path}"

    # Save config
    study_config = {
        'study_name': study_name,
        'mode': args.mode,
        'log_dir': log_dir,
        'storage_path': storage_path,
        'target_trials': args.trials,
        'hyperparameter_ranges': {
            'gamma_min': args.gamma_min,
            'gamma_max': args.gamma_max,
            'gamma_spatial_min': args.gamma_spatial_min,
            'gamma_spatial_max': args.gamma_spatial_max,
            'gamma_l1_min': args.gamma_l1_min,
            'gamma_l1_max': args.gamma_l1_max,
            'D_min': args.D_min,
            'D_max': args.D_max,
            'A': args.A,
            'B': args.B,
            'log_scale': args.log_scale,
        },
        'training_config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'sparsity': args.sparsity,
            'dataset': args.dataset,
            'model': args.model,
        },
        'timeouts': {
            'heartbeat': args.heartbeat_timeout,
            'trial': args.trial_timeout,
        }
    }

    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(study_config, f, indent=2)

    # Create study
    sampler = TPESampler(seed=42, constant_liar=True)  # constant_liar helps with parallel trials

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        load_if_exists=True,
    )

    print(f"Study: {study_name}")
    print(f"Storage: {storage_path}")
    print(f"Config: {config_path}")

    return study, study_config


def load_study(study_name, log_dir='tune_distributed'):
    """Load existing study."""
    global study, study_config

    study_dir = os.path.join(log_dir, study_name)
    config_path = os.path.join(study_dir, 'config.json')
    storage_path = os.path.join(study_dir, 'study.db')

    if not os.path.exists(config_path):
        raise ValueError(f"Study config not found: {config_path}")

    with open(config_path, 'r') as f:
        study_config = json.load(f)

    storage = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    print(f"Resumed study: {study_name}")
    print(f"Completed trials: {len(study.trials)}")

    return study, study_config


def suggest_hyperparameters(trial):
    """Suggest hyperparameters for a trial."""
    config = study_config
    ranges = config['hyperparameter_ranges']
    mode = config['mode']
    log_scale = ranges['log_scale']

    if mode == 'l1':
        if log_scale:
            gamma = trial.suggest_float('gamma', ranges['gamma_min'], ranges['gamma_max'], log=True)
        else:
            gamma = trial.suggest_float('gamma', ranges['gamma_min'], ranges['gamma_max'])
        return {'gamma': gamma}
    else:  # spatial
        if log_scale:
            gamma_spatial = trial.suggest_float('gamma_spatial', ranges['gamma_spatial_min'], ranges['gamma_spatial_max'], log=True)
            if ranges['gamma_l1_max'] > 0:
                gamma_l1 = trial.suggest_float('gamma_l1', max(ranges['gamma_l1_min'], 1e-6), ranges['gamma_l1_max'], log=True)
            else:
                gamma_l1 = 0
            D = trial.suggest_float('D', ranges['D_min'], ranges['D_max'], log=True)
        else:
            gamma_spatial = trial.suggest_float('gamma_spatial', ranges['gamma_spatial_min'], ranges['gamma_spatial_max'])
            gamma_l1 = trial.suggest_float('gamma_l1', ranges['gamma_l1_min'], ranges['gamma_l1_max'])
            D = trial.suggest_float('D', ranges['D_min'], ranges['D_max'])

        return {
            'gamma_spatial': gamma_spatial,
            'gamma_l1': gamma_l1,
            'D': D,
            'A': ranges['A'],
            'B': ranges['B'],
        }


def check_stale_trials():
    """Check for trials that have timed out and mark them as failed."""
    global active_trials, failed_count

    now = time.time()
    stale_trials = []

    with lock:
        for trial_id, info in list(active_trials.items()):
            time_since_heartbeat = now - info['last_heartbeat']
            time_since_start = now - info['start_time']

            if time_since_heartbeat > HEARTBEAT_TIMEOUT:
                print(f"Trial {trial_id} timed out (no heartbeat for {time_since_heartbeat:.0f}s)")
                stale_trials.append(trial_id)
            elif time_since_start > TRIAL_TIMEOUT:
                print(f"Trial {trial_id} exceeded max duration ({time_since_start:.0f}s)")
                stale_trials.append(trial_id)

        for trial_id in stale_trials:
            del active_trials[trial_id]
            failed_count += 1
            # Mark as failed in Optuna
            try:
                study.tell(trial_id, state=optuna.trial.TrialState.FAIL)
            except:
                pass


def stale_trial_checker():
    """Background thread to check for stale trials."""
    while True:
        time.sleep(60)  # Check every minute
        check_stale_trials()


@app.route('/get_trial', methods=['GET'])
def get_trial():
    """Worker requests a new trial to run."""
    global active_trials, completed_count

    worker_id = request.args.get('worker_id', 'unknown')

    with lock:
        # Check if we've completed enough trials
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed >= study_config['target_trials']:
            return jsonify({'status': 'done', 'message': 'All trials completed'})

        # Create new trial
        try:
            trial = study.ask()
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(trial)

        # Track active trial
        active_trials[trial.number] = {
            'worker_id': worker_id,
            'start_time': time.time(),
            'last_heartbeat': time.time(),
            'hyperparams': hyperparams,
        }

        print(f"Assigned trial {trial.number} to worker {worker_id}: {hyperparams}")

        return jsonify({
            'status': 'ok',
            'trial_id': trial.number,
            'hyperparams': hyperparams,
            'training_config': study_config['training_config'],
            'mode': study_config['mode'],
        })


@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Worker sends heartbeat to indicate it's still alive."""
    data = request.get_json(silent=True) or {}
    trial_id = data.get('trial_id')
    worker_id = data.get('worker_id', 'unknown')
    progress = data.get('progress', {})

    with lock:
        if trial_id in active_trials:
            active_trials[trial_id]['last_heartbeat'] = time.time()
            active_trials[trial_id]['progress'] = progress
            return jsonify({'status': 'ok'})
        else:
            return jsonify({'status': 'error', 'message': 'Trial not found or expired'}), 404


@app.route('/report', methods=['POST'])
def report():
    """Worker reports trial results."""
    global active_trials, completed_count, failed_count

    data = request.get_json(silent=True) or {}
    trial_id = data.get('trial_id')
    success = data.get('success', False)
    result = data.get('result', {})
    worker_id = data.get('worker_id', 'unknown')

    with lock:
        if trial_id not in active_trials:
            return jsonify({'status': 'error', 'message': 'Trial not found or expired'}), 404

        trial_info = active_trials.pop(trial_id)

        if success:
            # Report success to Optuna
            accuracy_after = result.get('accuracy_after_pruning', 0)

            try:
                study.tell(trial_id, accuracy_after)

                # Store additional attributes (except training_log and model_data which we save separately)
                trial = study.trials[trial_id]
                for key, value in result.items():
                    if key not in ('training_log', 'model_data'):
                        trial.set_user_attr(key, value)

                # Build trial name from hyperparameters
                params = trial.params
                mode = study_config['mode']
                if mode == 'l1':
                    trial_name = f"trial{trial_id}_g{params['gamma']:.1f}"
                else:  # spatial
                    trial_name = f"trial{trial_id}_gs{params['gamma_spatial']:.1f}_gl1{params.get('gamma_l1', 0):.1f}_D{params['D']:.2f}"

                # Save training log to file (results/<mode>/<trial>.txt)
                if 'training_log' in result:
                    results_dir = os.path.join(study_config['log_dir'], 'results', mode)
                    os.makedirs(results_dir, exist_ok=True)
                    log_path = os.path.join(results_dir, f'{trial_name}.txt')
                    with open(log_path, 'w') as f:
                        f.write(result['training_log'])

                # Save model file (models/<mode>/<trial>.pt)
                if 'model_data' in result:
                    import base64
                    models_dir = os.path.join(study_config['log_dir'], 'models', mode)
                    os.makedirs(models_dir, exist_ok=True)
                    model_path = os.path.join(models_dir, f'{trial_name}.pt')
                    with open(model_path, 'wb') as f:
                        f.write(base64.b64decode(result['model_data']))

                # Save metrics (metrics/<mode>/<trial>.json)
                metrics_dir = os.path.join(study_config['log_dir'], 'metrics', mode)
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_path = os.path.join(metrics_dir, f'{trial_name}.json')
                metrics_data = {k: v for k, v in result.items() if k not in ('training_log', 'model_data')}
                metrics_data['params'] = params
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)

                completed_count += 1
                print(f"Trial {trial_id} completed: acc={accuracy_after:.2f}% (worker: {worker_id})")

                # Save intermediate results
                save_results()

            except Exception as e:
                print(f"Error recording trial {trial_id}: {e}")
                failed_count += 1
                return jsonify({'status': 'error', 'message': str(e)}), 500
        else:
            # Report failure
            error_msg = result.get('error', 'Unknown error')
            print(f"Trial {trial_id} failed: {error_msg} (worker: {worker_id})")

            try:
                study.tell(trial_id, state=optuna.trial.TrialState.FAIL)
            except:
                pass

            failed_count += 1

        return jsonify({'status': 'ok'})


@app.route('/status', methods=['GET'])
def status():
    """Get current optimization status."""
    with lock:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        running = len(active_trials)

        best_value = None
        best_params = None
        try:
            if study.best_trial:
                best_value = study.best_value
                best_params = study.best_params
        except ValueError:
            # No completed trials yet
            pass

        return jsonify({
            'study_name': study_config['study_name'],
            'target_trials': study_config['target_trials'],
            'completed': completed,
            'failed': failed,
            'running': running,
            'active_workers': list(set(t['worker_id'] for t in active_trials.values())),
            'best_accuracy': best_value,
            'best_params': best_params,
        })


@app.route('/results', methods=['GET'])
def results():
    """Get all trial results."""
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'state': str(trial.state),
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
        })

    try:
        best_trial_num = study.best_trial.number if study.best_trial else None
        best_value = study.best_value if study.best_trial else None
        best_params = study.best_params if study.best_trial else None
    except ValueError:
        best_trial_num = None
        best_value = None
        best_params = None

    return jsonify({
        'study_name': study_config['study_name'],
        'trials': trials_data,
        'best_trial': best_trial_num,
        'best_value': best_value,
        'best_params': best_params,
    })


def save_results():
    """Save current results to disk."""
    results = {
        'study_name': study_config['study_name'],
        'config': study_config,
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'best_trial': None,
        'all_trials': [],
    }

    if study.best_trial:
        results['best_trial'] = {
            'number': study.best_trial.number,
            'value': study.best_value,
            'params': study.best_params,
            'user_attrs': study.best_trial.user_attrs,
        }

    for trial in study.trials:
        results['all_trials'].append({
            'number': trial.number,
            'state': str(trial.state),
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
        })

    results_path = os.path.join(study_config['log_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    global study, study_config, HEARTBEAT_TIMEOUT, TRIAL_TIMEOUT

    args = parse_args()

    HEARTBEAT_TIMEOUT = args.heartbeat_timeout
    TRIAL_TIMEOUT = args.trial_timeout

    if args.resume:
        study, study_config = load_study(args.resume, args.log_dir)
    else:
        if not args.mode:
            print("Error: --mode is required when starting a new study")
            return
        study, study_config = create_study(args)

    print(f"\nServer starting on {args.host}:{args.port}")
    print(f"Mode: {study_config['mode']}")
    print(f"Target trials: {study_config['target_trials']}")
    print(f"Heartbeat timeout: {HEARTBEAT_TIMEOUT}s")
    print(f"Trial timeout: {TRIAL_TIMEOUT}s")
    print()
    print("Endpoints:")
    print(f"  GET  /get_trial  - Worker requests a trial")
    print(f"  POST /heartbeat  - Worker sends heartbeat")
    print(f"  POST /report     - Worker reports results")
    print(f"  GET  /status     - Get optimization status")
    print(f"  GET  /results    - Get all results")
    print()

    # Start background thread for stale trial checking
    checker_thread = threading.Thread(target=stale_trial_checker, daemon=True)
    checker_thread.start()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
