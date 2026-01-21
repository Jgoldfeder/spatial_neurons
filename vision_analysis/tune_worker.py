#!/usr/bin/env python3
"""
Worker for distributed Bayesian hyperparameter optimization.

Usage:
    # Connect to server and start processing trials
    python tune_worker.py --server http://localhost:5000

    # Specify worker ID (auto-generated if not provided)
    python tune_worker.py --server http://localhost:5000 --worker-id gpu0

    # Specify GPU device
    python tune_worker.py --server http://localhost:5000 --device cuda:0
"""

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import json

import requests


def parse_args():
    parser = argparse.ArgumentParser(description='Worker for distributed hyperparameter optimization')

    parser.add_argument('--server', type=str, default='http://128.59.145.47:5000',
                        help='Server URL (default: http://128.59.145.47:5000)')
    parser.add_argument('--worker-id', type=str, default=None,
                        help='Worker identifier (auto-generated if not provided)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cpu). Auto-detected if not specified.')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use. Spawns one worker per GPU. Auto-detects if not specified.')
    parser.add_argument('--heartbeat-interval', type=int, default=60,
                        help='Seconds between heartbeats')
    parser.add_argument('--retry-delay', type=int, default=30,
                        help='Seconds to wait before retrying after error')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum retries for network errors')

    return parser.parse_args()


class HeartbeatThread(threading.Thread):
    """Background thread that sends heartbeats to server."""

    def __init__(self, server_url, worker_id, interval=60):
        super().__init__(daemon=True)
        self.server_url = server_url
        self.worker_id = worker_id
        self.interval = interval
        self.trial_id = None
        self.study_name = None  # For multi-study manager
        self.progress = {}
        self.running = True

    def set_trial(self, trial_id, study_name=None):
        self.trial_id = trial_id
        self.study_name = study_name
        self.progress = {}

    def update_progress(self, info):
        self.progress = info

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if self.trial_id is not None:
                try:
                    payload = {
                        'trial_id': self.trial_id,
                        'worker_id': self.worker_id,
                        'progress': self.progress,
                    }
                    # Include study_name if set (for multi-study manager)
                    if self.study_name:
                        payload['study_name'] = self.study_name

                    response = requests.post(
                        f"{self.server_url}/heartbeat",
                        json=payload,
                        timeout=10
                    )
                    if response.status_code != 200:
                        print(f"Heartbeat warning: {response.json()}")
                except Exception as e:
                    print(f"Heartbeat error: {e}")

            time.sleep(self.interval)


def get_trial(server_url, worker_id, max_retries=3):
    """Get a trial from the server with retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{server_url}/get_trial",
                params={'worker_id': worker_id},
                timeout=30
            )
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting trial (attempt {attempt + 1}): {e}")
                time.sleep(5)
            else:
                raise


def report_result(server_url, worker_id, trial_id, success, result, study_name=None, max_retries=3):
    """Report trial result to server with retries."""
    for attempt in range(max_retries):
        try:
            payload = {
                'trial_id': trial_id,
                'worker_id': worker_id,
                'success': success,
                'result': result,
            }
            # Include study_name if set (for multi-study manager)
            if study_name:
                payload['study_name'] = study_name

            response = requests.post(
                f"{server_url}/report",
                json=payload,
                timeout=30
            )
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error reporting result (attempt {attempt + 1}): {e}")
                time.sleep(5)
            else:
                raise


def run_trial(mode, hyperparams, training_config, device):
    """Run a single trial by calling train.py and return results."""

    # Build command line arguments for train.py
    cmd = [
        sys.executable, 'train.py',
        '--mode', mode,
        '--epochs', str(training_config['epochs']),
        '--lr', str(training_config['lr']),
        '--batch-size', str(training_config['batch_size']),
        '--sparsity', str(training_config['sparsity']),
        '--dataset', training_config['dataset'],
        '--model', training_config['model'],
    ]

    if mode == 'l1':
        cmd.extend(['--gamma', str(hyperparams['gamma'])])
    else:  # spatial
        cmd.extend([
            '--gamma-spatial', str(hyperparams['gamma_spatial']),
            '--gamma-l1', str(hyperparams.get('gamma_l1', 0)),
            '--D', str(hyperparams['D']),
            '--A', str(hyperparams.get('A', 20.0)),
            '--B', str(hyperparams.get('B', 20.0)),
        ])

    # Set device via environment variable
    env = os.environ.copy()
    if device:
        env['CUDA_VISIBLE_DEVICES'] = device.replace('cuda:', '')

    print(f"Running: {' '.join(cmd)}")

    # Run train.py and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    # Print output for logging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"train.py failed with code {result.returncode}: {result.stderr}")

    # Find and read the metrics file
    # train.py prints "Logs saved to: <path>" at the end
    logs_path = None
    for line in result.stdout.split('\n'):
        if 'Logs saved to:' in line:
            logs_path = line.split('Logs saved to:')[1].strip()
            break

    if not logs_path:
        raise RuntimeError("Could not find logs path in train.py output")

    metrics_path = os.path.join(logs_path, 'metrics', 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Read full training logs
    training_log_path = os.path.join(logs_path, 'results', 'training.txt')
    with open(training_log_path, 'r') as f:
        training_log = f.read()

    # Read model file as base64
    import base64
    model_path = os.path.join(logs_path, 'models', 'model.pt')
    with open(model_path, 'rb') as f:
        model_data = base64.b64encode(f.read()).decode('utf-8')

    return {
        'accuracy_before_pruning': metrics['evaluation']['accuracy_before_pruning'],
        'accuracy_after_pruning': metrics['evaluation']['accuracy_after_pruning'],
        'actual_sparsity': metrics['evaluation']['actual_sparsity'],
        'threshold': metrics['evaluation']['threshold'],
        'training_log': training_log,
        'model_data': model_data,
    }


def download_datasets():
    """Download datasets once before starting trials, with file lock to prevent concurrent downloads."""
    import torchvision
    import filelock

    lock_path = './data/.download_lock'
    os.makedirs('./data', exist_ok=True)

    lock = filelock.FileLock(lock_path, timeout=600)
    with lock:
        print("Downloading datasets...")
        torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        print("Datasets ready.")


def run_worker(server, worker_id, device, heartbeat_interval, retry_delay, max_retries):
    """Run a single worker loop for one GPU."""
    print(f"Worker ID: {worker_id}")
    print(f"Server: {server}")
    print(f"Device: {device}")
    print()

    # Download datasets once before starting
    download_datasets()

    # Start heartbeat thread
    heartbeat_thread = HeartbeatThread(server, worker_id, heartbeat_interval)
    heartbeat_thread.start()

    # Main loop
    while True:
        try:
            # Get trial from server
            print(f"[{worker_id}] Requesting trial from server...")
            trial_info = get_trial(server, worker_id, max_retries)

            if trial_info.get('status') == 'done':
                print(f"[{worker_id}] All trials completed! Exiting.")
                break

            if trial_info.get('status') != 'ok':
                print(f"[{worker_id}] Unexpected response: {trial_info}")
                time.sleep(retry_delay)
                continue

            trial_id = trial_info['trial_id']
            hyperparams = trial_info['hyperparams']
            training_config = trial_info['training_config']
            mode = trial_info['mode']
            study_name = trial_info.get('study_name')  # For multi-study manager

            print(f"\n[{worker_id}] {'='*50}")
            if study_name:
                print(f"[{worker_id}] Study: {study_name}")
            print(f"[{worker_id}] Starting trial {trial_id}")
            print(f"[{worker_id}] Mode: {mode}")
            print(f"[{worker_id}] Hyperparams: {hyperparams}")
            print(f"[{worker_id}] {'='*50}\n")

            # Set trial for heartbeat (include study_name for multi-study manager)
            heartbeat_thread.set_trial(trial_id, study_name)

            # Run trial
            try:
                result = run_trial(mode, hyperparams, training_config, device)

                print(f"\n[{worker_id}] Trial {trial_id} completed:")
                print(f"[{worker_id}]   Accuracy before pruning: {result['accuracy_before_pruning']:.2f}%")
                print(f"[{worker_id}]   Accuracy after pruning: {result['accuracy_after_pruning']:.2f}%")
                print(f"[{worker_id}]   Actual sparsity: {result['actual_sparsity']:.2f}%")

                # Report success
                report_result(server, worker_id, trial_id, True, result, study_name, max_retries)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"\n[{worker_id}] Trial {trial_id} failed: {error_msg}")

                # Report failure
                report_result(server, worker_id, trial_id, False, {'error': error_msg}, study_name, max_retries)

            # Clear trial from heartbeat
            heartbeat_thread.set_trial(None)

        except KeyboardInterrupt:
            print(f"\n[{worker_id}] Interrupted by user. Exiting.")
            break

        except Exception as e:
            print(f"[{worker_id}] Error in main loop: {e}")
            traceback.print_exc()
            time.sleep(retry_delay)

    heartbeat_thread.stop()


def get_num_gpus():
    """Detect number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 0


def main():
    args = parse_args()

    # Determine number of GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = get_num_gpus()
        if num_gpus == 0:
            num_gpus = 1  # CPU fallback

    hostname = socket.gethostname()

    # If only one GPU or device specified, run directly
    if num_gpus == 1 or args.device is not None:
        worker_id = args.worker_id or f"{hostname}_gpu0"
        device = args.device or "cuda:0"
        run_worker(args.server, worker_id, device, args.heartbeat_interval, args.retry_delay, args.max_retries)
        return

    # Multiple GPUs - spawn processes
    print(f"Detected {num_gpus} GPUs. Spawning {num_gpus} worker processes...")
    import multiprocessing
    processes = []

    for gpu_id in range(num_gpus):
        worker_id = args.worker_id or f"{hostname}_gpu{gpu_id}"
        if args.worker_id:
            worker_id = f"{args.worker_id}_gpu{gpu_id}"
        device = f"cuda:{gpu_id}"

        p = multiprocessing.Process(
            target=run_worker,
            args=(args.server, worker_id, device, args.heartbeat_interval, args.retry_delay, args.max_retries)
        )
        p.start()
        processes.append(p)
        print(f"Started worker {worker_id} on {device}")

    # Wait for all processes
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating all workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
