#!/usr/bin/env python3
"""
run_on_gpus.py

Run multiple shell commands in parallel on a fixed set of GPUs,
never oversubscribing any device.
"""

import threading
import queue
import subprocess
import os

def get_command(cmd,gpu):
    dataset = cmd[0]
    model = cmd[1]
    mode = cmd[2]
    gamma = cmd[3]
    return "bash finetune.sh " + str(gpu) + " " + mode + " " + str(gamma) + " " + dataset + " " + model

def worker(cmd: str, gpu_queue: queue.Queue):
    # Grab a free GPU (blocks until one is available)
    gpu_id = gpu_queue.get()
    try:
        print(f"[GPU {gpu_id}] Starting ➜ {cmd}")
        env = os.environ.copy()

        # Run the command
        subprocess.run(get_command(cmd,gpu_id), shell=True, check=True, env=env)

        print(f"[GPU {gpu_id}] Finished ✔ {cmd}")
    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] Error ✖ {cmd} (exit code {e.returncode})")
    finally:
        # Release the GPU back into the pool
        gpu_queue.put(gpu_id)


def main(commands: list[str], gpus: list[int]):
    # Build the GPU queue
    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    # Launch one thread per command; each will block until a GPU is free
    threads = []
    for cmd in commands:
        t = threading.Thread(target=worker, args=(cmd, gpu_queue), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all threads (and thus all commands) to finish
    for t in threads:
        t.join()

datasets = ["pets"]
models = ["resnet50"]
modes = {
    "L1" : [50, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000, 120000, 300000],
    "spatial": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
    "spatial-swap": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
    "spatial-learn": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
    "spatial-both": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
}

commands = []
for dataset in datasets:
    for model in models:
        for mode, gammas in modes.items():
            for gamma in gammas:
                commands.append([dataset,model,mode,gamma]) 


gpus = [0, 1, 2]  

main(commands, gpus)
