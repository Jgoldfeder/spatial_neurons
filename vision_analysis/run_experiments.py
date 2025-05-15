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
import os.path
import run_remote

def get_command(cmd,gpu):
    dataset = cmd[0]
    model = cmd[1]
    mode = cmd[2]
    gamma = cmd[3]
    return "bash finetune.sh " + str(gpu) + " " + mode + " " + str(gamma) + " " + dataset + " " + model

def worker(cmd: str, gpu_queue: queue.Queue,test):
    # Grab a free GPU (blocks until one is available)
    run_info = gpu_queue.get()
    print(run_info)
    node = run_info[0]
    gpu_id = run_info[1]
    if node == "local":
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
            gpu_queue.put(run_info)
    else:
        try:
            dataset = cmd[0]
            model = cmd[1]
            mode = cmd[2]
            gamma = cmd[3]
            run_remote.run_remotely(dataset,model,mode,gamma,gpu_id,nodes[node],test=test)
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Error ✖ {cmd} (exit code {e.returncode})")
        finally:
            # Release the GPU back into the pool
            gpu_queue.put(run_info)

def main(commands: list[str], gpus,test=False):
    # Build the GPU queue
    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    # Launch one thread per command; each will block until a GPU is free
    threads = []
    for cmd in commands:
        t = threading.Thread(target=worker, args=(cmd, gpu_queue,test), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all threads (and thus all commands) to finish
    for t in threads:
        t.join()


datasets = ["pets"]
models = ["resnet50"]
modes = {
    "baseline": [0],
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
                path = f"./metrics/{dataset}/{mode}/{mode}:{model}:{gamma}.pkl" 
                if not os.path.exists(path):
                    commands.append([dataset,model,mode,gamma]) 
                else:
                    print("already ran",dataset,model,mode,gamma )


local = {
    "num_gpus": 3,
}
node_0 = {
    "port": "24217",
    "host" : "root@174.94.157.109",
    "key_path": "./.ssh/id_ed",
    "num_gpus": 8,
}

node_1 = {
    "port": "41741",
    "host" : "root@171.7.52.77",
    "key_path": "./.ssh/id_ed",
    "num_gpus": 8,
}

node_2 = {
    "port": "34887",
    "host" : "root@73.97.179.179",
    "key_path": "./.ssh/id_ed",
    "num_gpus": 8,
}

nodes = {
    "local" : local,
    # "0" : node_0,
    # "1" : node_1,
    # "2" : node_2,

}
gpus = []  
for node_id in nodes.keys():
    node = nodes[node_id]
    num_gpus = node["num_gpus"]
    for gpu_id in range(num_gpus):
        gpus.append((node_id,gpu_id))
        print(node_id,gpu_id)

main(commands, gpus,test=True)