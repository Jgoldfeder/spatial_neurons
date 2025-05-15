import threading
import time
import requests
import threading
import queue
import subprocess
import os
import os.path
import pickle
# 1) Configuration (keep your existing definitions)
API_KEY = "super-secret-key-123"
SERVER  = "http://127.0.0.1:5000"
SERVER  = "http://cml1.seas.columbia.edu:5000"
HEADERS = {"X-API-KEY": API_KEY}

def get_task():
    resp = requests.get(f"{SERVER}/get_task", headers=HEADERS)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json().get("task")

def send_report(report_dict):
    resp = requests.post(f"{SERVER}/report", json=report_dict, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def thread_main(thread_id):
    """
    Main loop for each worker thread.
    """
    while True:
        task = get_task()
        if task is None:
            # no work: back off for a bit
            time.sleep(2)
            continue

        print(f"[Thread {thread_id}] Fetched task: {task}")
        command = "bash finetune.sh " + str(thread_id) + " " + task['mode'] + " " + str(task['gamma']) + " " + task['dataset'] + " " + task['model']
        
        env = os.environ.copy()
        # Run the command
        subprocess.run(command, shell=True, check=True, env=env)
        print(f"[Thread {thread_id}] Completed task: {task}")

        dataset = task['dataset']
        mode = task['mode']
        model = task['model']
        gamma = task['gamma']
        
        path = dataset +"/" + mode + "/" 
        file_name = mode + ":" +model+":"+str(gamma) 
        name = "./metrics/"+path+ file_name +'.pkl'
        result={}


        if os.path.exists(name):
            with open(name, 'rb') as f:
                result = pickle.load(f)
        result['mode'] = mode
        result['model_name'] = model
        result['gamma'] = gamma
        result['dataset_name'] = dataset

        resp = send_report(result)
        print(f"[Thread {thread_id}] Report response: {resp}")

if __name__ == "__main__":
    threads = []
    NUM_THREADS = 3

    # Spawn threads 0 through 7
    for i in range(NUM_THREADS):
        t = threading.Thread(target=thread_main, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        print(f"Started thread {i}")

    # Keep main alive while threads run
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Shutting down workers…")
