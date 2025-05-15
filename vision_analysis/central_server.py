# server.py

import os
from flask import Flask, request, jsonify, abort
from queue import Queue
from threading import Lock
import pickle
app = Flask(__name__)

# Load your secret key from environment (fallback provided for example)
VALID_API_KEY = os.environ.get("API_KEY", "super-secret-key-123")

# Thread-safe in-memory task queue
task_queue = Queue()
lock = Lock()


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
                    command = {
                        "dataset" : dataset,
                        "mode" : mode,
                        "model" : model,
                        "gamma" : gamma,
                    }
                    commands.append(command) 
                else:
                    print("already ran", dataset, model, mode, gamma)

for command in commands:
    task_queue.put(command)
print(commands)
@app.before_request
def require_api_key():
    """
    Enforce API key on every request.
    Checks header 'X-API-KEY' first, then query param 'key'.
    """
    key = request.headers.get("X-API-KEY") or request.args.get("key")
    if key != VALID_API_KEY:
        abort(401, description="Unauthorized: invalid or missing API key")

@app.route('/get_task', methods=['GET'])
def get_task():
    """Worker pulls one task; returns 204 if queue is empty."""
    with lock:
        if task_queue.empty():
            return jsonify({"task": None}), 204
        task = task_queue.get()
    return jsonify({"task": task}), 200

@app.route('/add_task', methods=['POST'])
def add_task():
    """
    Optional: allow adding new tasks.
    Expects JSON body: {"task": {...}}
    """
    data = request.get_json(silent=True) or {}
    task = data.get("task")
    if not task:
        return jsonify(error="no task provided"), 400
    with lock:
        task_queue.put(task)
    return jsonify(ok=True), 201

@app.route('/report', methods=['POST'])
def report():
    """
    Worker sends a report dict as JSON; returns 200 on success.
    """
    data = request.get_json(silent=True)
    mode = data['mode']
    model_name =  data['model_name']
    gamma =  data['gamma']
    dataset_name =  data['dataset_name']

    path = dataset_name +"/" + mode + "/" 
    file_name = mode + ":" +model_name+":"+str(gamma)

    # os.makedirs("./metrics/"+path, exist_ok=True)
    # os.makedirs("./models/"+path, exist_ok=True)
    # with open("./metrics/"+path+ file_name + '.pkl', 'wb') as f:
    #     pickle.dump(data, f)


    if not data or not isinstance(data, dict):
        return jsonify(error="expected JSON object"), 400
    # Process the report (e.g., log, store in DB)
    app.logger.info(f"Received report: {data}")
    return jsonify(ok=True), 200

if __name__ == '__main__':
    # Listen on all interfaces so workers (and only those with the key) can connect
    app.run(host='0.0.0.0', port=5000)
