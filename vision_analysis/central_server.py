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

commands = []

# modes = {
#     "baseline": [0],
#     "L1" : [50, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000, 120000, 300000],
#     "spatial": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
#     "spatial-swap": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
#     "spatial-learn": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
#     "spatial-both": [40,140, 325, 650, 1300, 2600, 4000, 8000, 16000, 32000, 64000],
# }



# datasets = ['cifar100','cifar10',"pets","tiny_imagenet",'svhn','birds','caltech101','DTD']
# models = ["vit_tiny_patch16_224","resnet50"]
# modes = {
#     "baseline": [0],
#     "L1" : [50, 200, 500, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
#     "spatial": [20,40,80,120,162,325,650,1300,2600,5000],
#     "spatial-swap": [20,40,80,120,162,325,650,1300,2600,5000],
#     "spatial-learn":[20,40,80,120,162,325,650,1300,2600,5000],
#     "spatial-both": [20,40,80,120,162,325,650,1300,2600,5000],
# }



# for dataset in datasets:
#     for model in models:
#         for mode, gammas in modes.items():
#             for gamma in gammas:
#                 path = f"./metrics/{dataset}/{mode}/{mode}:{model}:{gamma}.pkl"  
#                 if not os.path.exists(path):
#                     command = {
#                         "dataset" : dataset,
#                         "mode" : mode,
#                         "model" : model,
#                         "gamma" : gamma,
#                     }
#                     commands.append(command) 
#                 else:
#                     print("already ran", dataset, model, mode, gamma)

# datasets = ['cifar100']
# models = ['vit_base_patch16_224','resnet101','efficientnet_b0','vgg19','visformer_small','swin_base_patch4_window7_224','mobilenetv3_small_100','densenet121']
# for dataset in datasets:
#     for model in models:
#         for mode, gammas in modes.items():
#             for gamma in gammas:
#                 path = f"./metrics/{dataset}/{mode}/{mode}:{model}:{gamma}.pkl"  
#                 if not os.path.exists(path):
#                     command = {
#                         "dataset" : dataset,
#                         "mode" : mode,
#                         "model" : model,
#                         "gamma" : gamma,
#                     }
#                     commands.append(command) 
#                 else:
#                     print("already ran", dataset, model, mode, gamma)




modes = {
    # "baseline": [0],
    "L1" : [50, 200, 500, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
    "spatial": [20,40,80,120,162,325,650,1300,2600,5000],
    #"spatial-group4": [20,40,80,120,162,325,650,1300,2600,5000],
    #"L1-group4" : [50, 200, 500],#, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
    #"group4" : [50, 200, 500],#, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
    "block-4" : [50, 200, 500, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
    "block-16" : [50, 200, 500, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
    
    "spatial-swap": [20,40,80,120,162,325,650,1300,2600,5000],
    #"spatial-learn":[20,40,80,120,162,325,650,1300,2600,5000],
    
    # "spatial-both": [20,40,80,120,162,325,650,1300,2600,5000],
    
    
    # "uniform": [20,40,80,120,162,325,650,1300,2600,5000,10000,20000,40000],
    # "gaussian": [20,40,80,120,162,325,650,1300,2600,5000,10000,20000,40000],
    # "cluster4": [20,40,80,120,162,325,650,1300,2600,5000,10000,20000,40000],
    # "cluster40": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "cluster400": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-polar": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-squared": [1,2,3,5,10,15,20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-circle": [1,5,10,20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-polar": [20,40,80,120,162,325,650,1300,2600,5000,10000,20000],
    # "spatial-learn-euclidean": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-ndim3": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-ndim4": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-ndim5": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-ndim10": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-ndim100": [20,40,80,120,162,325,650,1300,2600,5000,10000],
    # "spatial-learn-squared": [1,2,3,5,10,15,20,40,80,120,162,325,650,1300,2600,5000,10000],

}
datasets = ['cifar100']
models = ["vit_tiny_patch16_224"]

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
                    print(command)
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

    if not data or not isinstance(data, dict):
        return jsonify(error="expected JSON object"), 400

    # Check if metrics are empty (worker crashed)
    # Valid metrics should have keys like '0.01', '0.001', '100', '90', etc.
    metric_keys = [k for k in data.keys() if k not in ('mode', 'model_name', 'gamma', 'dataset_name')]
    if not metric_keys:
        app.logger.warning(f"Received empty metrics from worker, not saving: {data.get('mode')} {data.get('model_name')} {data.get('gamma')}")
        return jsonify(error="empty metrics, not saved"), 400

    mode = data['mode']
    model_name = data['model_name']
    gamma = data['gamma']
    dataset_name = data['dataset_name']

    path = dataset_name + "/" + mode + "/"
    file_name = mode + ":" + model_name + ":" + str(gamma)

    os.makedirs("./metrics/" + path, exist_ok=True)
    os.makedirs("./models/" + path, exist_ok=True)
    with open("./metrics/" + path + file_name + '.pkl', 'wb') as f:
        pickle.dump(data, f)

    app.logger.info(f"Received report: {data}")
    return jsonify(ok=True), 200

if __name__ == '__main__':
    # Listen on all interfaces so workers (and only those with the key) can connect
    app.run(host='0.0.0.0', port=5000)
