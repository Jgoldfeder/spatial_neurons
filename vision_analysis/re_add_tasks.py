import requests
import os
import pickle
import time

API_KEY = "super-secret-key-123"
SERVER  = "http://cml1.seas.columbia.edu:5000"

def enqueue_tasks(tasks, server_url=SERVER, api_key=API_KEY):
    """
    Given a list of task dicts, POST each to the Flask server's /add_task endpoint.

    Args:
        tasks (List[dict]): List of task dictionaries, e.g.
            [{"dataset": "cifar100", "mode": "baseline", "model": "resnet50", "gamma": 0}, ...]
        server_url (str): Base URL of your Flask server. Defaults to localhost.
        api_key (str): Your API key for authentication.

    Raises:
        RuntimeError: If adding any task fails (status code ≠ 201).
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key
    }
    endpoint = f"{server_url.rstrip('/')}/add_task"

    for task in tasks:
        resp = requests.post(endpoint, json={"task": task}, headers=headers)
        if resp.status_code != 201:
            raise RuntimeError(f"Failed to enqueue task {task!r}: {resp.status_code} {resp.text.strip()}")


while True:
    time.sleep(2.5)
    commands = []

    # datasets = ['cifar100','cifar10',"pets","tiny_imagenet",'svhn','birds','DTD'] # 'caltech101'
    # models = ["vit_tiny_patch16_224","resnet50"]
    # modes = {
    #     "baseline": [0],
    #     "L1" : [50, 200, 500, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
    #     "spatial": [20,40,80,120,162,325,650,1300,2600,5000],
    #     "spatial-swap": [20,40,80,120,162,325,650,1300,2600,5000],
    #     "spatial-learn":[20,40,80,120,162,325,650,1300,2600,5000],
    #     "spatial-both": [20,40,80,120,162,325,650,1300,2600,5000],
    # }


    modes = {
        "baseline": [0],
        "L1" : [50, 200, 500, 1000, 2000, 4000, 5000, 10000, 20000, 40000],
        "spatial": [20,40,80,120,162,325,650,1300,2600,5000],
        "spatial-swap": [20,40,80,120,162,325,650,1300,2600,5000],
        "spatial-learn":[20,40,80,120,162,325,650,1300,2600,5000],
        "spatial-both": [20,40,80,120,162,325,650,1300,2600,5000],
        "uniform": [20,40,80,120,162,325,650,1300,2600,5000],
        "gaussian": [20,40,80,120,162,325,650,1300,2600,5000],
        "cluster4": [20,40,80,120,162,325,650,1300,2600,5000],
        "cluster40": [20,40,80,120,162,325,650,1300,2600,5000],
        "cluster400": [20,40,80,120,162,325,650,1300,2600,5000],
    }
    datasets = ['cifar100']
    models = ["vit_tiny_patch16_224"]

    for dataset in datasets:
        for model in models:
            for mode, gammas in modes.items():
                for gamma in gammas:
                    path = f"./metrics/{dataset}/{mode}/{mode}:{model}:{gamma}.pkl"  
                    command = {
                        "dataset" : dataset,
                        "mode" : mode,
                        "model" : model,
                        "gamma" : gamma,
                    }
                    commands.append(command) 
                    

    datasets = ['cifar100']  # removed 'visformer_small',
    models = ['vit_base_patch16_224','resnet101','efficientnet_b0','vgg19','swin_base_patch4_window7_224','mobilenetv3_small_100','densenet121']
    for dataset in datasets:
        for model in models:
            for mode, gammas in modes.items():
                for gamma in gammas:
                    path = f"./metrics/{dataset}/{mode}/{mode}:{model}:{gamma}.pkl"  
                    command = {
                        "dataset" : dataset,
                        "mode" : mode,
                        "model" : model,
                        "gamma" : gamma,
                    }
                    commands.append(command) 
    
    error=0
    total=0
    commands_to_add=[]
    for command in commands:
        dataset=command['dataset']
        mode=command['mode']
        model=command['model']
        gamma=command['gamma']

        path = dataset +"/" + mode + "/" 
        file_name = mode + ":" +model+":"+str(gamma) 
        name = "./metrics/"+path+ file_name +'.pkl'
        if os.path.exists(name):
            total+=1
            with open(name, 'rb') as f:
                result = pickle.load(f)
            if len(result.keys()) <=4:
                print(command)
                error+=1
                commands_to_add.append(command)
                os.remove(name)

    enqueue_tasks(commands_to_add)  
    print(len(commands),total,error)