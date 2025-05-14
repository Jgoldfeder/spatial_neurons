import util
import glob
import timm
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pickle
import os
model = timm.create_model('vit_base_patch16_224', pretrained=False)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 100)

cifar100=True

threshold=0.001
baseline_results=[]
spatial_results=[]
L1_results=[]
circle_results=[]
spatial_l2 = []
spatial_swap = []
spatial_learnable=[]
cluster400_results=[]
cluster40_results=[]
cluster10_results=[]

reg_types = ["baseline","spatial-swap","spatial","spatial-learn","L1"]
for reg_type in reg_types:
    file_list =  glob.glob("./big_vit_models/"+reg_type+"/*")
    file_list.sort(key=lambda x: int(x.split("_")[3].split(".")[0]))
    for file_name in file_list:
        print(file_name)
        
        name = file_name.replace(".","")

        if os.path.exists("./metrics_max/"+name):
            with open("./metrics_max/"+name+'.pkl', 'rb') as f:
                result = pickle.load(f)
                print(result)
            print("File loaded successfully:")
        else:
            result= {}
            for p in [100,90,80,70,60,50,40,30,20,10,5,3,2,1]:
                state_dict = torch.load(file_name)
                model.load_state_dict(state_dict)
                threshold = util.compute_pruning_threshold_cpu(model,p)
                initial_acc, percent_small, final_acc = util.evaluate_vit_pruning(model, threshold=threshold,cifar100=cifar100)
                dead_neuron_counts, total_dead, total_neurons = util.count_dead_neurons(state_dict,threshold)        
    

                result[p] = {
                    "initial_acc" : initial_acc,
                    "percent_below_t" : percent_small,
                    "final_acc" : final_acc,
                    "dead_neurons": total_dead,
                    "percent_dead_neurons": total_dead/total_neurons,
                }



            os.makedirs("./metrics_max/"+name, exist_ok=True)

            with open("./metrics_max/"+name+'.pkl', 'wb') as f:
                pickle.dump(result, f)
            print(result)
        if reg_type == "baseline":
            baseline_results.append(result)
        if reg_type == "spatial":
            spatial_results.append(result)
        if reg_type == "L1":
            L1_results.append(result)
        if reg_type=="spatial-circle":
            circle_results.append(result)
        if reg_type=="spatiall2":
            spatial_l2.append(result)
        if reg_type=="spatial-swap":
            spatial_swap.append(result)
        if reg_type=="spatial-learn":
            spatial_learnable.append(result)
        if reg_type=="cluster10":
            cluster10_results.append(result)
        if reg_type=="cluster40":
            cluster40_results.append(result)
        if reg_type=="cluster400":
            cluster400_results.append(result)
