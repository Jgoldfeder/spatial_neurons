import torch
import numpy as np
import timm
import spatial_wrapper_learnable
import spatial_wrapper_swap
import spatial_wrapper_learnable_ndim
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import sys
import util
import classification_util
import tensorflow_datasets as tfds
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pickle, os
import copy
from block_sparsity import block_sparsity_after_reorder

# Set device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = sys.argv[1]
gamma = int(sys.argv[2])
dataset_name = sys.argv[3]
model_name = sys.argv[4]
og_mode=mode
batchsize=128
if model_name in ['visformer_small','swin_base_patch4_window7_224']:
    batchsize=64
if model_name == "resnet50" and dataset_name == "DTD":
    batchsize=64 
if dataset_name == "caltech101":
    batchsize=32
train_loader, test_loader, num_classes = classification_util.get_data_loaders(dataset_name,batch_size=batchsize)
model = classification_util.get_model(model_name,num_classes,pretrained=True)

modes = [
    "baseline",
    "L1",
    "spatial",
    "spatiall1",
    "spatial-swap",
    "spatial-learn",
    "spatial-learn-polar",
    "spatial-both",
    "spatial-circle",
    "spatial-circle-swap",
    "cluster",
    "uniform",
    "gaussian",
    "spatial-squared",
    "spatial-learn-euclidean",
    "spatial-learn-ndim",
    "spatial-learn-squared",
    "block",
    "group",           # group lasso only
    "spatial-group",   # spatial + group lasso
    "L1-group",        # L1 + group lasso
]

# Parse group size for group lasso modes
group_size = -1
if mode.startswith('L1-group'):
    group_size = int(mode.split("group")[1])
    mode = "L1-group"
elif mode.startswith('spatial-group'):
    group_size = int(mode.split("group")[1])
    mode = "spatial-group"
elif mode.startswith('group') and mode != "gaussian":  # avoid matching "gaussian"
    group_size = int(mode.split("group")[1])
    mode = "group"

cluster = -1
block_group = -1
block_binary = False
if mode.startswith('block'):
    parts = mode.split("-")
    block_group = int(parts[1])
    # Check for binary mode: block-32-b
    if len(parts) > 2 and parts[2] == 'b':
        block_binary = True
    mode = "block"

if mode.startswith('cluster'):
    cluster = int(mode.split("r")[1])
    mode = "cluster"

ndim = -1
if mode.startswith('spatial-learn-ndim'):
    ndim = int(mode.split("m")[1])
    mode = "spatial-learn-ndim"

# spatial parameters
A = 20.0
B = 20.0
D = 1.0 

if mode.startswith('spatial-circle-swap'):
    if len(mode.split("-")) > 3:
        D = float(mode.split("-")[3])
    mode = "spatial-circle-swap"
    print("D=",D)
elif mode.startswith('spatial-swap'):
    if len(mode.split("-")) > 2:
        A = float(mode.split("-")[2])
        B = float(mode.split("-")[2])
    mode = "spatial-swap"
    print("A,B=",A)

if mode.startswith('spatial-learn'):
    if len(mode.split("-")) > 2:
        D = float(mode.split("-")[2])
    mode = "spatial-learn"
    print("D=",A)

if mode not in modes:
    raise ValueError("Mode "+mode+" not recognized!")



# A = 0.001
# B = 0.001
# D = 1.0 
if mode in ["spatial","spatial-swap","spatial-circle-swap","spatial-circle","cluster","uniform","gaussian","spatial-squared","spatiall1","block","spatial-group"]:
    distribution="spatial"
    if mode in ["uniform","gaussian","block"]:
        distribution = mode
    use_circle = mode in ["spatial-circle","spatial-circle-swap"]
    model = spatial_wrapper_swap.SpatialNet(model,A, B, D,circle=use_circle,cluster=cluster,block_group=block_group,block_binary=block_binary,distribution=distribution)
if mode in ['spatial-learn','spatial-both',"spatial-learn-polar" ,"spatial-learn-euclidean","spatial-learn-squared"]:
    use_polar = mode in ["spatial-learn-polar"]
    use_euclidean = mode in ["spatial-learn-euclidean"]
    model = spatial_wrapper_learnable.SpatialNet(model,A, B, D,use_polar=use_polar,euclidean=use_euclidean)
if mode in ["spatial-learn-ndim"]:
    model = spatial_wrapper_learnable_ndim.SpatialNet(model,A, B, D,n_dims=ndim)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

if mode in ['spatial-learn','spatial-both',"spatial-learn-polar" ,"spatial-learn-euclidean","spatial-learn-ndim","spatial-learn-squared"]:
    # we want the positions to have a higher learning rate than the weights
    optimizer = Adam([
        {'params': model.model.parameters(), 'lr': 1e-4},  
        {'params': model.value_distance_matrices.parameters(), 'lr': 1e-2},
        {'params': model.linear_distance_matrices.parameters(), 'lr': 1e-2},    
        {'params': model.conv_distance_matrices.parameters(), 'lr': 1e-2},    

    ])

# Training loop
# cifar100 has 391 batches per epoch, and we train for 10 for a total of 3910 batches
# we want to keep this roughly constant across datasets. But also have 10 be the minimum, and 50 max
num_batches = len(train_loader)
num_epochs = min(50,max(10,4000//num_batches))

def get_epochs(num_epochs: int) -> list[int]:
    # We need 10 points: i = 0..9, step = (num_epochs-1)/9
    step = (num_epochs - 1) / 9
    # round() to get the nearest integer each time
    return [int(round(i * step)) for i in range(10)]
swap_epochs = get_epochs(num_epochs)

for epoch in range(num_epochs):
    if mode in ['spatial-learn','spatial-both',"spatial-learn-polar" ,"spatial-learn-euclidean","spatial-learn-ndim","spatial-learn-squared"]:
        # make sure neurons do not collapse or explode
        print(model.get_stats())
    if mode in ["spatial-swap",'spatial-both',"spatial-circle-swap","block"]:
        # optimize via swapping
        if epoch in swap_epochs:
            model.optimize()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if mode in ["spatiall1","L1","L1-group"]:
            #wrong version: l1_norm = sum(p.abs().mean() for p in model.parameters())/len([p for p in model.parameters()])
            #corret but includes all params l1_norm = sum(p.abs().sum() for p in model.parameters()) / sum(p.numel() for p in model.parameters())
            l1_norm = util.l1_linear_and_conv(model)
            loss+=l1_norm*gamma
        if mode in ["group","spatial-group","L1-group"]:
            group_lasso = util.group_lasso_linear_and_conv(model, group_size)
            loss += group_lasso*gamma
        if mode in ["spatial-circle-swap","spatiall1","spatial","spatial-swap","spatial-learn","spatial-learn-polar" ,"spatial-learn-euclidean","spatial-circle","cluster",'spatial-both',"uniform","gaussian","spatial-squared","spatial-learn-ndim","spatial-learn-squared","block","spatial-group"]:
            use_quadratic = mode in ["spatial-squared","spatial-learn-squared"]
            factor = 1
            if mode in ['spatiall1']:
                factor = 100
            loss += model.get_cost(quadratic=use_quadratic)*gamma/factor

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / total
    train_acc = 100. * correct / total

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    
    test_loss /= total_test
    test_acc = 100. * correct_test / total_test

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
if mode in ["spatiall1","spatial","spatial-swap","spatial-circle-swap","spatial-learn","spatial-learn-polar" ,"spatial-learn-euclidean","spatial-circle","cluster",'spatial-both',"uniform","gaussian","spatial-squared","spatial-learn-ndim","spatial-learn-squared","block"]:
    # extract the model from the wrapper
    model=model.model

state_dict=copy.deepcopy(model.state_dict())

results = {}
# compute fixed threshold metrics
for threshold in [0.01,0.001,0.0001]:
    model.load_state_dict(state_dict)
    initial_acc, percent_small, final_acc = util.evaluate_pruning(model, threshold=threshold,dataset_name=dataset_name)
    dead_neuron_counts, total_dead, total_neurons = util.count_dead_neurons(state_dict,threshold)   
    # this next version also includes input neurons, and considers both incoming weights or outoing weights     
    dead_neuron_indices, unique_dead, unique_total_neurons = util.count_unique_dead_neurons(state_dict,threshold)
    modularity = util.model_modularity(model, threshold=threshold)
    block_stats = block_sparsity_after_reorder(model, block_size=16, sparsity_threshold=threshold)

    results[threshold] = {
        "initial_acc" : initial_acc,
        "percent_below_t" : percent_small,
        "final_acc" : final_acc,
        "dead_neurons": total_dead,
        "percent_dead_neurons": total_dead/total_neurons,
        "unique_dead_neurons": unique_dead,
        "percent_unique_dead_neurons": unique_dead/unique_total_neurons,
        "modularity" : modularity,
        "block_sparsity_reordered": block_stats['block_sparsity_ratio'],
    }

# compute fixed sparsity max metrics

for p in [100,90,80,70,60,50,40,30,20,10,5,3,2,1]:
    print(p)
    model.load_state_dict(state_dict)
    try:
        threshold = util.compute_pruning_threshold_cpu(model,p)
    except:
        continue
    initial_acc, percent_small, final_acc = util.evaluate_pruning(model, threshold=threshold,dataset_name=dataset_name)
    dead_neuron_counts, total_dead, total_neurons = util.count_dead_neurons(state_dict,threshold)   
    # this next version also includes input neurons, and considers both incoming weights or outoing weights     
    dead_neuron_indices, unique_dead, unique_total_neurons = util.count_unique_dead_neurons(state_dict,threshold)       
    # shift_accuracy = util.evaluate_on_synthetic_shifts(model,dataset_name=dataset_name)
    modularity = util.model_modularity(model, threshold=threshold)
    block_stats = block_sparsity_after_reorder(model, block_size=16, sparsity_threshold=threshold)
    model.load_state_dict(state_dict)

    results[p] = {
        "initial_acc" : initial_acc,
        "percent_below_t" : percent_small,
        "final_acc" : final_acc,
        "dead_neurons": total_dead,
        "percent_dead_neurons": total_dead/total_neurons,
        "unique_dead_neurons": unique_dead,
        "percent_unique_dead_neurons": unique_dead/unique_total_neurons,
        "modularity" : modularity,
        "block_sparsity_reordered": block_stats['block_sparsity_ratio'],
    }

if mode == "cluster":
    mode = mode + str(cluster)
if mode == 'spatial-learn-ndim':
    mode = mode + str(ndim)
if mode == 'spatial-swap' and A != 20:
    mode = og_mode

if mode == 'spatial-learn' and D != 1:
    mode = og_mode

if mode == 'block':
    mode = og_mode

# Restore original mode name for group lasso modes to preserve block size
if mode in ["group", "spatial-group", "L1-group"]:
    mode = og_mode

path = dataset_name +"/" + mode + "/"
file_name = mode + ":" +model_name+":"+str(gamma)

os.makedirs("./metrics/"+path, exist_ok=True)
os.makedirs("./models/"+path, exist_ok=True)

torch.save(state_dict,"./models/"+path + file_name +".pt")
with open("./metrics/"+path+ file_name + '.pkl', 'wb') as f:
    pickle.dump(results, f)

print(results)