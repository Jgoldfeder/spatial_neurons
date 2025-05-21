import torch
import numpy as np
import timm
import spatial_wrapper_learnable
import spatial_wrapper_swap
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

# Set device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = sys.argv[1]
gamma = int(sys.argv[2])
dataset_name = sys.argv[3]
model_name = sys.argv[4]

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
    "spatial-swap",
    "spatial-learn",
    "spatial-both",    
    "spatial-circle",
    "cluster",
    "uniform",
    "gaussian",
    "spatial-squared",
]

cluster = -1
if mode.startswith('cluster'):
    cluster = int(mode.split("r")[1])
    mode = "cluster"

if mode not in modes:
    raise ValueError("Mode "+mode+" not recognized!")


# spatial parameters
A = 20.0
B = 20.0
D = 1.0 

if mode in ["spatial","spatial-swap","spatial-circle","cluster","uniform","gaussian","spatial-squared"]:
    distribution="spatial"
    if mode in ["uniform","gaussian"]:
        distribution = mode
    use_circle = mode in ["spatial-circle"]       
    model = spatial_wrapper_swap.SpatialNet(model,A, B, D,circle=use_circle,cluster=cluster,distribution=distribution)
if mode in ['spatial-learn','spatial-both']:
    model = spatial_wrapper_learnable.SpatialNet(model,A, B, D)

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

if mode in ['spatial-learn','spatial-both']:
    # we want the positions to have a higher learning rate than the weights
    optimizer = Adam([
        {'params': model.model.parameters(), 'lr': 1e-4},  
        {'params': model.value_distance_matrices.parameters(), 'lr': 1e-2},
        {'params': model.linear_distance_matrices.parameters(), 'lr': 1e-2},    
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
    if mode in ['spatial-learn','spatial-both']:
        # make sure neurons do not collapse or explode
        print(model.get_stats())
    if mode in ["spatial-swap",'spatial-both']:
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

        if mode in ["L1"]:
            l1_norm = sum(p.abs().mean() for p in model.parameters())/len([p for p in model.parameters()])
            loss+=l1_norm*gamma
        if mode in ["spatial","spatial-swap","spatial-learn","spatial-circle","cluster",'spatial-both',"uniform","gaussian","spatial-squared"]:
            use_quadratic = mode in ["spatial-squared"]
            loss += model.get_cost(quadratic=use_quadratic)*gamma

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
    
if mode in ["spatial","spatial-swap","spatial-learn","spatial-circle","cluster",'spatial-both',"uniform","gaussian","spatial-squared"]:
    # extract the model from the wrapper
    model=model.model

state_dict=model.state_dict()

results = {}
# compute fixed threshold metrics
for threshold in [0.01,0.001,0.0001]:
    model.load_state_dict(state_dict)
    initial_acc, percent_small, final_acc = util.evaluate_pruning(model, threshold=threshold,dataset_name=dataset_name)
    dead_neuron_counts, total_dead, total_neurons = util.count_dead_neurons(state_dict,threshold)   
    # this next version also includes input neurons, and considers both incoming weights or outoing weights     
    dead_neuron_indices, unique_dead, unique_total_neurons = util.count_unique_dead_neurons(state_dict,threshold)       
    shift_accuracy = util.evaluate_on_synthetic_shifts(model,dataset_name=dataset_name)
    modularity = util.model_modularity(model, threshold=threshold)

    results[threshold] = {
        "initial_acc" : initial_acc,
        "percent_below_t" : percent_small,
        "final_acc" : final_acc,
        "dead_neurons": total_dead,
        "percent_dead_neurons": total_dead/total_neurons,   
        "unique_dead_neurons": unique_dead,
        "percent_unique_dead_neurons": unique_dead/unique_total_neurons,  
        "shift_accuracy": shift_accuracy,
        "modularity" : modularity,
    }

# compute fixed sparsity max metrics

for p in [100,90,80,70,60,50,40,30,20,10,5,3,2,1]:
    model.load_state_dict(state_dict)
    try:
        threshold = util.compute_pruning_threshold_cpu(model,p)
    except:
        continue
    initial_acc, percent_small, final_acc = util.evaluate_pruning(model, threshold=threshold,dataset_name=dataset_name)
    dead_neuron_counts, total_dead, total_neurons = util.count_dead_neurons(state_dict,threshold)   
    # this next version also includes input neurons, and considers both incoming weights or outoing weights     
    dead_neuron_indices, unique_dead, unique_total_neurons = util.count_unique_dead_neurons(state_dict,threshold)       
    shift_accuracy = util.evaluate_on_synthetic_shifts(model,dataset_name=dataset_name)
    modularity = util.model_modularity(model, threshold=threshold)

    results[p] = {
        "initial_acc" : initial_acc,
        "percent_below_t" : percent_small,
        "final_acc" : final_acc,
        "dead_neurons": total_dead,
        "percent_dead_neurons": total_dead/total_neurons,   
        "unique_dead_neurons": unique_dead,
        "percent_unique_dead_neurons": unique_dead/unique_total_neurons,  
        "shift_accuracy": shift_accuracy,
        "modularity" : modularity,
    }

if mode == "cluster":
    mode = mode + str(cluster)

path = dataset_name +"/" + mode + "/" 
file_name = mode + ":" +model_name+":"+str(gamma)

os.makedirs("./metrics/"+path, exist_ok=True)
os.makedirs("./models/"+path, exist_ok=True)

torch.save(state_dict,"./models/"+path + file_name +".pt")
with open("./metrics/"+path+ file_name + '.pkl', 'wb') as f:
    pickle.dump(results, f)

print(results)