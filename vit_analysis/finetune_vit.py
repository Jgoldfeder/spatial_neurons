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

# Set device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = sys.argv[1]
if mode not in ["baseline","L1","spatial","spatial-swap","spatial-learn","spatial-circle","L2","spatial-L2","cluster10","cluster40","cluster400"]:
    raise ValueError("Mode "+mode+" not recognized!")

gamma = int(sys.argv[2])



# Define normalization constants for CIFAR100 (approximate values)
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

# Data augmentation and normalization for training, and normalization for testing
train_transform = transforms.Compose([
    transforms.Resize(256),                # Resize to a bit larger than final crop
    transforms.RandomCrop(224),            # Random crop to 224x224
    transforms.RandomHorizontalFlip(),     # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=test_transform)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

# Load a pretrained model and modify the final fully connected layer for 100 classes
#model = timm.create_model('vit_base_patch16_224', pretrained=True)
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 100)

# spatial parameters
A = 20.0
B = 20.0
D = 1.0 
if mode in ["spatial","spatial-swap","spatial-L2","spatial-circle","cluster10","cluster40","cluster400"]:
    use_circle = mode in ["spatial-circle"]
    cluster=-1
    if mode in ["cluster10"]:
        cluster=10
    if mode in ["cluster40"]:
        cluster=40    
    if mode in ["cluster400"]:
        cluster=400
    model = spatial_wrapper_swap.SpatialNet(model,A, B, D,circle=use_circle,cluster=cluster)
if mode in ['spatial-learn']:
    model = spatial_wrapper_learnable.SpatialNet(model,A, B, D)

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

if mode in ['spatial-learn']:
    # we want the positions to have a higher learning rate than the weights
    optimizer = Adam([
        {'params': model.model.parameters(), 'lr': 1e-4},  
        {'params': model.value_distance_matrices.parameters(), 'lr': 1e-2},
        {'params': model.linear_distance_matrices.parameters(), 'lr': 1e-2},    
    ])

# Training loop for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
    if mode in ['spatial-learn']:
        # make sure neurons do not collapse or explode
        print(model.get_stats())
    if mode in ["spatial-swap"]:
        # optimize via swapping
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
        if mode in ["spatial","spatial-swap","spatial-L2","spatial-learn","spatial-circle","cluster10","cluster40","cluster400"]:
            use_quadratic = mode in ["spatial-L2"]
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
    
if mode in ["spatial","spatial-swap","spatial-L2","spatial-learn","spatial-circle","cluster10","cluster40","cluster400"]:
    # extract the model from the wrapper
    model=model.model

state_dict=model.state_dict()
torch.save(state_dict,"./models/"+mode+"/"+mode +"_"+str(gamma)+".pt")

threshold = 1e-2  # Set your desired threshold
dead_counts, total_dead, total_neurons = util.count_dead_neurons(state_dict, threshold)

print(f"Total neurons examined: {total_neurons}")
print(f"Total dead neurons: {total_dead}")
print("Dead neuron counts per layer:")
for layer, count in dead_counts.items():
    print(f"  {layer}: {count}")


# Analyze distributions from state dict
layer_weights = {}
all_weights = []
def is_regularized_weight(name):
    """Check if this parameter had L1 regularization applied during training"""
    if 'bias' in name or 'ln' in name or 'wte' in name or 'wpe' in name:
        return False
    if 'weight' not in name:
        return False
    return True
 
for name, param in state_dict.items():
    if is_regularized_weight(name) and param.dim() > 1:
        weights = param.detach().cpu().numpy().flatten()
        layer_weights[name] = weights
        all_weights.extend(weights)

all_weights = np.array(all_weights)

# Print statistics
print("\nOverall Weight Statistics (regularized weights only):")
print(f"Std: {np.std(all_weights):.6f}")

# Print percentiles
util.print_percentiles(all_weights)