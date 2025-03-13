import torch
import numpy as np
import timm
from spatial_wrapper_vit import SpatialNet
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import sys

# Set device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = sys.argv[1]
gamma = int(sys.argv[2])

def count_dead_neurons(state_dict, threshold=1e-3):
    """
    Counts dead neurons in 2D weight matrices.
    
    A dead neuron is defined as one for which all incoming weights (i.e. all elements in its row)
    have an absolute value below the given threshold.
    
    Parameters:
        state_dict (dict): The state dictionary loaded from the model.
        threshold (float): The threshold below which a weight is considered "dead".
        
    Returns:
        dead_neuron_counts (dict): Dictionary with layer names as keys and number of dead neurons as values.
        total_dead (int): Total count of dead neurons.
        total_neurons (int): Total count of neurons examined.
    """
    dead_neuron_counts = {}
    total_neurons = 0
    total_dead = 0
    
    for name, param in state_dict.items():
        # Check if the parameter is a weight matrix (2D tensor) from a linear layer
        if "weight" in name and param.ndim == 2:
            # Detach and convert to a NumPy array for easier analysis
            weight = param.detach().cpu().numpy()
            out_features = weight.shape[0]  # each row corresponds to one neuron
            layer_dead = 0
            for i in range(out_features):
                # Check if every incoming weight for this neuron is below the threshold
                if (abs(weight[i]) < threshold).all():
                    layer_dead += 1
            dead_neuron_counts[name] = layer_dead
            total_neurons += out_features
            total_dead += layer_dead
    
    return dead_neuron_counts, total_dead, total_neurons

def print_percent_below(arr, t):
    """Prints the percentage of values in 'arr' that are below the threshold t."""
    percent = np.mean(arr < t) * 100
    print(f"Percentage of values below {t}: {percent:.2f}%")

def print_percentiles(weights, model_name):
    """Print percentiles of absolute weights"""
    abs_weights = np.abs(weights)
    percentiles = range(10, 101, 10)
    print(f"\nPercentiles for {model_name} (absolute values):")
    for p in percentiles:
        value = np.percentile(abs_weights, p)
        print(f"P{p}: {value:.6f}")
    print_percent_below(abs_weights,0.001)

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

# Load a pretrained ResNet50 and modify the final fully connected layer for 100 classes
#model = timm.create_model('vit_base_patch16_224', pretrained=True)
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 100)

if "spatial" in mode:
    A = 20.0
    B = 20.0
    D = 1.0 
    # Create a spatially wrapped ResNet50 (for example, with 100 output classes).
    model = SpatialNet(model,A, B, D)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop for 50 epochs
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    #batch_count=0
    for inputs, labels in train_loader:
        #print(batch_count,len(train_loader))
        #batch_count+=1
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if "L1" in mode:
            l1_norm = sum(p.abs().mean() for p in model.parameters())/len([p for p in model.parameters()])
            loss+=l1_norm*gamma
            #print(l1_norm*1000)
        if mode=="spatial":
            loss += model.get_cost()*gamma
            #print( model.get_cost()*1300)
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
if "spatial" in mode:
    model=model.model
state_dict=model.state_dict()
if "spatial" in mode:
    torch.save(state_dict,"./models/spatial/"+mode +"_"+str(gamma)+".pt")
if "baseline" in mode:
    torch.save(state_dict,"./models/baseline/"+mode +"_"+str(gamma)+".pt")
if "L1" in mode:
    torch.save(state_dict,"./models/L1/"+mode +"_"+str(gamma)+".pt")

threshold = 1e-2  # Set your desired threshold
dead_counts, total_dead, total_neurons = count_dead_neurons(state_dict, threshold)

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
print_percentiles(all_weights, mode)
