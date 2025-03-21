import matplotlib.pyplot as plt
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import torchvision.transforms.functional as TF
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
    
def visualize_vit_architecture(model, neuron_count=192, connection_threshold=0.1, include_identity=False):
    """
    Visualize a Vision Transformer architecture as a schematic diagram.
    
    For each transformer block, the function visualizes:
      - (Optionally) An "input" layer, which is an identity mapping (default: off).
      - The "Attn" layer from the attention projection (block.attn.proj.weight).
      - The "MLP" layer from the second linear layer in the MLP (block.mlp.fc2.weight).
    
    Then, the final classification head is visualized (model.head.weight).
    
    Each layer is drawn as a vertical column of neurons (circles), and weight connections
    between neurons are drawn as lines (only if |weight| ≥ connection_threshold).
    
    Parameters:
      - model: The ViT model (e.g. from timm).
      - neuron_count: Number of neurons to draw per layer.
      - connection_threshold: Minimum absolute weight value to draw a connection.
      - include_identity: If True, include the input identity layers for each transformer block.
                         Default is False.
    """
    layers = []  # List of tuples: (label, weight_matrix)
    
    # Extract transformer blocks.
    if hasattr(model, 'blocks'):
        blocks = model.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        blocks = model.transformer.blocks
    else:
        raise ValueError("Cannot extract transformer blocks from model.")
    
    # For each block, add layers.
    for i, block in enumerate(blocks):
        if include_identity:
            # Add an identity layer to represent the block's input.
            identity = np.eye(neuron_count, dtype=np.float32)
            layers.append((f"B{i+1} Input", identity))
        # Layer for attention output.
        if hasattr(block.attn, 'proj'):
            attn_weight = block.attn.proj.weight.detach().cpu().numpy()  # expected shape: (hidden_dim, hidden_dim)
        else:
            attn_weight = np.eye(neuron_count, dtype=np.float32)
        layers.append((f"B{i+1} Attn", attn_weight))
        
        # Layer for MLP output.
        if hasattr(block.mlp, 'fc2'):
            mlp_weight = block.mlp.fc2.weight.detach().cpu().numpy()  # expected shape: (hidden_dim, hidden_dim)
        else:
            mlp_weight = np.eye(neuron_count, dtype=np.float32)
        layers.append((f"B{i+1} MLP", mlp_weight))
    
    # Add the final classification layer.
    head_weight = model.head.weight.detach().cpu().numpy()  # shape: (num_classes, hidden_dim)
    num_classes, hd = head_weight.shape
    # Adjust so the final layer is drawn on the same vertical scale.
    if num_classes < neuron_count:
        padded = np.zeros((neuron_count, hd), dtype=np.float32)
        padded[:num_classes, :] = head_weight
        final_weight = padded
    else:
        final_weight = head_weight[:neuron_count, :]
    layers.append(("Final", final_weight))
    
    # Total number of drawing layers is one more than the number of weight matrices.
    num_draw_layers = len(layers) + 1
    
    # Compute x positions for each layer.
    x_spacing = 1.0
    x_positions = np.arange(num_draw_layers) * x_spacing
    
    # Compute y positions for neurons in each layer.
    y_positions = np.linspace(0, 1, neuron_count)
    
    # Store neuron coordinates for each layer.
    neuron_coords = []
    for x in x_positions:
        coords = [(x, y) for y in y_positions]
        neuron_coords.append(coords)
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=(num_draw_layers * 1.5, 8))
    
    # Draw neurons as circles.
    neuron_radius = 0.02
    for layer in neuron_coords:
        for (x, y) in layer:
            circle = plt.Circle((x, y), neuron_radius, color='black', zorder=2)
            ax.add_artist(circle)
    
    # Draw connections between consecutive layers.
    for i, (label, weight_matrix) in enumerate(layers):
        W = weight_matrix  # Expected shape: (neuron_count, neuron_count)
        for j in range(neuron_count):      # neuron index in layer i (input)
            for k in range(neuron_count):  # neuron index in layer i+1 (output)
                w = W[k, j]
                if abs(w) < connection_threshold:
                    continue
                color = 'blue' if w > 0 else 'red'
                lw = 0.5 + 2.5 * abs(w)
                x0, y0 = neuron_coords[i][j]
                x1, y1 = neuron_coords[i+1][k]
                ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.5, zorder=1)
    
    # Set up layer labels.
    layer_labels = []
    # First layer is always "Input" (the activations or patch embeddings).
    layer_labels.append("Input")
    for label, _ in layers:
        layer_labels.append(label)
    
    for x, lab in zip(x_positions, layer_labels):
        ax.text(x, 1.02, lab, rotation=90, va='bottom', ha='center', fontsize=8)
    
    ax.set_xlim(-0.5, x_positions[-1] + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title("ViT Architecture Visualization\n(Neurons as Circles and Weight Connections)")
    plt.show()

def plot_lists(data_lists, title="My Chart", xlabel="X-axis", ylabel="Y-axis", labels=None):
    """
    Plots any number of lists. Each dataset should be a list of (y, x) tuples.

    Parameters:
        *data_lists: Variable number of lists, each a list of (y, x) tuples.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        labels (list): Optional list of labels for each dataset.
    """
    plt.figure(figsize=(8, 6))
    
    # If no labels provided, generate default ones.
    if labels is None:
        labels = [f"L{i+1}" for i in range(len(data_lists))]
    
    for i, data in enumerate(data_lists):
        if data:
            # Unzip the tuples into y and x coordinates.
            y, x = zip(*data)
        else:
            x, y = [], []
        plt.plot(x, y, marker='o', linestyle='-', label=labels[i] if i < len(labels) else None)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_two_lists(list1, list2, title="My Chart", xlabel="X-axis", ylabel="Y-axis",label1='L1',label2='Spatial'):
    # Unzip the tuples into separate lists for x and y coordinates
    y1, x1 = zip(*list1) if list1 else ([], [])
    y2, x2 = zip(*list2) if list2 else ([], [])
    
    # Create a new figure
    plt.figure(figsize=(8, 6))
    
    # Plot the first list with blue lines and circle markers
    plt.plot(x1, y1, color='blue', marker='o', linestyle='-', label=label1)
    
    # Plot the second list with red lines and square markers
    plt.plot(x2, y2, color='red', marker='s', linestyle='-', label=label2)
    
    # Label the axes and add a title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Optionally, add a legend
    plt.legend()
    
    # Display the plot
    plt.show()


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def evaluate_vit_pruning(vit, threshold, device=None, batch_size=128):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit.to(device)
    vit.eval()

    # Define normalization constants for CIFAR100 (approximate values)
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=test_transform)

    batch_size = 128
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    # Define a helper function to evaluate the model on the test set
    def evaluate_model(model):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return (correct / total) * 100

    # Evaluate the initial model accuracy
    initial_acc = evaluate_model(vit)

    # Count the total number of weights and the number of weights below threshold
    total_params = 0
    small_params = 0
    for param in vit.parameters():
        total_params += param.numel()
        small_params += (param.abs() < threshold).sum().item()
    percent_small = 100.0 * small_params / total_params

    # Zero out all parameters whose absolute value is below the threshold
    with torch.no_grad():
        for param in vit.parameters():
            mask = param.abs() < threshold
            param[mask] = 0.0

    # Evaluate the pruned model accuracy
    final_acc = evaluate_model(vit)

    # Print results
    print("Initial test accuracy: {:.2f}%".format(initial_acc))
    print("Percent of weights below threshold: {:.2f}%".format(percent_small))
    print("Final test accuracy after pruning: {:.2f}%".format(final_acc))

    return initial_acc, percent_small, final_acc

def print_percent_below(arr, t):
    """Prints the percentage of values in 'arr' that are below the threshold t."""
    percent = np.mean(arr < t) * 100
    print(f"Percentage of values below {t}: {percent:.2f}%")


def is_regularized_weight(name):
    """Check if this parameter had L1 regularization applied during training"""
    if 'bias' in name or 'ln' in name or 'wte' in name or 'wpe' in name:
        return False
    if 'weight' not in name:
        return False
    return True

def print_percentiles(weights):
    """Print percentiles of absolute weights"""
    abs_weights = np.abs(weights)
    percentiles = range(10, 101, 10)
    for p in percentiles:
        value = np.percentile(abs_weights, p)
        print(f"P{p}: {value:.6f}")
    print_percent_below(abs_weights,0.001)
def plot_binned_histogram(data, ax=None, title=None, bins=200,xlim=2,scale="log"):
    ax.set_xlim(-xlim, xlim)

    """Helper function to create binned histogram plots"""
    if ax is None:
        ax = plt.gca()
    
    counts, bins = np.histogram(data, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax.bar(bin_centers, counts, width=(bins[1]-bins[0]), alpha=0.7)
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Weight Value', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax.set_yscale(scale)
    ax.tick_params(labelsize=8)
    
    return ax
def load_and_analyze_weights(state_dict):
    """Load and analyze weights directly from state dict"""
    # Analyze distributions from state dict
    layer_weights = {}
    all_weights = []
    
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
    print_percentiles(all_weights)
    return all_weights


def evaluate_metrics(model):
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

    def evaluate(model, dataloader, device):
        """
        Evaluate the model on the given dataloader.
        
        Returns:
            total_loss: Average loss over the dataset.
            accuracy: Fraction of correctly predicted samples.
            precision: Macro-averaged precision.
            recall: Macro-averaged recall.
            f1: Macro-averaged F1 score.
            rocauc: Macro-averaged ROC AUC (one-vs-rest).
        """
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        running_loss = 0.0
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                
                # Get predicted classes and probability distribution
                _, preds = torch.max(outputs, 1)
                probs = nn.functional.softmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        total_loss = running_loss / len(dataloader.dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Calculate macro-averaged precision, recall, and F1 score.
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # Concatenate probability predictions from all batches
        all_probs = np.concatenate(all_probs, axis=0)
        
        # For ROC AUC, we need to binarize the target labels.
        all_targets_binarized = label_binarize(all_targets, classes=np.arange(100))
        try:
            rocauc = roc_auc_score(all_targets_binarized, all_probs, average='macro', multi_class='ovr')
        except Exception as e:
            print("ROC AUC calculation error:", e)
            rocauc = float('nan')
        
        return total_loss, accuracy, precision, recall, f1, rocauc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Evaluate on training data
    train_loss, train_acc, train_precision, train_recall, train_f1, train_rocauc = evaluate(model, train_loader, device)
    # Evaluate on test data
    test_loss, test_acc, test_precision, test_recall, test_f1, test_rocauc = evaluate(model, test_loader, device)
    return train_loss, train_acc, train_precision, train_recall, train_f1, train_rocauc, test_loss, test_acc, test_precision, test_recall, test_f1, test_rocauc





def fgsm_attack(model, loss_fn, images, labels, epsilon):
    """
    Applies the Fast Gradient Sign Method (FGSM) attack on a batch of images.
    
    Args:
        model (torch.nn.Module): The model to attack.
        loss_fn (callable): The loss function used to calculate the gradients.
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): True labels corresponding to the images.
        epsilon (float): The perturbation magnitude.
    

    """
    # Set requires_grad attribute for the input images.
    images.requires_grad = True

    # Forward pass.
    outputs = model(images)
    loss = loss_fn(outputs, labels)

    # Zero all existing gradients.
    model.zero_grad()
    # Compute gradients of loss w.r.t. images.
    loss.backward()

    # Collect the sign of the gradients.
    grad_sign = images.grad.sign()
    # Create perturbed images by adjusting each pixel.
    perturbed_images = images + epsilon * grad_sign

    perturbed_images = torch.clamp(
        perturbed_images, images.min().item(), images.max().item()
    )
    return perturbed_images

def evaluate_robust_accuracy(model, epsilon, batch_size=128, num_workers=2, device=None):
    """
    Loads the CIFAR-100 test dataloader, applies the FGSM attack using the provided epsilon,
    and returns the robust accuracy of the model on the adversarial examples.
    
    Args:
        model (torch.nn.Module): The pretrained model to evaluate.
        epsilon (float): The perturbation magnitude for the FGSM attack.
        batch_size (int, optional): Batch size for the dataloader. Default is 128.
        num_workers (int, optional): Number of subprocesses for data loading. Default is 2.
        device (torch.device, optional): The device to run the evaluation on. If None,
                                         uses GPU if available.
    
    Returns:
        float: Robust accuracy on adversarial examples generated by FGSM.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformation and load CIFAR-100 test set.
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load CIFAR100 dataset
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=test_transform)
    batch_size = 128
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0

    # Loop over the test set batches.
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # Generate adversarial examples.
        adv_images = fgsm_attack(model, loss_fn, images, labels, epsilon)
        # Re-classify the perturbed images.
        outputs = model(adv_images)
        # Get predictions.
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    robust_acc = correct / total
    return robust_acc



def synthetic_transform(img):
    """
    Applies a fixed synthetic shift to the input PIL image.
    Here we rotate by 15° and reduce brightness by 10%.
    """
    # Rotate the image by 15 degrees.
    img = TF.rotate(img, angle=15)
    # Adjust brightness (brightness_factor < 1 reduces brightness).
    img = TF.adjust_brightness(img, brightness_factor=0.9)
    return img

def evaluate_on_synthetic_shifts(model, batch_size=128, num_workers=2, device=None):
    """
    Loads CIFAR-100 test set with a synthetic shift applied, evaluates the model,
    and returns the accuracy.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of worker processes for data loading.
        device (torch.device, optional): Device on which to perform evaluation.

    Returns:
        float: Accuracy of the model on the synthetically shifted CIFAR-100 test set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-100 normalization values.
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    # Define the transform pipeline including the synthetic shift.
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(synthetic_transform),  # Apply synthetic shift here.
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load CIFAR-100 test dataset.
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

