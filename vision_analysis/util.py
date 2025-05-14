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
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
import numpy as np
import igraph as ig
import torch.nn as nn

import classification_util


def model_modularity(model, threshold: float = 0.0) -> float:
    """
    Compute size-weighted average Louvain Q over all nn.Linear and nn.Conv2d layers.
    If a layer has no edges > threshold, we set its Q=0 rather than nan.
    """
    total_Q, total_sz = 0.0, 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            W = m.weight.detach().abs().cpu().numpy()
            if isinstance(m, nn.Conv2d):
                W = W.sum(axis=(2, 3))  # collapse k×k → (O, I)

            js, is_ = np.nonzero(W > threshold)
            if js.size == 0:
                Q = 0.0
            else:
                weights = W[js, is_]
                n_in = W.shape[1]
                edges = [(int(i), int(n_in + j)) for i, j in zip(is_, js)]
                g = ig.Graph(edges=edges, directed=False)
                g.es['weight'] = weights.tolist()
                comm = g.community_multilevel(weights='weight')
                Q = g.modularity(comm, weights='weight')
                if np.isnan(Q):
                    Q = 0.0

            total_Q += Q * W.size
            total_sz += W.size

    return (total_Q / total_sz) if total_sz > 0 else 0.0


def count_unique_dead_neurons(state_dict, threshold=1e-3):
    """
    Counts unique dead neurons: those with all-zero incoming or outgoing weights,
    avoiding double-counting neurons that satisfy both conditions.

    Parameters:
        state_dict (dict): Model state_dict
        threshold (float): Threshold below which weights are considered zero

    Returns:
        dead_neuron_counts (dict): Layer-wise dead neuron indices
        total_dead (int): Total unique dead neurons
        total_neurons (int): Total neurons examined
    """
    dead_neuron_indices = {}
    total_neurons = 0
    total_dead = 0

    weight_keys = [k for k in state_dict if "weight" in k and state_dict[k].ndim == 2]
    weight_keys = sorted(weight_keys)

    for i, name in enumerate(weight_keys):
        weight = state_dict[name].detach().cpu().numpy()

        # This layer's output neurons (rows = incoming weights)
        num_out = weight.shape[0]
        incoming_dead = set(idx for idx in range(num_out)
                            if (abs(weight[idx]) < threshold).all())

        dead_neuron_indices[name] = incoming_dead
        total_neurons += num_out

    for i in range(len(weight_keys) - 1):
        curr_name = weight_keys[i]
        next_name = weight_keys[i + 1]
        curr_weight = state_dict[curr_name].detach().cpu().numpy()
        num_in = curr_weight.shape[1]

        outgoing_dead = set(idx for idx in range(num_in)
                            if (abs(curr_weight[:, idx]) < threshold).all())

        # Merge with previous record (from next layer's incoming weights)
        dead_neuron_indices.setdefault(next_name, set())
        dead_neuron_indices[next_name].update(outgoing_dead)

        total_neurons += num_in  # count input neurons once here

    # Now count unique dead neurons across all layers
    unique_dead = sum(len(neuron_set) for neuron_set in dead_neuron_indices.values())
    
    return dead_neuron_indices, unique_dead, total_neurons


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

def plot_lists(data_lists, title="My Chart", xlabel="X-axis", ylabel="Y-axis", labels=None,log=False):
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
    if log:
        plt.yscale('log')   # logarithmic y-axis
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

def evaluate_pruning(vit, threshold, device=None, batch_size=128,dataset_name="cifar100"):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit.to(device)
    vit.eval()

    train_loader, test_loader, num_classes = classification_util.get_data_loaders(dataset_name)

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


def evaluate_metrics(model,cifar100=True,dataset_name="cifar100"):


    train_loader, test_loader, num_classes = classification_util.get_data_loaders(dataset_name)


    def evaluate(model, dataloader, device,cifar100=True):
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
        num_classes=102

        all_targets_binarized = label_binarize(all_targets, classes=np.arange(num_classes))
        try:
            rocauc = roc_auc_score(all_targets_binarized, all_probs, average='macro', multi_class='ovr')
        except Exception as e:
            print("ROC AUC calculation error:", e)
            rocauc = float('nan')
        
        return total_loss, accuracy, precision, recall, f1, rocauc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Evaluate on training data
    train_loss, train_acc, train_precision, train_recall, train_f1, train_rocauc = evaluate(model, train_loader, device,cifar100)
    # Evaluate on test data
    test_loss, test_acc, test_precision, test_recall, test_f1, test_rocauc = evaluate(model, test_loader, device,cifar100)
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

def evaluate_robust_accuracy(model, epsilon, batch_size=128, num_workers=2, device=None,dataset_name="cifar100"):
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


    train_loader, test_loader, num_classes = classification_util.get_data_loaders(dataset_name)



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

def evaluate_on_synthetic_shifts(model, batch_size=128, num_workers=2, device=None, dataset_name="cifar100"):
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


    train_loader, test_loader, num_classes = classification_util.get_data_loaders(dataset_name)

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


import torch

def compute_pruning_threshold(model, p):
    """
    Computes the threshold value t for pruning such that, if all weights with magnitudes below t 
    are pruned (set to zero), approximately p percent of the weights in the model remain.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model containing the parameters to be pruned.
    p : float
        The percentage (between 0 and 100, exclusive of 0 and inclusive of 100) of weights
        that should remain after pruning.

    Returns:
    --------
    float
        The threshold value t for pruning.
        
    Raises:
    -------
    ValueError
        If p is not in the interval (0, 100].
    """
    if p <= 0 or p > 100:
        raise ValueError("p must be in the interval (0, 100].")

    # Gather all absolute weight values from the model into one tensor.
    all_abs_weights = torch.cat([param.detach().abs().flatten() for param in model.parameters()])

    # Determine the quantile such that (100 - p)% of weights are below the threshold.
    # p percent of weights remaining means we want to prune 1 - (p/100) fraction of weights.
    quantile_value = 1.0 - (p / 100.0)
    threshold = torch.quantile(all_abs_weights, quantile_value)
    
    return threshold.item()


def compute_pruning_threshold_cpu(model, p):
    if not (0 < p <= 100):
        raise ValueError("p must be in (0,100]")

    # 1) Count total parameters
    total = sum(param.numel() for param in model.parameters())

    # 2) Allocate one big CPU tensor
    all_abs = torch.empty(total, dtype=torch.float32, device="cpu")

    # 3) Copy into it
    idx = 0
    for param in model.parameters():
        flat = param.detach().abs().flatten().cpu()
        n = flat.numel()
        all_abs[idx:idx+n].copy_(flat)
        idx += n

    # 4) Compute exact quantile (torch.kthvalue under the hood)
    #    We want the kth largest where k = ceil((p/100)*total)
    k = int(torch.ceil(torch.tensor(p/100.0 * total)).item()) - 1  # zero-based
    # since torch.kthvalue finds the k-th **smallest**, we need the (total−k)th smallest
    threshold, _ = all_abs.kthvalue(total - k)
    return threshold.item()
