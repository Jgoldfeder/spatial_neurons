import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import spatial_wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # Define the transform for CIFAR-100 (resize and normalize)
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to match the input size of ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
    ])

    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    model.head = nn.Linear(model.head.in_features, 100)  # CIFAR-100 has 100 classes
    model.to(device)
    spatial_net = spatial_wrapper.SpatialNet(model, A=1.0, B=1.0, D=1.0, spatial_cost_scale=1e-4,device=device)  # Adjust spatial cost scale
    
    optimizer = optim.Adam(spatial_net.parameters(), lr=0.0001)  # Try a lower learning rate
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        spatial_net.train()
        running_loss = 0.0
        running_cost = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = spatial_net(inputs)
            loss = criterion(outputs, targets)

            # Add the custom cost to the loss
            cost = spatial_net.get_cost()
            total_loss = loss + cost

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_cost += cost.item()
            newtime = datetime.datetime.now()

        print(f"Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader)}")
        print("cost:",running_cost)
        # Validation accuracy
        spatial_net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = spatial_net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc}%")

# Train the model
train_model()
