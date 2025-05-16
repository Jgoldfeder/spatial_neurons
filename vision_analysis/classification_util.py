import os
import zipfile
import shutil
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import tensorflow_datasets as tfds
from datasets import load_dataset
import timm

# === Wrap in a PyTorch Dataset ===
class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class TinyImageNetDataset:
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    ZIP_NAME = "tiny-imagenet-200.zip"
    DIR_NAME = "tiny-imagenet-200"
    
    def __init__(self, root="data", batch_size=128, num_workers=4, image_size=224):
        self.root = root
        self.data_dir = os.path.join(root, self.DIR_NAME)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.mean = [0.485, 0.456, 0.406]  # ImageNet stats
        self.std  = [0.229, 0.224, 0.225]

        self._prepare_dataset()
        self._init_datasets()

    def _prepare_dataset(self):
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            return  # already downloaded and unpacked
        
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, self.ZIP_NAME)

        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(self.URL, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)

        os.remove(zip_path)
        self._reorganize_validation_folder()

    def _reorganize_validation_folder(self):
        print("Reorganizing validation directory...")
        val_dir = os.path.join(self.data_dir, 'val')
        images_dir = os.path.join(val_dir, 'images')
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')

        with open(annotations_file, 'r') as f:
            for line in f:
                img_name, class_name = line.strip().split('\t')[:2]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                shutil.move(os.path.join(images_dir, img_name),
                            os.path.join(class_dir, img_name))

        shutil.rmtree(images_dir)

    def _init_datasets(self):
        # Use Resize(256) + Crop(224) if image_size >= 224; else direct resize
        if self.image_size >= 224:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

        train_path = os.path.join(self.data_dir, 'train')
        val_path = os.path.join(self.data_dir, 'val')

        self.train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        self.val_dataset = datasets.ImageFolder(val_path, transform=test_transform)

    def get_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader

# Wrap TFDS as a PyTorch Dataset
class TFDSPytorchWrapper(torch.utils.data.Dataset):
    def __init__(self, tfds_split, transform=None):
        self.ds = list(tfds.as_numpy(tfds_split))
        self.transform = transform

    def __getitem__(self, idx):
        example = self.ds[idx]
        image = example['image']
        label = example['label']
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.ds)
        
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def f(x):
    if x.shape[0] == 1:
        return x.repeat(3,1,1)
    return x


# Data augmentation and normalization for training, and normalization for testing
train_transform = transforms.Compose([
    transforms.Resize(256),                # Resize to a bit larger than final crop
    transforms.RandomCrop(224),            # Random crop to 224x224
    transforms.RandomHorizontalFlip(),     # Data augmentation
    transforms.ToTensor(),
    transforms.Lambda(lambda x: f(x)),
    transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: f(x)),
    transforms.Normalize(mean, std),
])


def get_data_loaders(name,batch_size=128):
    datasets = ["STL10","DTD","birds","cars196","flowers102","food101","aircraft","tiny_imagenet","caltech101","cifar10","cifar100", "pets", "svhn","dogs"]
    if name not in datasets:
        raise ValueError("invalid dataset name, choose from", datasets)
    num_classes_dict ={    
        'cifar100': 100,
        'cifar10': 10,
        'aircraft':102,
        'food101': 101,
        'flowers102': 102,
        'STL10': 10,
        'svhn':10,
        'pets': 37,
        'caltech101': 102, # off by one indexing
        'DTD': 47,
        'dogs': 120,
        'birds': 200,
        'cars196': 196,
        'tiny_imagenet': 200,
    }
    num_classes = num_classes_dict[name]
    if name == "svhn":
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', transform=train_transform, download=True)
        test_dataset  = torchvision.datasets.SVHN(root='./data', split='test', transform=test_transform, download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    if name == "pets":
        train_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval",
                                                    download=True, transform=train_transform)
        test_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split="test",
                                                    download=True, transform=test_transform)
        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    if name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                    download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                    download=True, transform=test_transform)

        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2) 
        
    if name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=test_transform)

        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)                                      
                            
    if name == "caltech101":


        # Load TFDS splits
        tfds_train = tfds.load('caltech101', split='train', as_supervised=False)
        tfds_test = tfds.load('caltech101', split='test', as_supervised=False)

        # Wrap into PyTorch datasets
        train_dataset = TFDSPytorchWrapper(tfds_train, transform=train_transform)
        test_dataset = TFDSPytorchWrapper(tfds_test, transform=test_transform)


        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    if name == "tiny_imagenet":
        dataset = TinyImageNetDataset(root="data", batch_size=batch_size, image_size=224)
        train_loader, test_loader = dataset.get_loaders()

    if name == "aircraft":
        train_dataset = torchvision.datasets.FGVCAircraft(root='./data', split='train', download=True,
                                transform=train_transform)
        test_dataset = torchvision.datasets.FGVCAircraft(root='./data', split='test', download=True,
                                    transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if name== "food101":
        train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)

        # === Wrap in DataLoaders ===
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if name== "DTD":
        train_dataset = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=test_transform)

        # === Wrap in DataLoaders ===
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if name== "STL10":
        train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=test_transform)

        # === Wrap in DataLoaders ===
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if name == "flowers102":    
        train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=test_transform)

        # === Wrap in DataLoaders ===
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if name == "cars196":
        dataset = load_dataset("tanganke/stanford_cars")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        torch_train_dataset = HFImageDataset(train_dataset, transform=train_transform)
        torch_test_dataset  = HFImageDataset(test_dataset, transform=test_transform)

        # === Create DataLoaders ===
        train_loader = DataLoader(torch_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader  = DataLoader(torch_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if name == "dogs":
        # Load TFDS splits
        tfds_train = tfds.load('stanford_dogs', split='train', as_supervised=False)
        tfds_test = tfds.load('stanford_dogs', split='test', as_supervised=False)

        # Wrap into PyTorch datasets
        train_dataset = TFDSPytorchWrapper(tfds_train, transform=train_transform)
        test_dataset = TFDSPytorchWrapper(tfds_test, transform=test_transform)


        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)   

    if name == "birds":
        dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        torch_train_dataset = HFImageDataset(train_dataset, transform=train_transform)
        torch_test_dataset  = HFImageDataset(test_dataset, transform=test_transform)

        # === Create DataLoaders ===
        train_loader = DataLoader(torch_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader  = DataLoader(torch_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader,test_loader, num_classes



def get_model(name,num_classes,pretrained=True):
    models = ['vit_tiny_patch16_224','vit_base_patch16_224','resnet50','resnet101','efficientnet_b0','vgg19','visformer_small','swin_base_patch4_window7_224','mobilenetv3_small_100','densenet121']
    if name not in models:
        print("invalid model")
    
    model = timm.create_model(name, pretrained=pretrained)

    if name in ['vit_tiny_patch16_224','vit_base_patch16_224','visformer_small',]:
        num_features = model.head.in_features
        model.head = torch.nn.Linear(num_features, num_classes)
    if name in ['resnet50','resnet101']:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    if name in ['efficientnet_b0','densenet121','mobilenetv3_small_100']:
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, num_classes)
    if name in ['vgg19','swin_base_patch4_window7_224']:
        num_features = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_features, num_classes)
    return model