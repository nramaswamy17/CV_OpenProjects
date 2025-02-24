# scripts/data_loader.py

"""
Creates data loaders needed for training, testing, and validation of the model
"""


import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import config  # Import configuration settings

# Define transformations for training (with augmentation) and validation/test
# Transform for training data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1 channel
])

# Transform for validation/test data
val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1 channel
])

# Load datasets
train_dataset = ImageFolder(root=f'{config.DATA_PATH}/train', transform=train_transform)
val_dataset = ImageFolder(root=f'{config.DATA_PATH}/val', transform=val_test_transform)
test_dataset = ImageFolder(root=f'{config.DATA_PATH}/test', transform=val_test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)