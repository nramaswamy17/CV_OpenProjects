# scripts/train.py
import torch
import torch.optim as optim
from model import SimpleCNN, ResNet18
from data_loader import train_loader, val_loader
from data_loader import train_dataset # For class weight calculation
import config
import torch.nn as nn
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## apply class weights to loss function
# Calculate the number of samples per class
class_counts = Counter(train_dataset.targets)
N_normal = class_counts[0]  # Assuming class 0 is 'Normal'
N_pneumonia = class_counts[1]  # Assuming class 1 is 'Pneumonia'
N_total = len(train_dataset)

# Calculate class weights: weight_class = N_total / (2 * N_class)
weight_normal = N_total / (2.0 * N_normal)
weight_pneumonia = N_total / (2.0 * N_pneumonia)

# Create a tensor of class weights
class_weights = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32).to(device)
## End class weights application

# Set model type
model_type = config.MODEL

# Initialize model, loss, and optimizer
if model_type == "easy_CNN":
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
elif model_type == 'ResNet18':
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Training loop
print("Initialization success")
for epoch in range(config.NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {running_loss / len(train_loader):.4f}')

    # Validation step
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), f'{config.MODEL_SAVE_PATH}/simple_cnn.pth')