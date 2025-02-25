# scripts/model.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # 224 / 4 = 56 after two pooling layers
        self.fc2 = nn.Linear(128, 2)  # 2 output classes: NORMAL, PNEUMONIA

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def ResNet18():
    # Load pre-trained resnet model
    model = models.resnet18(pretrained=True)

    # modify first conv layer to take grayscale images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Adjust the final layer for binary classification (Normal vs Pneumonia)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Set model to only train the final layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True  # Train only the final layer

    return model