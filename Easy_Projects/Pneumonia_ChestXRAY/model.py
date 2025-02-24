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