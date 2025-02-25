# Explore the dataset
```
import os
from PIL import Image
import matplotlib.pyplot as plt

# Set the path to your dataset
data_dir = "/Users/Neal/Documents/Projects/CV_OpenProjects/Easy_Projects/Pneumonia_ChestXRAY/chest_xray/"  # Replace with your folder path
print(os.listdir())
# Count images in each class
for split in ["train", "val", "test"]:
    normal_path = os.path.join(data_dir, split, "NORMAL")
    pneumonia_path = os.path.join(data_dir, split, "PNEUMONIA")
    normal_count = len(os.listdir(normal_path))
    pneumonia_count = len(os.listdir(pneumonia_path))
    print(f"{split} - Normal: {normal_count}, Pneumonia: {pneumonia_count}")

# Display a sample image
sample_image_path = os.path.join(data_dir, "test", "NORMAL", os.listdir(normal_path)[0])
img = Image.open(sample_image_path)
print("Image size:", img.size)
print("Image mode:", img.mode)  # 'L' for grayscale, 'RGB' for color

plt.imshow(img, cmap="gray")
plt.title("Sample X-Ray (Normal)")
plt.show()
```
- See the number of samples in each group

- See an example image

# Dataloader Setup

```
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
```
# Define a model
```
# scripts/model.py
import torch.nn as nn
import torch.nn.functional as F

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
```
- Define super basic CNN (4 layers).

    - Layer 1 - 32 3x3 kernels

    - Layer 2 - 64 3x3 kernels

    - Layer 3 - fully connected layer

    - Layer 4 - fully connected layer 

- Uses ReLU for activation function

# Config 
```
# config.py
DATA_PATH = "/Users/Documents/Projects/CV_OpenProjects/Easy_Projects/Pneumonia_ChestXRAY/chest_xray/"  # dataset location
MODEL_SAVE_PATH = 'models'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
```
- Sets the hyperparameters

# Train Model
```
# scripts/train.py
import torch
import torch.optim as optim
from model import SimpleCNN
from data_loader import train_loader, val_loader
import config
import torch.nn as nn

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Training loop
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
```
- Use Adam optimizer

- iterate through epochs and report validation accuracy at each epoch 

# Evaluation
```
# scripts/evaluate.py
import torch
from model import SimpleCNN
from data_loader import test_loader
from sklearn.metrics import classification_report
import config

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(f'{config.MODEL_SAVE_PATH}/simple_cnn.pth'))
model.eval()

# Evaluate on test set
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print detailed metrics
print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))
```
- Reports precision, recall, f1-score, and support

## Interpreting the output

### Definitions

- Precision - Amount of positive predictions that are actually correct

- Recall - proportion of positive cases the model correctly identifies

- F1-score - the mean of precision and recall (balances both equally)

- Support - number of samples in test set for each class

- Accuracy - % correct predictions

### Interpretation (How do I read my results?)
```
              precision    recall  f1-score   support

      Normal       0.90      0.41      0.57       234
   Pneumonia       0.73      0.97      0.84       390

    accuracy                           0.76       624
   macro avg       0.82      0.69      0.70       624
weighted avg       0.80      0.76      0.74       624
```
1. This model indicates a bias towards Pneumonia detection

    -   Excellent at identifying pneumonia (possibly to a fault)
        -   High Recall for Pneumonia (.97), so it is only missing 3% of the actual pneumonia cases

        -   Moderate Precision (.73), this means that 27% of the cases that it predicted as being pneumonia are actually normal (false positive). 

    - Normal Class

        -   High precision (.90), can guess normal correctly most of the time

        -   Low Recall (.41), misses 59% of actual normal cases

    -   Behavior

        -   This is a cautious model, favoring pneumonia predictions to avoid missing cases, but it raises quite a few false alarms  

2. Class Imbalance

    -   The test set has 390 Pneumonia Cases and 234 Normal Cases. Pneumonia being the majority class will cause the model to favor predicting pneumonia because its probability is higher. 

3. Overall Effectiveness

    -   With an accuracy of 76%, things may seem good, but it is a red herring. 

    -   The F1-score illustrates the accuracy is largely driven by class imbalance

        -   Pneumonia (.84) vs Normal (.57) clearly illustrates the model’s performance is substantially better on Pneumonia than Normal. 

        -   We want these to be similar to each other and both > .8 for sure. 

### Takeaways & Improvement

1. Train for more epochs

    -   1 epoch may not be enough for the model to learn the features necessary for accurate prediction. 

2.  Start handling for class imbalance

    -   Use class weights to penalize mistakes on minority class more heavily

    -   Apply data augmentation to generate more samples and balance the training data

3.  Model Changes

    -   Consider transfer learning with a re-trained model (I.e. ResNet18, VGG18)

4.  Additional ideas

    -   Run a ROC curve on different thresholds for the trade-off between precision & recall



## Trying the Improvements

#### More epochs (1 → 5)
```
              precision    recall  f1-score   support

      Normal       0.96      0.33      0.49       234
   Pneumonia       0.71      0.99      0.83       390

    accuracy                           0.74       624
   macro avg       0.84      0.66      0.66       624
weighted avg       0.81      0.74      0.70       624
```
1. Normal Class Performance

    -   Precision (.90 → .96), meaning the model when it predicts normal will very likely be correct

    -   Recall (.41 → .33) - the model is finding fewer true “Normal” cases

    -   F1-score (.57 → .49) - decreased score illustrates an even further disparity in the precision / recall “preference”

2. Pneumonia Class Performance

    -   Class values were largely unchanged (negligible result)

#### TL;DR
Adding epochs did not give us the result we hoped for



### Class Imbalance (Weighted loss function)

We add the following code to the train function
```
from data_loader import train_dataset # For class weight calculation
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
```
Results:
```
              precision    recall  f1-score   support

      Normal       0.90      0.52      0.66       234
   Pneumonia       0.77      0.97      0.86       390

    accuracy                           0.80       624
   macro avg       0.84      0.74      0.76       624
weighted avg       0.82      0.80      0.78       624
```
1. “Normal” Class Performance

    - Precision remained the same (.90 → .90)

    - Recall improved (.41 → .52) - we see meaningful improvement in the model’s ability to distinguish normal cases from pneumonia cases

    - F1-Score has improved (.57 → .66)

2. “Pneumonia” Class Performance

    - Precision improved (.73 → .77)

    - Recall stayed the same (.97 → .97), retaining its high sensitivity to pneumonia features

    - F1-score improved slightly (.84 → .86)

3. Other metrics

    - Accuracy improved (.76 → .80)

    - Macro Avg F1-Score Improved (.70 → .80)

        - Shows each class as equal 

    - Weighted Avg F1-Score Improved (.74 → .78)

        - Weights each class by the support, this metric benefitted from the handling of our minority class

#### TL;DR
Weighting by classes was HIGHLY SUCCESSFUL at improving model performance in virtually all metrics. 

### Data Augmentation

-   I currently employ basic techniques (Random horizontal flip and Random rotation  of up to 10 degrees). 

-   We are going to add the following in order to have our model learn more robust features

    -   Random Cropping and Padding

        -   Crops a portion of the image randomly and pads it back w/ 0s or constant pixel value to encourage model to not memorize any specific part of the image

        ```
        transforms.RandomCrop(size=224, padding=10)
        ```

    -   Gaussian Noise addition

        -   Simulates variations in image quality in the X and Y direction to develop robustness to noisy inputs

    -   Brightness and Contrast Adjustments

        -   Self explanatory

    -   Elastic Transformations

        -   Deforms the image through stretching or squeezing



#### Adjusted data_loader.py script
```
# scripts/data_loader.py

"""
Creates data loaders needed for training, testing, and validation of the model
"""

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import config  # Import configuration settings

# Define Albumentations transforms for training (with augmentation)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add Gaussian noise (optional)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),  # Normalize for 1 channel
    ToTensorV2()
])

# Define Albumentations transforms for validation/test (no augmentation)
val_test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),  # Normalize for 1 channel
    ToTensorV2()
])

# Custom dataset class to apply Albumentations transforms and load grayscale images
class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(AlbumentationsDataset, self).__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path).convert('L')  # Load image as grayscale PIL image
        sample = np.array(sample)  # Convert to NumPy array (shape: (H, W))
        augmented = self.transform(image=sample)  # Apply Albumentations transform
        image = augmented['image']  # Get transformed image (shape: (1, H, W))
        return image, target

# Load datasets with Albumentations transforms
train_dataset = AlbumentationsDataset(root=f'{config.DATA_PATH}/train', transform=train_transform)
val_dataset = AlbumentationsDataset(root=f'{config.DATA_PATH}/val', transform=val_test_transform)
test_dataset = AlbumentationsDataset(root=f'{config.DATA_PATH}/test', transform=val_test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
```


Results:
```
              precision    recall  f1-score   support

      Normal       0.89      0.37      0.52       234
   Pneumonia       0.72      0.97      0.83       390

    accuracy                           0.75       624
   macro avg       0.80      0.67      0.68       624
weighted avg       0.78      0.75      0.71       624
```
The results do not support an improvement over the previous experiment. It may be the case that using only 1 epoch does not provide enough time for the model to learn the features we want it to, and that now as we proceed to higher epochs we will see better results. 

>>> Note: Test at higher epochs

## Trying new architectures

### ResNet18

#### Code Adjustments
Add the following to the model.py script
```
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
```

And the following to train.py:
```
# Initialize model, loss, and optimizer
if model_type == "easy_CNN":
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
elif model_type == 'ResNet18':
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
```

And to evaluate.py
```
# Load the trained model
if config.MODEL == 'ResNet18':
    model = ResNet18().to(device)
elif config.MODEL == 'easy_CNN':
    model = SimpleCNN().to(device)
```

And to config.py
```
MODEL = "ResNet18"
```

#### Results
```
              precision    recall  f1-score   support

      Normal       0.88      0.54      0.67       234
   Pneumonia       0.78      0.96      0.86       390

    accuracy                           0.80       624
   macro avg       0.83      0.75      0.76       624
weighted avg       0.82      0.80      0.79       624
```
Pretty much no improvement seen...why? 

1. Limited Training - ResNet18 is a larger model and therefore will likely require more time to reach convergence than a smaller model (like SimpleCNN)
2. Pre-training effect - Because ResNet18 was trained on the ImageNet dataset, it may require some time to adapt to the new dataset on medical images.
3. ResNet18 just isn't that good - perhaps ResNet18 cannot find any more features than SimpleCNN could. 

#### Limited Training
I ran ResNet18 for 5 epochs, here's the result:
```
              precision    recall  f1-score   support

      Normal       0.89      0.65      0.75       234
   Pneumonia       0.82      0.95      0.88       390

    accuracy                           0.84       624
   macro avg       0.86      0.80      0.82       624
weighted avg       0.85      0.84      0.83       624
```
1. ResNet18 after 5 Epochs vs ResNet18 after 1 Epoch
    - Normal Class
        - Precision (.88 -> .89)
        - Recall (.54 -> .65)
        - F1 (.67 -> .75)
    - Pneumonia Class
        - Precision (.78 -> .82)
        - Recall (.96 -> .95)
        - F1 (.86 -> .88)
    - Overall Metrics 
        - Accuracy (.80 -> .84)
        - Macro avg F1 (.76 -> .82)
        - Weighted F1 (.79 -> 83)`

Pretty clear improvement in all aspects of this model. particular improvement noted to Recall improvement on Normal class. Clearly, more epochs on the ResNet18 suggests undertraining at 1 epoch. Conversely, the SimpleCNN model did not have such an issue. 

2. ResNet18 after 5 Epochs vs SimpleCNN after 5 epochs
Foramt is (SimpleCNN -> ResNet18)
    - Normal Class
        - Precision (.89 -> .89)
        - Recall (.37 -> .65)
        - F1 (.52 -> .75)
    - Pneumonia Class
        - Precision (.72 -> .82)
        - Recall (.97 -> .95)
        - F1 (.83 -> .88)
    - Overall Metrics 
        - Accuracy (.75 -> .84)
        - Macro avg F1 (.68 -> .82)
        - Weighted F1 (.71 -> .83)`

Remarkable Improvement from SimpleCNN to ResNet18!!