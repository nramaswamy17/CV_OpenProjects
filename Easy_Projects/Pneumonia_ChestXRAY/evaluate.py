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