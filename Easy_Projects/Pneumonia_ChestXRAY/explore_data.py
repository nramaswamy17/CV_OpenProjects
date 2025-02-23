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