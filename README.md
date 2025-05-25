# BCR in Embedded Code

Train Google Colab:
```python
# CIFAR-10 Simple Neural Network (< 1000 parameters)
# Run this in Google Colab

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import time

# Network Architecture: 3072 -> 16 -> 10 (Total: 3072*16 + 16*10 + 16 + 10 = 49,338 parameters)
# This is too many! Let's reduce to: 3072 -> 8 -> 10 (Total: 3072*8 + 8*10 + 8 + 10 = 24,674 parameters)
# Still too many! Let's use: 32*32*3 averaged to patches -> 64 features -> 8 -> 10

class SimpleCIFAR10Net(nn.Module):
    def __init__(self):
        super(SimpleCIFAR10Net, self).__init__()
        # Input: 32x32x3 = 3072, but we'll use 4x4 average pooling to get 8x8x3 = 192 features
        # Then: 192 -> 8 -> 10
        # Parameters: 192*8 + 8*10 + 8 + 10 = 1536 + 80 + 18 = 1634 (still too many)
        
        # Let's use: 8x8x3 averaged to 4x4x3 = 48 features -> 16 -> 10
        # Parameters: 48*16 + 16*10 + 16 + 10 = 768 + 160 + 26 = 954 parameters ✓
        
        self.pool = nn.AvgPool2d(8, stride=8)  # 32x32 -> 4x4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48, 16)  # 4*4*3 = 48 -> 16
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)  # 16 -> 10 classes
        
    def forward(self, x):
        x = self.pool(x)      # [batch, 3, 32, 32] -> [batch, 3, 4, 4]
        x = self.flatten(x)   # [batch, 48]
        x = self.fc1(x)       # [batch, 16]
        x = self.relu(x)
        x = self.fc2(x)       # [batch, 10]
        return x

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize model
model = SimpleCIFAR10Net()
param_count = count_parameters(model)
print(f"Total parameters: {param_count}")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# CIFAR-10 classes
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
model.train()
for epoch in range(20):  # Quick training
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
            running_loss = 0.0

# Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Extract weights for C code
model.eval()
model.to('cpu')

# Get weights and biases
fc1_weight = model.fc1.weight.detach().numpy()  # [16, 48]
fc1_bias = model.fc1.bias.detach().numpy()      # [16]
fc2_weight = model.fc2.weight.detach().numpy()  # [10, 16]
fc2_bias = model.fc2.bias.detach().numpy()      # [10]

print("\n=== WEIGHTS FOR C CODE ===")
print("// FC1 Weight [16][48]")
print("float fc1_weight[16][48] = {")
for i in range(16):
    print("  {", end="")
    for j in range(48):
        print(f"{fc1_weight[i][j]:.6f}f", end="")
        if j < 47: print(", ", end="")
    print("},")
print("};")

print("\n// FC1 Bias [16]")
print("float fc1_bias[16] = {")
for i in range(16):
    print(f"  {fc1_bias[i]:.6f}f", end="")
    if i < 15: print(",")
print("\n};")

print("\n// FC2 Weight [10][16]")
print("float fc2_weight[10][16] = {")
for i in range(10):
    print("  {", end="")
    for j in range(16):
        print(f"{fc2_weight[i][j]:.6f}f", end="")
        if j < 15: print(", ", end="")
    print("},")
print("};")

print("\n// FC2 Bias [10]")
print("float fc2_bias[10] = {")
for i in range(10):
    print(f"  {fc2_bias[i]:.6f}f", end="")
    if i < 9: print(",")
print("\n};")

# Test single inference time
test_input = torch.randn(1, 3, 32, 32)
model.eval()
with torch.no_grad():
    start_time = time.time()
    output = model(test_input)
    end_time = time.time()
    print(f"\nPyTorch inference time: {(end_time - start_time) * 1000:.3f} ms")
```

Image to Value:
```python
# CIFAR-10 RGB888 Extractor
# This script loads a CIFAR-10 image and generates C code with int [32][32][3] format

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from google.colab import files

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Class names in CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Get user input for class and image number
print("Available classes:", class_names)
class_name = input("Enter class name (e.g., airplane): ").strip().lower()
image_number = int(input("Enter image number (1-10): ").strip())

# Validate inputs
if class_name not in class_names:
    print(f"Error: '{class_name}' is not a valid class. Using 'airplane' instead.")
    class_name = 'airplane'

if image_number < 1 or image_number > 10:
    print(f"Error: Image number must be between 1 and 10. Using 1 instead.")
    image_number = 1

# Get the class index
class_idx = class_names.index(class_name)

# Find the indices for this class
indices = np.where(y_train == class_idx)[0]

# Get the RGB888 values for the specified image (0-indexed)
rgb888_values = x_train[indices[image_number-1]]

# Display the image
plt.figure(figsize=(4, 4))
plt.imshow(rgb888_values)
plt.title(f"CIFAR-10: {class_name} #{image_number}")
plt.axis('off')
plt.show()

# Generate C-style int [32][32][3] array declaration
c_code = f"// RGB888 values for CIFAR-10 image: {class_name} #{image_number}\n"
c_code += "// Format: int [32][32][3] where the 3 values are R, G, B (0-255)\n"
c_code += "int rgb888_values[32][32][3] = {\n"

for i in range(32):
    c_code += "    {\n        "
    for j in range(32):
        r, g, b = rgb888_values[i, j]
        c_code += f"{{{r}, {g}, {b}}}"
        if j < 31:
            c_code += ", "
    c_code += "\n    }"
    if i < 31:
        c_code += ","
    c_code += "\n"

c_code += "};"

# Create a C header file
header_filename = f"cifar10_{class_name}_{image_number}_rgb888.h"
with open(header_filename, "w") as f:
    f.write(f"""/*
 * CIFAR-10 RGB888 Image Data
 * Class: {class_name}
 * Image: {image_number}
 * Size: 32x32x3 (RGB888)
 */

#ifndef CIFAR10_{class_name.upper()}_{image_number}_RGB888_H
#define CIFAR10_{class_name.upper()}_{image_number}_RGB888_H

{c_code}

#endif /* CIFAR10_{class_name.upper()}_{image_number}_RGB888_H */
""")

# Display a small preview of the C code
print("\nPreview of the C code:")
print("\n".join(c_code.split("\n")[:10]))
print("...\n")

# Save the file and provide download link
print(f"Created header file: {header_filename}")
files.download(header_filename)

# Alternative: Create a function to load the image in C
c_function = f"""
/*
 * Function to load the CIFAR-10 image into a buffer
 *
 * Parameters:
 * - buffer: pointer to an int[32][32][3] array to hold the image data
 */
void load_cifar10_{class_name}_{image_number}(int buffer[32][32][3]) {{
    static const int image_data[32][32][3] = {{
"""

for i in range(32):
    c_function += "        {\n            "
    for j in range(32):
        r, g, b = rgb888_values[i, j]
        c_function += f"{{{r}, {g}, {b}}}"
        if j < 31:
            c_function += ", "
    c_function += "\n        }"
    if i < 31:
        c_function += ","
    c_function += "\n"

c_function += """    };

    // Copy data to the buffer
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            buffer[i][j][0] = image_data[i][j][0]; // R
            buffer[i][j][1] = image_data[i][j][1]; // G
            buffer[i][j][2] = image_data[i][j][2]; // B
        }
    }
}
"""

# Create a C source file with the function
source_filename = f"cifar10_{class_name}_{image_number}_rgb888.c"
with open(source_filename, "w") as f:
    f.write(c_function)

# Provide download link for the source file
print(f"Created source file: {source_filename}")
files.download(source_filename)

# Display sample pixel values
print("\nSample pixel values:")
print(f"Pixel (0,0): R={rgb888_values[0,0,0]}, G={rgb888_values[0,0,1]}, B={rgb888_values[0,0,2]}")
print(f"Pixel (15,15): R={rgb888_values[15,15,0]}, G={rgb888_values[15,15,1]}, B={rgb888_values[15,15,2]}")
print(f"Pixel (31,31): R={rgb888_values[31,31,0]}, G={rgb888_values[31,31,1]}, B={rgb888_values[31,31,2]}")
```
