# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# STEP 1: Data Preparation
# ------------------------------------------------------------

# Define the transform:
# - Convert images to tensors
# - Normalize images to have mean 0 and standard deviation 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize R,G,B channels
])

# Set the batch size
batch_size = 128

# Download and load the training dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

# Download and load the test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False)

# CIFAR-10 classes
classes = trainset.classes

# ------------------------------------------------------------
# STEP 2: Define the Convolutional Neural Network (CNN)
# ------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: 3 input channels, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Max pooling layer (2x2 pool size)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer to reduce overfitting (25% dropout)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layer: input features = 64*16*16 after pooling
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        
        # Output layer: 512 input features to 10 classes (CIFAR-10)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Pass input through the first convolutional layer + ReLU activation
        x = F.relu(self.conv1(x))
        
        # Pass through second convolutional layer + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layer + ReLU
        x = F.relu(self.fc1(x))
        
        # Apply dropout again
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the model and move it to the device
net = Net().to(device)

# STEP 3: Define Loss Function and Optimizer

# Cross-Entropy Loss for classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer with learning rate 0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)

# STEP 4: Training the Network

# Lists to store loss and accuracy for plotting
train_losses = []
train_accuracies = []

# Number of training epochs
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop over batches
    for inputs, labels in trainloader:
        # Move inputs and labels to the device (CPU or GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update loss and accuracy statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predicted class
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Calculate and store average loss and accuracy for the epoch
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Training complete.")

# ------------------------------------------------------------
# STEP 5: Plotting Training Loss and Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='green')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 6: Display Predictions on Test Images
# ------------------------------------------------------------

# Function to display an image
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize the image
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Get some random test images
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Get model predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)

# Show the images
plt.figure(figsize=(10, 10))
imshow(torchvision.utils.make_grid(images[:16]))  # Show 16 images
plt.axis('off')  # No axis for cleaner view
plt.show()

# Print Ground Truth and Predictions
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(16)))
print('Predicted:   ', ' '.join(f'{classes[predicted[j]]}' for j in range(16)))
