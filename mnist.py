"""
docker run -it \
    --gpus all \
    -v /home/oop/dev/simicam:/workspace/simicam \
    pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
"""
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
import random

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Shuffle labels for train_shuffled and test_shuffled
def shuffle_labels(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    shuffled_labels = [dataset[i][1] for i in indices]
    images = [dataset[i][0] for i in indices]
    return TensorDataset(torch.stack(images), torch.tensor(shuffled_labels))

# Create shuffled datasets
train_shuffled = shuffle_labels(train_dataset)
test_shuffled = shuffle_labels(test_dataset)

# DataLoader for training and test sets
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_shuffled_loader = DataLoader(train_shuffled, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_shuffled_loader = DataLoader(test_shuffled, batch_size=batch_size, shuffle=False)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
test_accuracy_list = []
test_shuffled_accuracy_list = []

for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    # Test accuracy on normal test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    test_accuracy_list.append(test_accuracy)

    # Test accuracy on shuffled test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_shuffled_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_shuffled_accuracy = 100 * correct / total
    test_shuffled_accuracy_list.append(test_shuffled_accuracy)

    print(f"Epoch {epoch+1}, Test Accuracy: {test_accuracy}, Shuffled Test Accuracy: {test_shuffled_accuracy}")

# Plotting test accuracy
plt.figure(figsize=(10, 5))
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.plot(test_shuffled_accuracy_list, label='Shuffled Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('test_accuracies.png')
