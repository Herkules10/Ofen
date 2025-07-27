import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet152, ResNet152_Weights
from torch.utils.data import DataLoader
import time
from GeneticOperators import generate_random_genome
from NeuralNetwork import NeuralNetwork

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms - ResNet expects larger images, so we'll resize
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Fashion-MNIST is grayscale
])

# Load Fashion-MNIST
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

input_shape = (1, 64, 64)
output_size = 10
# Modify ResNet18 for Fashion-MNIST (grayscale)
# model = resnet18(weights=ResNet18_Weights.DEFAULT)
genome = generate_random_genome(input_shape=input_shape, output_size=output_size, num_layers=5)
model = NeuralNetwork(input_shape, output_size, genome)
# model = resnet152(weights=ResNet152_Weights.DEFAULT)
# Change first layer to accept 1 channel instead of 3
#model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Adjust final layer to output 10 classes
#model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
                
        # Test the model after each epoch
        test_accuracy = test()
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%')

# Test the model
def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_inference_time = 0
        num_inferences = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            inference_start = time.time()
            outputs = model(images)
            inference_end = time.time()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            inference_time = inference_end - inference_start
            total_inference_time += inference_time
            num_inferences += 1
        accuracy = 100 * correct / total
        average_inference_time = total_inference_time / num_inferences
        print(f"avg inference time: {average_inference_time}")
        return accuracy

# Train and test
print("Starting training...")
train(num_epochs=3)
print("Final test accuracy: {:.2f}%".format(test()))