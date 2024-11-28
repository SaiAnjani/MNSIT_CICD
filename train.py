import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Define a simple 3-layer DNN with convolutional and fully connected layers

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # input channels: 1, output channels: 32
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # input channels: 32, output channels: 64
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)  # input channels: 64, output channels: 128

        # Fully connected layer
        self.fc = nn.Linear(32 * 7 * 7, 10)  # Assuming input images are 28x28 (MNIST)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        # MaxPooling layers
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by 2
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by 2

    def forward(self, x):
        # Apply convolution + ReLU + MaxPooling
        x = self.pool1(F.leaky_relu(self.bn2(self.conv2(F.leaky_relu(self.bn1(self.conv1(x)))))))  # First convolution block
        x = self.pool2(F.leaky_relu(self.conv3(x)))  # Second convolution block

        # Flatten the tensor before feeding it into the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 128 * 7 * 7)

        # Apply the fully connected layer and log_softmax
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
        
#         # First Convolutional Layer: 1 input channel, 4 filters, 3x3 kernel
#         self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # 28x28 -> 28x28
#         self.pool = nn.MaxPool2d(2, 2)  # Max pooling with a 2x2 window
        
#         # Second Convolutional Layer: 4 input channels, 8 filters, 3x3 kernel
#         self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # 28x28 -> 28x28
#         # Max Pooling
#         self.conv2_pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(8 * 7 * 7, 61)  # Flattening the 14x14x8 output to 64
#         self.fc2 = nn.Linear(61, 10)  # 10 output units for MNIST classes
        
#     def forward(self, x):
#         # Apply first convolution, then max pooling
#         x = self.pool(torch.relu(self.conv1(x)))
        
#         # Apply second convolution, then max pooling
#         x = self.conv2_pool(torch.relu(self.conv2(x)))
        
#         # Flatten the tensor before passing it to fully connected layers
#         x = x.view(-1, 8 * 7 * 7)  # Flatten the 14x14x8 to 1D vector
        
#         # Apply fully connected layers
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)  # Output layer
#         return x


# Transformations and data loading
transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

# Instantiate model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.02)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
print(f'model parameters: {sum(p.numel() for p in model.parameters())}')

# Train for 1 epoch
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
            
            # Compute accuracy
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += labels.size(0)  # Number of examples in the batch
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    accuracy = (correct / total) * 100  # Accuracy as a percentage
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    

# Save the model as 'model_latest.pth'
torch.save(model.state_dict(), 'model_latest.pth')
