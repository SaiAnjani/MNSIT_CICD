import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import glob
from train import SimpleCNN  # Import the model from the training script

# Find the latest model file based on timestamp
model_files = glob.glob("model_*.pth")  # Matches files like model_YYYYMMDD-HHMMSS.pth
if not model_files:
    raise FileNotFoundError("No model files found. Please train a model first.")
latest_model_file = max(model_files, key=os.path.getmtime)  # Get the latest model based on modification time

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load(latest_model_file))

# Check model parameters
def check_model_parameters(model):
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 25000, f"Model has too many parameters: {param_count}"

# Check model input size
def check_input_size(model, sample_input):
    try:
        output = model(sample_input)
        print(f''Model size validated & Test passed)
    except Exception as e:
        assert False, f"Model failed on 28x28 input: {e}"

# Check model output size
def check_output_size(model):
    dummy_input = torch.randn(1, 1, 28, 28)  # Example of a single 28x28 MNIST image
    output = model(dummy_input)
    assert output.shape[1] == 10, f"Output size is not 10: {output.shape[1]}"

# Check accuracy
def check_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    assert accuracy > 90, f"Accuracy is too low: {accuracy}%"

# Load MNIST test data
testset = datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())
# datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Perform checks
check_model_parameters(model)
check_input_size(model, torch.randn(1, 1, 28, 28))  # Test with a dummy input
check_output_size(model)
check_accuracy(model, testloader)

print("All tests passed!")
