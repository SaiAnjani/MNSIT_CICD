import torch
import time

# Define the model again
from train import SimpleCNN

# Save model with timestamp suffix
model = SimpleCNN()
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"model_{timestamp}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")
