import torch
from model import Dual_Interface  # Replace with your model class

# Load the state_dict from the .pt file
state_dict = torch.load("models/best_model_trial_8.pt")

# # Create a model instance
model = Dual_Interface()  # Replace with your model initialization

# Load the weights into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode (optional)
model.eval()

# Now you can use the model for inference or further training