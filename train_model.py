# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

class TaxiDriverDataset(Dataset):
    """
    Custom dataset class for Taxi Driver Classification.
    Handles loading and preparing data for the model.
    """
    def __init__(self, X, y, device='cpu'):
        """
        Initialize the dataset with data and labels.
        
        Parameters:
        - X: A numpy array of shape (num_samples, 100, 8) representing the input features.
        - y: A numpy array of shape (num_samples,) representing the target labels.
        - device: The device ('cpu' or 'cuda') the data should be transferred to.
        """
        
        self.X = torch.tensor(X, dtype=torch.float).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
        self.device = device
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Retrieve the `idx`th sample from the dataset.
        
        Parameters:
        - idx: The index of the sample to retrieve.
        
        Returns:
        A tuple (input, label) where input is the feature tensor for the sample, and label is the corresponding label.
        """
        return self.X[idx], self.y[idx]

def train(model, optimizer, criterion, train_loader, device):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """
    # Set the model to training mode. This is important because some layers, like
    # dropout and batch normalization, behave differently during training and inference.
    model.train()
    
    # Initialize accumulators for total loss and accuracy to compute averages later.
    total_loss = 0
    correct = 0
    total = 0

    # Loop over each batch from the DataLoader.
    for X_batch, y_batch in train_loader:
        # Move the batch of data and labels to the specified device (GPU or CPU).
        # This is important for compatibility with the model's device.
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Zero the gradients computed in the previous iteration to prevent accumulation.
        # If not zeroed, gradients from multiple backward passes would accumulate. Cumulativ gradients is a way to optimize training in multiple GPUs
        optimizer.zero_grad()
        
        # Forward pass: Pass the input through the model to compute predictions.
        outputs = model(X_batch)
        
        # Compute the loss by comparing the model's predictions to the actual labels.
        loss = criterion(outputs, y_batch)
        
        # Backward pass: Compute the gradient of the loss with respect to model parameters.
        loss.backward()
        
        # Update the model parameters using the gradients computed by the backward pass.
        optimizer.step()
        
        # Add the loss of the current batch to the total loss accumulator.
        total_loss += loss.item()
        
        # Predictions are the indices of the max log-probability. The `torch.max`
        # function returns both the maximum values and their indices, but we're
        # only interested in the indices (as they represent the predicted classes).
        _, predicted = torch.max(outputs.data, 1)
        
        # Count the total number of labels for accuracy calculation.
        total += y_batch.size(0)
        
        # Count how many predictions match the actual labels.
        correct += (predicted == y_batch).sum().item()
        
    # Calculate the average loss over all batches.
    train_loss = total_loss / len(train_loader)
    
    # Calculate the accuracy as the percentage of correct predictions.
    train_acc = correct / total
    
    # Return the average loss and accuracy for the epoch.
    return train_loss, train_acc

# Define the testing function
def evaluate(model, criterion, test_loader, device):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    """
    # Set model to evaluation mode to turn off dropout and batch normalization
    model.eval()

    # Initialize variables to track loss and accuracy
    total_loss = 0
    correct = 0
    total = 0

    # Disable gradient computation for evaluation to save memory and computations
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move the batch to the specified device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass: compute the model's predictions for the batch
            outputs = model(X_batch)
            
            # Compute the loss for the batch
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # Convert outputs to probabilities and predict the class with the highest probability
            _, predicted = torch.max(outputs.data, 1)
            
            # Update the total number of correct predictions and the total number of predictions
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    # Calculate average loss and accuracy over all batches in the test set
    test_loss = total_loss / len(test_loader)
    test_acc = correct / total

    # Return the average loss and accuracy
    return test_loss, test_acc

def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """

    num_epochs = 3
    batch_size = 1000
    lr = 0.001
    input_dim = 8
    output_dim = 5 
    
    # Load the dataset
    X, y = load_data("data/train/*.csv") 
    X_val, y_val = load_data("data/val/*.csv") 
    dataset = TaxiDriverDataset(X, y, device='cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = TaxiDriverDataset(X_val, y_val, device='cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = TaxiDriverClassifier(input_dim = input_dim, output_dim = output_dim)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training the model  
    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, position=0, leave=True)
        for X_batch, y_batch in train_loop:
            train_loop.set_description(f'Epoch {epoch+1}/{num_epochs} [Training]')        
            train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
            train_loop.set_postfix(loss=train_loss, acc=train_acc)  
            
        val_loop = tqdm(val_loader, position=0, leave=True)        
        for X_batch, y_batch in val_loop:
            val_loop.set_description(f'Epoch {epoch+1}/{num_epochs} [Validation]')            
            test_loss, test_acc = evaluate(model, criterion, val_loader, device)
            # Update progress bar postfix with validation loss and accuracy
            val_loop.set_postfix(val_loss=test_loss, val_acc=test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}, Accuracy: {train_acc:.4f}, Val Accuracy: {test_acc:.4f}')     

    #Save the final model
    final_model_filename = "model/final_model.pt"
    torch.save(model.state_dict(), final_model_filename)
    print(f"Final model saved to {final_model_filename}")        
    print("Training complete")     
