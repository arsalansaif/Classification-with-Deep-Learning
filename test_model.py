# test_model.py

import torch
from torch.utils.data import DataLoader, Dataset
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data
from tqdm import tqdm
from train_model import TaxiDriverDataset

# Assuming the extract_feature.py handles preprocessing in a generalized way that can be applied to any data
        
def test(model, test_loader, device):
    """
    Test the model performance on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()  
    
    with torch.no_grad():  # No need to track gradients
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = test_correct / total
    return test_loss, test_accuracy


def test_model():
    """
    Main function to initiate the model testing process.
    Includes loading test data, setting up the model and test loader,
    and executing the testing function.
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the test data
    X,y = X, y = load_data("data/test/*.csv") 
   
    # Create DataLoader
    test_dataset = TaxiDriverDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load your model
    model = TaxiDriverClassifier(input_dim=8, output_dim=5).to(device)  # import trained model
    model.load_state_dict(torch.load('model/final_model.pt'))
    model.to(device)
    
    # Run the test
    test_loss, test_accuracy = test(model, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
