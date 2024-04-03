# model.py

import torch
import torch.nn as nn

class TaxiDriverClassifier(nn.Module):
    """
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    """
    def __init__(self, input_dim, output_dim):
        super(TaxiDriverClassifier, self).__init__()

        self.layers = 2
        self.hidden_layer = 64
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, self.hidden_layer, self.layers, batch_first=True)
        
        # Fully connected layer that outputs the logits for each class
        self.fc = nn.Linear(self.hidden_layer, output_dim)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #Initialize hidden and cell states
        # Dimensions: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_layer).to(x.device)
        c0 = torch.zeros(self.layers, x.size(0), self.hidden_layer).to(x.device)
        
        # Forward propagate the LSTM
        # out: tensor of shape (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        x = self.fc(out[:, -1, :])
        
        return x

    
