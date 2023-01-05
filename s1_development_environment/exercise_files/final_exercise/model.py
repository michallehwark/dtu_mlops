from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        #output
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = nn.LogSoftmax(dim=1)
        
        return x