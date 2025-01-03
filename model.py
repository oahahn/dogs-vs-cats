import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 3x128x128 -> 16x128x128
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 16x64x64 -> 32x64x64
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer
        self.fc1 = nn.Linear(32 * 32 * 32, 2)  # Adjust for second pooling output size

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pooling
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pooling
        x = torch.flatten(x, 1)  # Flatten the tensor for the FC layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc1(x)  # Fully connected layer
        return x
    