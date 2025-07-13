import torch.nn as nn
import torch.nn.functional as F

class PyTorchCNN(nn.Module):
  def __init__(self):
    super(PyTorchCNN, self).__init__()

    # Block 1:
    self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)
    self.dropout1 = nn.Dropout(0.25)

    # Block 2:
    self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool2d(stride=4, kernel_size=2)
    self.dropout2 = nn.Dropout(0.25)

    # Fully connected layers
    self.fc1 = nn.Linear(in_features=2048, out_features=256)
    self.dropout3 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(in_features=256, out_features=1)

  def forward(self, x):
    # Block 1:
    x = F.relu(self.conv1_1(x))
    x = F.relu(self.conv1_2(x))
    x = self.pool1(x)
    x = self.dropout1(x)

    # Block 2:
    x = F.relu(self.conv2_1(x))
    x = F.relu(self.conv2_2(x))
    x = self.pool2(x)
    x = self.dropout2(x)

    # full connected
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.dropout3(x)
    x = self.fc2(x)
    x = F.sigmoid(x)

    return x