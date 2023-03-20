import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_classes, leaky_slope=0.1):
        super(Model, self).__init__()
        self.n_classes = n_classes
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding='same')
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, self.leaky_slope)
        x = self.conv2(x)
        x = F.leaky_relu(x, self.leaky_slope)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.leaky_relu(x, self.leaky_slope)
        x = self.conv4(x)
        x = F.leaky_relu(x, self.leaky_slope)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.leaky_relu(x, self.leaky_slope)
        x = self.dropout2(x)
        output = self.fc2(x)

        return output

