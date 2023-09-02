import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFont(nn.Module):
    def __init__(self, num_classes):
        super(DeepFont, self).__init__()

        # Cross-domain subnetwork layers (Cu)

        # Input shape to this block [batch, 1, 105, 105].
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=48)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)

        # Input shape to this block [batch, 64, 29, 29].
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=24)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)

        # Input shape to this block [batch, 128, 3, 3].
        self.conv3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=24, stride=2, padding=11
        )
        self.up1 = nn.Upsample(scale_factor=2)

        # Input shape to this block [batch, 128, 12, 12].
        self.conv4 = torch.nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=12, stride=2, padding=5
        )
        self.up2 = nn.Upsample(scale_factor=2)

        # Domain-specific layers (Cs)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=12)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=12)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=12)

        self.fc1 = nn.Linear(256 * 15 * 15, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 2383)
        self.fc4 = nn.Linear(2383, num_classes)

    def forward(self, x):
        # Cu Layers
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.up1(x)

        x = F.relu(self.conv4(x))
        x = self.up2(x)

        # Cs Layers
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, 256 * 15 * 15)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x
