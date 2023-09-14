import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFontAutoencoder(nn.Module):
    def __init__(self):
        super(DeepFontAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=12, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # See https://stackoverflow.com/a/58207809/261698
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=12,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepFont(nn.Module):
    def __init__(self, autoencoder: DeepFontAutoencoder, num_classes: int):
        super(DeepFont, self).__init__()

        # Cross-domain subnetwork layers (Cu)
        # This is coming from the encoder.
        self.ae_encoder = autoencoder.encoder
        # Make sure we don't train the autoencoder again.
        for param in self.ae_encoder.parameters():
            param.requires_grad = False

        # Domain-specific layers (Cs)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )

        self.flatten = nn.Flatten()

        hidden_units = 4096
        self.fc1 = nn.Linear(256 * 12 * 12, hidden_units)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        # Cu Layers
        with torch.no_grad():
            out = self.ae_encoder(x)

        # Cs Layers
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))

        out = self.flatten(out)

        out = F.relu(self.fc1(out))
        out = self.drop1(out)
        out = F.relu(self.fc2(out))
        out = self.drop2(out)
        out = self.fc3(out)

        return out
