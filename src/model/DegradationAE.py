import torch
import torch.nn as nn
import torchvision.models as models

class DegradationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 4), stride=2, output_padding=0):
        super().__init__()
        self.convT = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=1, output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        return self.relu(x)

class DegradationAE(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            DegradationBlock(512, 256),
            DegradationBlock(256, 128),
            DegradationBlock(128, 64),
            DegradationBlock(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
