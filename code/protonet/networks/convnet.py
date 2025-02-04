import torch.nn as nn

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, fourier=False, fmt=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        self.x_dim = x_dim
        self.fourier = fourier
        self.fmt = fmt
        
    def forward(self, x):
        x = self.encoder(x)
        if self.x_dim == 3 or self.fmt:
            x = nn.MaxPool2d(5)(x)
        elif self.x_dim == 6:
            x = nn.MaxPool2d(10)(x)
        
        x = x.view(x.size(0), -1)
        return x

