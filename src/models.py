import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=4, output_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        
        # Initial convolution
        self.initial_down = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        
        # Downsampling path (Encoder)
        self.down1 = self._block(features, features * 2, 4, 2, 1)      # 128
        self.down2 = self._block(features * 2, features * 4, 4, 2, 1)  # 256
        self.down3 = self._block(features * 4, features * 8, 4, 2, 1)  # 512
        self.down4 = self._block(features * 8, features * 8, 4, 2, 1)  # 512 (Bottleneck)
        self.down5 = self._block(features * 8, features * 8, 4, 2, 1)  # 512
        self.down6 = self._block(features * 8, features * 8, 4, 2, 1)  # 512

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        # Upsampling path (Decoder)
        self.up1 = self._block(features * 8, features * 8, 4, 2, 1, transpose=True, dropout=True)
        self.up2 = self._block(features * 16, features * 8, 4, 2, 1, transpose=True, dropout=True) # Skip connection: 8 + 8 = 16
        self.up3 = self._block(features * 16, features * 8, 4, 2, 1, transpose=True, dropout=True)
        self.up4 = self._block(features * 16, features * 8, 4, 2, 1, transpose=True)
        self.up5 = self._block(features * 16, features * 4, 4, 2, 1, transpose=True)
        self.up6 = self._block(features * 8, features * 2, 4, 2, 1, transpose=True)
        self.up7 = self._block(features * 4, features, 4, 2, 1, transpose=True)

        # Final convolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, output_channels, 4, 2, 1),
            nn.Tanh(), # Output generation is usually -1 to 1 or 0 to 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, dropout=False):
        layers = []
        if transpose:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect"))
        
        layers.append(nn.BatchNorm2d(out_channels))
        
        if transpose:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU(0.2)) # Standard for GAN discriminators/generators
            
        if dropout:
            layers.append(nn.Dropout(0.5))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        bottleneck = self.bottleneck(d7)
        
        # Upsample with Skip Connections
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final_up(torch.cat([u7, d1], 1))

def test():
    x = torch.randn((1, 4, 256, 256))
    model = UNetGenerator(input_channels=4, output_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
