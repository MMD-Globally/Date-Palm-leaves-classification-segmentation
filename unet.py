import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net model for multi-class segmentation.
    out_channels should match the number of classes (3 for your case).
    """
    def __init__(self, in_channels=3, out_channels=3, use_batchnorm=True, dropout=0.2):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            ]
            if use_batchnorm:
                layers.insert(1, nn.BatchNorm2d(out_ch))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        # Encoder (Downsampling path)
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.center = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))

        # Decoder (Upsampling path) - add dropout here too
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512))
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        # Final 1x1 convolution (output logits for each class)
        self.final = nn.Conv2d(64, out_channels, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        # Bottleneck
        center = self.center(self.pool4(enc4))
        # Decoder
        dec4 = self.dec4(torch.cat([self.up4(center), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], 1))
        return self.final(dec1)  # No softmax here; use CrossEntropyLoss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)