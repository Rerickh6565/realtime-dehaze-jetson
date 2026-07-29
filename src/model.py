"""
model.py — LightDehaze_Net Architecture

Lightweight CNN for single-image dehazing (Light-DehazeNet, IEEE TIP 2021).
The network jointly estimates the transmission map and atmospheric light using
a transformed atmospheric scattering model, producing dehazed output in one pass.

Reference:
    Ullah et al., "Light-DehazeNet: A Novel Lightweight CNN Architecture
    for Single Image Dehazing", IEEE TIP, 2021.
    https://ieeexplore.ieee.org/abstract/document/9562276
"""

import torch
import torch.nn as nn


class LightDehaze_Net(nn.Module):
    """Lightweight dehazing network with dense skip connections."""

    def __init__(self):
        super(LightDehaze_Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Encoder blocks
        self.e_conv_layer1 = nn.Conv2d(3,  8,  1, 1, 0, bias=True)
        self.e_conv_layer2 = nn.Conv2d(8,  8,  3, 1, 1, bias=True)
        self.e_conv_layer3 = nn.Conv2d(8,  8,  5, 1, 2, bias=True)
        self.e_conv_layer4 = nn.Conv2d(16, 16, 7, 1, 3, bias=True)
        self.e_conv_layer5 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer6 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer7 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv_layer8 = nn.Conv2d(56,  3, 3, 1, 1, bias=True)

    def forward(self, img):
        conv1 = self.relu(self.e_conv_layer1(img))
        conv2 = self.relu(self.e_conv_layer2(conv1))
        conv3 = self.relu(self.e_conv_layer3(conv2))

        # Skip: concat conv1 + conv3
        concat1 = torch.cat((conv1, conv3), dim=1)

        conv4 = self.relu(self.e_conv_layer4(concat1))
        conv5 = self.relu(self.e_conv_layer5(conv4))
        conv6 = self.relu(self.e_conv_layer6(conv5))

        # Skip: concat conv4 + conv6
        concat2 = torch.cat((conv4, conv6), dim=1)

        conv7 = self.relu(self.e_conv_layer7(concat2))

        # Skip: concat conv2 + conv5 + conv7
        concat3 = torch.cat((conv2, conv5, conv7), dim=1)

        conv8 = self.relu(self.e_conv_layer8(concat3))

        # Atmospheric scattering model inversion:
        # J(x) = k(x)*I(x) - k(x) + 1  where k(x) = conv8, I(x) = img
        dehazed = self.relu((conv8 * img) - conv8 + 1)

        return dehazed
