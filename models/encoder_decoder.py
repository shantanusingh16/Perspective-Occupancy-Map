import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_encoder import ResnetEncoder

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Encoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation
    Attributes
    ----------
    num_layers : int
        Number of layers to use in the ResNet
    img_ht : int
        Height of the input RGB image
    img_wt : int
        Width of the input RGB image
    pretrained : bool
        Whether to initialize ResNet with pretrained ImageNet parameters
    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """

    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Encoder, self).__init__()

        self.encoder_base = ResnetEncoder(num_layers, pretrained)
        self.num_ch_enc = self.encoder_base.num_ch_enc

        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(self.num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.FloatTensor
            Batch of Image tensors
            | Shape: (batch_size, 3, img_height, img_width)
        Returns
        -------
        x : torch.FloatTensor
            Batch of low-dimensional image representations
            | Shape: (batch_size, 128, img_height/128, img_width/128)
        """

        batch_size, c, h, w = x.shape
        x = self.encoder_base(x)[-1]
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.pool(x)
        return x


class Decoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation
    Attributes
    ----------
    num_ch_enc : list
        channels used by the ResNet Encoder at different layers
    Methods
    -------
    forward(x, ):
        Processes input image features into output occupancy maps/layouts
    """

    def __init__(self, num_ch_enc=128, num_class=2, occ_map_size=64):
        super(Decoder, self).__init__()
        self.num_output_channels = num_class
        self.occ_map_size = occ_map_size
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = num_ch_enc if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(
            self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.rescale_output = lambda x: F.interpolate(x, (self.occ_map_size, self.occ_map_size), mode='bilinear')
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, x, is_training=True):
        """
        Parameters
        ----------
        x : torch.FloatTensor
            Batch of encoded feature tensors
            | Shape: (batch_size, 128, occ_map_size/2^5, occ_map_size/2^5)
        is_training : bool
            whether its training or testing phase
        Returns
        -------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)
        """
        h_x = 0
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)

        x = self.convs["topview"](x)
        x = self.rescale_output(x)
        if not is_training:
            x = nn.Softmax2d()(x)

        return x