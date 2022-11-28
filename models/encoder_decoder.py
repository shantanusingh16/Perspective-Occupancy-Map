import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_encoder import ResnetEncoder


################################################## Monolayout #################################################


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

    def __init__(self, num_layers, img_ht, img_wt, pretrained=True, last_pool=True):
        super(Encoder, self).__init__()

        self.encoder_base = ResnetEncoder(num_layers, pretrained)
        self.num_ch_enc = self.encoder_base.num_ch_enc

        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(self.num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.last_pool = last_pool

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
        if self.last_pool:
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



########################################## Occupancy Anticipation ##########################
'''
https://github.com/facebookresearch/OccupancyAnticipation/blob/aea6a2c0d9701336c01c9c85f5c3b8565d7c52ba/occant_baselines/models/unet.py
'''

class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, n_channels, nsf=16):
        super().__init__()
        self.inc = inconv(n_channels, nsf)
        self.down1 = down(nsf, nsf * 2)
        self.down2 = down(nsf * 2, nsf * 4)
        self.down3 = down(nsf * 4, nsf * 8)
        self.down4 = down(nsf * 8, nsf * 8)

    def forward(self, x):
        x1 = self.inc(x)  # (bs, nsf, ..., ...)
        x2 = self.down1(x1)  # (bs, nsf*2, ... ,...)
        x3 = self.down2(x2)  # (bs, nsf*4, ..., ...)
        x4 = self.down3(x3)  # (bs, nsf*8, ..., ...)
        x5 = self.down4(x4)  # (bs, nsf*8, ..., ...)

        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class UNetDecoder(nn.Module):
    def __init__(self, n_classes, nsf=16):
        super().__init__()
        self.up1 = up(nsf * 16, nsf * 4)
        self.up2 = up(nsf * 8, nsf * 2)
        self.up3 = up(nsf * 4, nsf)
        self.up4 = up(nsf * 2, nsf)
        self.outc = outconv(nsf, n_classes)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x1 = xin["x1"]  # (bs, nsf, ..., ...)
        x2 = xin["x2"]  # (bs, nsf*2, ..., ...)
        x3 = xin["x3"]  # (bs, nsf*4, ..., ...)
        x4 = xin["x4"]  # (bs, nsf*8, ..., ...)
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.up1(x5, x4)  # (bs, nsf*4, ..., ...)
        x = self.up2(x, x3)  # (bs, nsf*2, ..., ...)
        x = self.up3(x, x2)  # (bs, nsf, ..., ...)
        x = self.up4(x, x1)  # (bs, nsf, ..., ...)
        x = self.outc(x)  # (bs, n_classes, ..., ...)

        return x


#################################### FPN Block ########################################

class FPN(nn.Module):
    def __init__(self) -> None:
        super(FPN, self).__init__()
        self.encoder_base = ResnetEncoder(18, True)
        self.num_ch_enc = self.encoder_base.num_ch_enc

        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(self.num_ch_enc[-1], 512)
        self.conv2 = Conv3x3(512, 512)
        self.pool = nn.MaxPool2d(2)

        self.up1 = up(1024, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(768, 256)
        self.up4 = up(384, 128)

        self.scales = [8, 16, 32, 64, 128]
        self.num_channels = [128, 256, 512, 512, 512]

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder_base(x)
        x5 = self.pool(self.conv1(x4))
        x6 = self.pool(self.conv2(x5))

        y6 = x6
        y5 = self.up1(y6, x5)
        y4 = self.up2(y5, x4)
        y3 = self.up3(y4, x3)
        y2 = self.up4(y3, x2)

        return [y2, y3, y4, y5, y6]