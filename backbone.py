#!/usr/bin/env python3

from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential
from torchvision.models.mobilenetv2 import MobileNetV2, InvertedResidual, _make_divisible, ConvBNReLU
from torchvision.models.mobilenetv3 import InvertedResidual as InvertedResidual_v3
from torchvision.models.mobilenetv3 import InvertedResidualConfig
from torchvision.models.resnet import BasicBlock


class ResNet(nn.Module):
    def __init__(self, in_channels, use_deconv=False):
        super(ResNet, self).__init__()

        first_channel = 128
        self.outc = first_channel * 2
        self.use_deconv = use_deconv

        self.down0 = nn.Sequential(
            nn.Conv2d(in_channels, first_channel, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(first_channel),
            nn.ReLU(inplace=True),
        )
        self.down1 = self.gen_down(first_channel, first_channel, 2, nblocks=5)
        self.down2 = self.gen_down(first_channel, first_channel * 2, 2, nblocks=5)
        self.down3 = self.gen_down(first_channel * 2, first_channel * 4, 2, nblocks=5)
        self.up0 = self.gen_up(first_channel * 4, first_channel * 2, 4)
        self.up1 = self.gen_up(first_channel * 2 + first_channel, first_channel * 2, 4)

        # initialize
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def gen_down(self, inchannels, outchannels, stride, nblocks):
        downsample = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannels),
        )
        blocks = [BasicBlock(inchannels, outchannels, stride=stride, downsample=downsample)]
        for _ in range(nblocks - 1):
            blocks.append(
                BasicBlock(outchannels, outchannels, stride=1)
            )
        return nn.Sequential(*blocks)

    def gen_up(self, inchannels, outchannels, scale):
        if self.use_deconv:
            pad = 1 if scale == 2 else 0
            o_pad = 1 if scale == 2 else scale - 3
            return nn.Sequential(
                nn.ConvTranspose2d(inchannels, outchannels, 3, scale, pad, o_pad, bias=False),
                nn.BatchNorm2d(outchannels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inchannels, outchannels, 3, padding=1, bias=False),  # 3 is kernel_size.
                nn.BatchNorm2d(outchannels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale)
            )

    def forward(self, x):
        x = self.down0(x)
        x2 = self.down1(x)
        x = self.down2(x2)
        x = self.down3(x)
        x = self.up0(x)
        x = torch.cat((x, x2), 1)
        x = self.up1(x)
        return x


class ResNetv2(ResNet):
    """
    ResNet (overall downsampled by 2)
    """

    def __init__(self, in_channels, use_deconv=False):
        super(ResNetv2, self).__init__(in_channels, use_deconv)

        first_channel = 128
        self.down0 = nn.Sequential(
            nn.Conv2d(in_channels, first_channel, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(first_channel),
            nn.ReLU(inplace=True),
        )
        self.down1 = self.gen_down(first_channel, first_channel, 2, nblocks=5)
        self.down2 = self.gen_down(first_channel, first_channel * 2, 2, nblocks=5)
        self.down3 = self.gen_down(first_channel * 2, first_channel * 4, 2, nblocks=5)
        self.up0 = self.gen_up(first_channel * 4, first_channel * 2, 4)
        self.up1 = self.gen_up(first_channel * 2 + first_channel, first_channel * 2, 2)

        # initialize
        self.initialize_weights()

    def forward(self, x):
        x = self.down0(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up0(x4)
        x = torch.cat((x, x2), 1)
        x = self.up1(x)
        return x


class FPNStandard(nn.Module):

    def __init__(self,
                 num_class=1,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(64, 128, 256),
                 upsample_strides=(2, 4, 8),
                 num_upsample_filters=(128, 128, 128),
                 num_input_features=64,
                 name='FPNStandard'):

        super(FPNStandard, self).__init__()

        # output channels
        self.outc = sum(num_upsample_filters)

        # assert correct parameters passed
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)

        # upsampling strides (e.g. downsample factor relative to input dim)
        upsample_strides = [np.round(u).astype(np.int64) for u in upsample_strides]

        factors = []
        for i in range(len(layer_nums)):
            # ensure downsample + upsampling operation yields integer output dimension
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        # concatenation factor must be equal for all 3 blocks
        assert all([x == factors[0] for x in factors])

        # Conv Block 1
        modules = [nn.ZeroPad2d(1),
                   nn.Conv2d(num_input_features, num_filters[0], 3, stride=layer_strides[0], bias=False),
                   nn.BatchNorm2d(num_filters[0], 1e-3, 0.01),
                   nn.ReLU()]
        for i in range(layer_nums[0]):
            modules.append(nn.Conv2d(num_filters[0], num_filters[0], 3, padding=1, bias=False))
            modules.append(nn.BatchNorm2d(num_filters[0], 1e-3, 0.01))
            modules.append(nn.ReLU())
        self.block1 = Sequential(*modules)

        # Deconv Block 1
        self.deconv1 = Sequential(
            nn.ConvTranspose2d(num_filters[0],
                               num_upsample_filters[0],
                               upsample_strides[0],
                               stride=upsample_strides[0],
                               bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], 1e-3, 0.01),
            nn.ReLU(),
        )

        # Conv Block 2
        modules = [nn.ZeroPad2d(1),
                   nn.Conv2d(num_filters[0], num_filters[1], 3, stride=layer_strides[1], bias=False),
                   nn.BatchNorm2d(num_filters[1], 1e-3, 0.01),
                   nn.ReLU()]
        for i in range(layer_nums[1]):
            modules.append(nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1, bias=False))
            modules.append(nn.BatchNorm2d(num_upsample_filters[1], 1e-3, 0.01))
            modules.append(nn.ReLU())
        self.block2 = Sequential(*modules)

        # Deconv Block 2
        self.deconv2 = Sequential(
            nn.ConvTranspose2d(num_filters[1],
                               num_upsample_filters[1],
                               upsample_strides[1],
                               stride=upsample_strides[1],
                               bias=False),
            nn.BatchNorm2d(num_upsample_filters[1], 1e-3, 0.01),
            nn.ReLU(),
        )

        # Conv Block 3
        modules = [nn.ZeroPad2d(1),
                   nn.Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2], bias=False),
                   nn.BatchNorm2d(num_filters[2], 1e-3, 0.01),
                   nn.ReLU()]
        for i in range(layer_nums[2]):
            modules.append(nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1, bias=False))
            modules.append(nn.BatchNorm2d(num_filters[2], 1e-3, 0.01))
            modules.append(nn.ReLU())
        self.block3 = Sequential(*modules)

        # Deconv Block 3
        self.deconv3 = Sequential(
            nn.ConvTranspose2d(num_filters[2],
                               num_upsample_filters[2],
                               upsample_strides[2],
                               stride=upsample_strides[2],
                               bias=False),
            nn.BatchNorm2d(num_upsample_filters[2], 1e-3, 0.01),
            nn.ReLU(),
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        # compute convolutional feature maps
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        # concatenate multi-resolution feature maps
        x = torch.cat([up1, up2, up3], dim=1)
        # pass to decode network
        return x


class MobileNetV2(MobileNetV2):
    def __init__(self, in_channels, width_mult=1.0,
                 expand_ratio=(1, 6, 6, 6, 6, 6, 6),
                 num_filters=(16, 24, 32, 64, 96, 160, 320),
                 layer_nums=(1, 2, 3, 4, 3, 3, 1),
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 round_nearest=8,
                 input_channel=64,
                 last_channel=256
                 ):
        """
        MobileNet V2 main class
        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            expand ratio: used for hidden dimensions
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()

        block = InvertedResidual

        # ouput channels
        self.outc = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # assert corect parameters passed.
        assert len(expand_ratio) != 0
        assert len(num_filters) == len(expand_ratio)
        assert len(layer_nums) == len(expand_ratio)
        assert len(strides) == len(expand_ratio)

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        features = [ConvBNReLU(in_channels, input_channel, stride=2)]  # here, we are assuming 3 input channels.

        # building inverted residual blocks
        for i in range(len(expand_ratio)):
            output_channel = _make_divisible(num_filters[0] * width_mult, round_nearest)
            stride = strides[i]
            n = layer_nums[i]
            r = expand_ratio[i]
            for j in range(n):
                if j > 0:
                    stride = 1
                features.append(block(input_channel, output_channel, stride, r))
                input_channel = output_channel
        self.features = nn.Sequential(*features)
        # building last several layers
        self.up0 = self.gen_up(input_channel, input_channel // 2, 4)
        self.up1 = self.gen_up(input_channel // 2, input_channel // 2, 2)
        self.up2 = self.gen_up(input_channel // 2, self.outc, 2)

        # weight initialization
        self.initialize_weights()

    def gen_up(self, inchannels, outchannels, scale):

        pad = 1 if scale == 2 else 0
        o_pad = 1 if scale == 2 else scale - 3
        return nn.Sequential(
            nn.ConvTranspose2d(inchannels, outchannels, 3, scale, pad, o_pad, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        return x


# class MobileNetV3(MobileNetV3):
#     def __init__(self):
#         reduce_divider = 2  # or 1
#         dilation = 2  # or 1
#         width_mult = 1.0
#
#         bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
#         adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
#
#         inverted_residual_setting = [
#             # bneck_conf(input_channels: int,
#             #            kernel: int,
#             #            expanded_channels: int,
#             #            out_channels: int,
#             #            use_se: bool,
#             #            activation: str, HS=HardSwish, RE = Relu
#             #            stride: int,
#             #            dilation: int,
#             bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # 112
#             bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # 56
#             bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),  # 28
#             bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # 28
#             bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),  # 14
#             bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),  # 14
#             bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),  # 14
#             bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),  # 14
#             bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # 14
#             bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
#             # 7
#             bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
#             # 7
#         ]
#         last_channel = adjust_channels(1024 // reduce_divider)  ## CONV2d?
#
#         super(MobileNetV3, self).__init__(inverted_residual_setting, last_channel)
#         self.outc = 512


class MobileNetV3(nn.Module):

    def __init__(
            self,
            in_channel,
            model_type,
            out_channel=256,
            reduce_divider=False, dilated=False, width_mult=1.0,
            # inverted_residual_setting,
            block=InvertedResidual_v3,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
    ):
        """
        MobileNet V3 main class
        Args:
            model_type: "mobilenet_v3_large", "mobilenet_v3_small"
        """
        super(MobileNetV3, self).__init__()
        if model_type not in ["mobilenet_v3_large", "mobilenet_v3_small"]:
            raise ValueError("Unsupported model type. Supported model type: mobilenet_v3_large, mobilenet_v3_small")
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(model_type, reduce_divider, dilated, width_mult)
        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNReLU(in_channel, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                 activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 2 * lastconv_input_channels  # is 2 a good choice?
        layers.append(ConvBNReLU(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                 norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.outc = out_channel
        self.features = nn.Sequential(*layers)

        # upsampling layers, different upsampling scheme depending on the variable dilated.
        up = []
        up.append(self.gen_up(lastconv_output_channels, self.outc // 2, 4))
        if dilated is False:
            up.append(self.gen_up(self.outc // 2, self.outc // 2, 2))
        up.append(self.gen_up(self.outc // 2, self.outc, 2))
        self.up = nn.Sequential(*up)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def gen_up(self, inchannels, outchannels, scale):

        pad = 1 if scale == 2 else 0
        o_pad = 1 if scale == 2 else scale - 3
        return nn.Sequential(
            nn.ConvTranspose2d(inchannels, outchannels, 3, scale, pad, o_pad, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.up(x)
        return x


def _mobilenet_v3_conf(arch, reduce_divider=False, dilated=False, width_mult=1.0):
    """

    :param arch: One of the following strings: "mobilenet_v3_large", "mobilenet_v3_small"
    :param reduce_divider: True or False.
    :param dilated: True or False.
    :param width_mult: the default is 1.0, no width multiplication.
    :return:
    """
    # non-public config parameters
    if reduce_divider is True:
        reduce_divider = 2
    else:
        reduce_divider = 1
    if dilated is True:
        dilation = 2
    else:
        dilation = 1

    # use same width multi for all layers.
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    # bneck_conf(input_channel, kernel, expanded_channels, out_channels, use_se (bool), activation, stride, dilation)
    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return inverted_residual_setting, last_channel


class DarknetConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups=1):
        super(DarknetConv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leakyRelu(out)
        return out


class DarknetResidual(nn.Module):
    def __init__(self, in_channels):
        super(DarknetResidual, self).__init__()
        mid_channels = in_channels // 2
        self.layer1 = DarknetConv(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.layer2 = DarknetConv(mid_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        shortcut = x
        out = self.layer1(x)
        out = self.layer2(out)
        return out + shortcut


class Darknet53(nn.Module):
    def __init__(self, in_channel):
        super(Darknet53, self).__init__()
        self.in_count = in_channel
        self.conv1 = DarknetConv(in_ch=in_channel, out_ch=32, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv2 = DarknetConv(in_ch=32, out_ch=64, kernel_size=3, stride=2, padding=1, groups=1)
        self.block1 = self.make_block(in_channels=64, num_blocks=1)
        self.conv3 = DarknetConv(64, 128, kernel_size=3, stride=2, padding=1, groups=1)
        self.block2 = self.make_block(in_channels=128, num_blocks=2)
        self.conv4 = DarknetConv(128, 256, kernel_size=3, stride=2, padding=1, groups=1)
        self.block3 = self.make_block(in_channels=256, num_blocks=8)
        self.conv5 = DarknetConv(256, 512, kernel_size=3, stride=2, padding=1, groups=1)
        self.block4 = self.make_block(in_channels=512, num_blocks=8)
        self.conv6 = DarknetConv(512, 1024, kernel_size=3, stride=2, padding=1, groups=1)
        self.block5 = self.make_block(in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((160, 160))
        self.outc = 512

    def forward(self, x):
        # if self.in_count == 32:
        #     out = self.conv1(x)
        #     out = self.conv2(out)
        # elif self.in_count == 64:
        #     out = x
        out = self.block1(x)
        out = self.conv3(out)
        out = self.block2(out)
        # out = self.conv4(out)
        # out = self.block3(out)
        # out = self.conv5(out)
        # out = self.block4(out)
        # out = self.conv6(out)
        # out = self.block5(out)
        out = self.global_avg_pool(out)
        return out

    def make_block(self, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(DarknetResidual(in_channels))
        return nn.Sequential(*layers)

    def deconv(self, in_ch, out_ch, stride):
        return Sequential(
            nn.ConvTranspose2d(in_ch,
                               out_ch,
                               2,
                               stride=stride,
                               bias=False),
            nn.BatchNorm2d(128, 1e-3, 0.01),
            nn.ReLU(),
        )


def test_mobilev3():
    model = MobileNetV3(in_channel=64, model_type="mobilenet_v3_small")

    input = torch.randn(2, 64, 128, 128)
    print("mobilenet V3 input dim: ", input.shape)
    output = model(input)
    print("mobilenet V3 output dim: ", output.shape)


def test_mobilev2():
    mobile = MobileNetV2(64)
    input = torch.randn(2, 64, 128, 128)
    print("mobilenet V2 input dim: ", input.shape)
    output = mobile(input)
    print("mobilenet V2 output dim: ", output.shape)


def test_resnetv2():
    resnet = ResNetv2(in_channels=64)
    input = torch.randn(2, 64, 128, 128)
    print("input shape: ", input.shape)

    output = resnet.forward(input)
    print("output shape: ", output.shape)


if __name__ == "__main__":
    # test_resnet()
    # test_resnetv2()
    # test_fpns()
    # test_mobilev2()
    # test_mobilev3()
    pass
