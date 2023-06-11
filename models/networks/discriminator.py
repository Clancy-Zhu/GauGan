"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonoasis_norm_layer

import util.util as util


# OASIS Discriminator
class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--netD_subarch",
            type=str,
            default="n_layer",
            help="architecture of each discriminator",
        )
        parser.add_argument(
            "--num_D",
            type=int,
            default=2,
            help="number of discriminators to be used in multiscale",
        )
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(
            opt.netD_subarch + "discriminator", "models.networks.discriminator"
        )
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        output_channel = opt.semantic_nc + 1
        self.channles = [3, 128, 128, 256, 256, 512, 512]

        self.sequential_encoder = nn.Sequential()
        self.sequential_decoder = nn.Sequential()

        for i in range(opt.num_res_blocks):
            subnetD = NLayerDiscriminator(
                self.channles[i], self.channles[i + 1], opt, -1, first=(i == 0)
            )
            self.sequential_encoder.add_module("discriminator_encoder_%d" % i, subnetD)
        subnetD = NLayerDiscriminator(self.channles[-1], self.channles[-2], opt, 1)
        self.sequential_decoder.add_module("discriminator_decoder_0", subnetD)
        for i in range(1, opt.num_res_blocks - 1):
            subnetD = NLayerDiscriminator(
                2 * self.channles[-i - 1], self.channles[-i - 2], opt, 1
            )
            self.sequential_decoder.add_module("discriminator_decoder_%d" % i, subnetD)
        subnetD = NLayerDiscriminator(2 * self.channles[1], 64, opt, 1)
        self.sequential_decoder.add_module(
            "discriminator_decoder_%d" % (opt.num_res_blocks - 1), subnetD
        )
        self.layer_last = nn.Conv2d(
            64, output_channel, kernel_size=1, stride=1, padding=0
        )

    def downsample(self, input):
        # import ipdb
        # ipdb.set_trace()
        return nn.avg_pool2d(
            input, kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def execute(self, input):
        x = input
        # encoder
        encoder_res = []
        for i in range(len(self.sequential_encoder)):
            x = self.sequential_encoder[i](x)
            encoder_res.append(x)
        # decoder
        x = self.sequential_decoder[0](x)
        for i in range(1, len(self.sequential_decoder)):
            x = self.sequential_decoder[i](jt.concat([encoder_res[-i - 1], x], dim=1))
        # last layer
        ans = self.layer_last(x)
        return ans


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--n_layers_D", type=int, default=4, help="# layers in each discriminator"
        )
        return parser

    def __init__(self, fin, fout, opt, up_or_down, first=False):
        super().__init__()
        self.opt = opt
        norm_layer = get_nonoasis_norm_layer(opt, opt.norm_E)

        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = fin != fout
        fmiddle = fout

        self.conv1 = nn.Sequential()
        if first:
            self.conv1.add_module(
                "conv1",
                norm_layer(nn.Conv2d(fin, fmiddle, kernel_size=3, stride=1, padding=1)),
            )
        else:
            if self.up_or_down == 1:
                self.conv1.add_module(
                    "leakyrelu1",
                    nn.LeakyReLU(0.2),
                )
                self.conv1.add_module(
                    "upsample1",
                    nn.Upsample(scale_factor=2, mode="nearest"),
                )
                self.conv1.add_module(
                    "conv1",
                    norm_layer(
                        nn.Conv2d(fin, fmiddle, kernel_size=3, stride=1, padding=1),
                    ),
                )
            else:
                self.conv1.add_module(
                    "leakyrelu1",
                    nn.LeakyReLU(0.2),
                )
                self.conv1.add_module(
                    "conv1",
                    norm_layer(
                        nn.Conv2d(fin, fmiddle, kernel_size=3, stride=1, padding=1),
                    ),
                )
        self.conv2 = nn.Sequential()
        self.conv2.add_module(
            "leakyrelu2",
            nn.LeakyReLU(0.2),
        )
        self.conv2.add_module(
            "conv2",
            norm_layer(nn.Conv2d(fmiddle, fout, kernel_size=3, stride=1, padding=1)),
        )
        if self.learned_shortcut:
            self.conv_s = norm_layer(
                nn.Conv2d(fin, fout, kernel_size=1, stride=1, padding=0)
            )
        if up_or_down == 1:
            self.sample = nn.Upsample(scale_factor=2, mode="nearest")
        if up_or_down == -1:
            self.sample = nn.AvgPool2d(kernel_size=2)
        if up_or_down == 0:
            self.sample = nn.Identity()

    def shortcut(self, x):
        if self.first:
            if self.up_or_down == -1:
                x = self.sample(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down == 1:
                x = self.sample(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down == -1:
                x = self.sample(x)
            x_s = x
        return x_s

    def execute(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down == -1:
            dx = self.sample(dx)
        out = x_s + dx
        return out
