"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonoasis_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import OASISResnetBlock as OASISResnetBlock


class OASISGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G="spectraloasissyncbatch3x3")
        parser.add_argument(
            "--channels_G",
            type=int,
            default=64,
            help="# of gen filters in first conv layer in generator",
        )

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        ch = opt.channels_G
        self.channles = [16 * ch, 16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch, 1 * ch]
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            # and concatenate it with the downsampled input at first layer
            self.fc = nn.Conv2d(
                self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1
            )
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)

        self.body = nn.Sequential()
        for i in range(len(self.channles) - 1):
            self.body.add_module(
                "resnetblock_%d" % i,
                OASISResnetBlock(self.channles[i], self.channles[i + 1], opt),
            )
        self.up = nn.Upsample(scale_factor=2)

        self.conv_img = nn.Conv2d(self.channles[-1], 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        sw = opt.crop_size // (2 ** len(self.channles) - 1)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def execute(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = jt.randn(input.size(0), self.opt.z_dim)
                z = z.view(z.size(0), self.opt.z_dim, 1, 1)
                z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            # we concatenate the input segmap with z
            seg = jt.contrib.concat([seg, z], dim=1)

            x = nn.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
        else:
            # we downsample segmap and run convolution
            x = nn.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        for i in range(len(self.channles) - 1):
            x = self.body[i](x, seg)
            if i != len(self.channles) - 1:
                x = self.up(x)

        x = self.conv_img(nn.leaky_relu(x, 2e-1))
        x = jt.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--resnet_n_downsample",
            type=int,
            default=4,
            help="number of downsampling layers in netG",
        )
        parser.add_argument(
            "--resnet_n_blocks",
            type=int,
            default=9,
            help="number of residual blocks in the global generator network",
        )
        parser.add_argument(
            "--resnet_kernel_size",
            type=int,
            default=3,
            help="kernel size of the resnet block",
        )
        parser.add_argument(
            "--resnet_initial_kernel_size",
            type=int,
            default=7,
            help="kernel size of the first convolution",
        )
        parser.set_defaults(norm_G="instance")
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = (
            opt.label_nc
            + (1 if opt.contain_dontcare_label else 0)
            + (0 if opt.no_instance else 1)
        )

        norm_layer = get_nonoasis_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [
            nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
            norm_layer(
                nn.Conv2d(
                    input_nc,
                    opt.ngf,
                    kernel_size=opt.resnet_initial_kernel_size,
                    padding=0,
                )
            ),
            activation,
        ]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [
                norm_layer(
                    nn.Conv2d(
                        opt.ngf * mult,
                        opt.ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),
                activation,
            ]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [
                ResnetBlock(
                    opt.ngf * mult,
                    norm_layer=norm_layer,
                    activation=activation,
                    kernel_size=opt.resnet_kernel_size,
                )
            ]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [
                norm_layer(
                    nn.ConvTranspose2d(
                        nc_in,
                        nc_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                ),
                activation,
            ]
            mult = mult // 2

        # final output conv
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def execute(self, input, z=None):
        return self.model(input)
