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
            "--num_upsampling_layers",
            choices=("normal", "more", "most"),
            default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator",
        )

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = OASISResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = OASISResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = OASISResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = OASISResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = OASISResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = OASISResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = OASISResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == "most":
            self.up_4 = OASISResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == "normal":
            num_up_layers = 5
        elif opt.num_upsampling_layers == "more":
            num_up_layers = 6
        elif opt.num_upsampling_layers == "most":
            num_up_layers = 7
        else:
            raise ValueError(
                "opt.num_upsampling_layers [%s] not recognized"
                % opt.num_upsampling_layers
            )

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def execute(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = jt.randn(input.size(0), self.opt.z_dim)
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = nn.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if (
            self.opt.num_upsampling_layers == "more"
            or self.opt.num_upsampling_layers == "most"
        ):
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(nn.leaky_relu(x, 2e-1))
        x = jt.tanh(x)

        return x
