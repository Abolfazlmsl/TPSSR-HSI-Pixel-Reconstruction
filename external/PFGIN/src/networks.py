import torch
import torch.nn as nn
from kpn.network import KernelConv, KernelConv_SS,KernelConv_Spectral
import kpn.utils as kpn_utils
import numpy as np


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()
        self.filter_type = config.FILTER_TYPE
        self.kernel_size = config.kernel_size
        self.in_chans = config.in_chs
        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.in_chans, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        self.kernel_pred = KernelConv(kernel_size=[config.kernel_size[0]], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator(config)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        inputs = x.clone()
        x = self.encoder0(x) # 64*256*256
        x = self.encoder1(x) # 128*128*128

        kernels, kernels_img = self.kpn_model(inputs, x)
        # inputs:[B C H W] x:[B 128 H//2,W//2]
        print(kernels.shape,kernels_img.shape)
        # kernels :[B 256*K_spa^2*K_spe H//4, W//4]  'kernel for middel  '
        # Kernel_imgs:[B C//2*K_spa^2*K_spe H W] :kernel for fine tune
        x = self.encoder2(x) # 256*64*64
        print('s1',      x.shape,kernels.shape)
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)
        # x:[B 256 H//4,W//4] feature map ; kernels middle
        x = self.middle(x) # 256*64*64
        x = self.decoder(x) # 3*256*256
        print('s2,  ',x.shape,kernels_img.shape)
        # x: [B in//2 H W] final feature map; kernel_imgs final kernel up
        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        x = (torch.tanh(x) + 1) / 2

        return x

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)



class InpaintGenerator_SS(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True,in_chs = None):
        super(InpaintGenerator_SS, self).__init__()
        self.filter_type = config.FILTER_TYPE
        self.kernel_size = config.kernel_size
        if in_chs is None:
            self.in_chans = config.in_chs
        else:
            self.in_chans = in_chs
        self.out_chas = self.in_chans//2

        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.in_chans, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=self.out_chas, kernel_size=7, padding=0),
        )

        self.kernel_pred1 = KernelConv_SS(Kernel_size=[self.kernel_size[0],self.kernel_size[1]], sep_conv=False, core_bias=False)
        self.kernel_pred2 = KernelConv_SS (Kernel_size=[self.kernel_size[2],self.kernel_size[3]], sep_conv=False, core_bias=False)
        self.kpn_model = kpn_utils.create_generator_ss(Kernel_size=self.kernel_size,in_chs=self.in_chans,inde=False)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        inputs = x.clone()
        x = self.encoder0(x) # 64*256*256
        x = self.encoder1(x) # 128*128*128

        kernels, kernels_img = self.kpn_model(inputs, x)
        # inputs:[B C H W] x:[B 128 H//2,W//2]

        # kernels :[B 256*K_spa^2*K_spe H//4, W//4]  'kernel for middel  '
        # Kernel_imgs:[B C//2*K_spa^2*K_spe H W] :kernel for fine tune
        x = self.encoder2(x) # 256*64*64

        x = self.kernel_pred1(x, kernels, white_level=1.0, rate=1)
        # x:[B 256 H//4,W//4] feature map ; kernels middle
        x = self.middle(x) # 256*64*64
        x = self.decoder(x) # 3*256*256

        # x: [B in//2 H W] final feature map; kernel_imgs final kernel up
        x = self.kernel_pred2(x, kernels_img, white_level=1.0, rate=1)
        x = (torch.tanh(x) + 1) / 2

        return x

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)


class InpaintGenerator_s_s(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True,in_chs = None):
        super(InpaintGenerator_s_s, self).__init__()
        self.filter_type = config.FILTER_TYPE
        self.kernel_size = config.kernel_size
        if in_chs is None:
            self.in_chans = config.in_chs
        else:
            self.in_chans = in_chs
        self.out_chas = self.in_chans//2

        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.in_chans, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=self.out_chas, kernel_size=7, padding=0),
        )

        self.kernel_pred_1 = KernelConv_SS(Kernel_size=self.kernel_size, sep_conv=False, core_bias=False)
        self.kernel_pred_2 = KernelConv_Spectral (Kernel_size=self.kernel_size[1], sep_conv=False, core_bias=False)
        self.kpn_model = kpn_utils.create_generator_ss(Kernel_size=self.kernel_size,in_chs=self.in_chans,inde=True)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        inputs = x.clone()
        x = self.encoder0(x) # 64*256*256
        x = self.encoder1(x) # 128*128*128

        kernels, kernels_img = self.kpn_model(inputs, x)
        # inputs:[B C H W] x:[B 128 H//2,W//2]

        # kernels :[B 256*K_spa^2*K_spe H//4, W//4]  'kernel for middel  '
        # Kernel_imgs:[B C//2*K_spa^2*K_spe H W] :kernel for fine tune
        x = self.encoder2(x) # 256*64*64

        x = self.kernel_pred_1(x, kernels, white_level=1.0, rate=1)
        # x:[B 256 H//4,W//4] feature map ; kernels middle
        x = self.middle(x) # 256*64*64
        x = self.decoder(x) # 3*256*256

        # x: [B in//2 H W] final feature map; kernel_imgs final kernel up
        x = self.kernel_pred_2(x, kernels_img, white_level=1.0, rate=1)
        x = (torch.tanh(x) + 1) / 2

        return x

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.out_chans = in_channels//2

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Discriminator2(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator2, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.out_chans = in_channels//2

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


if __name__ == '__main__':
    import argparse

    import os
    import cv2
    import random
    import numpy as np
    import torch
    import argparse
    from shutil import copyfile
    from src.config import Config
    from thop import profile
    import torch.nn as nn
    def load_config(mode=None):

        parser = argparse.ArgumentParser ( )
        parser.add_argument ('--path', '--checkpoints', type=str, default='./checkpoints',
                             help='model checkpoints path (default: ./checkpoints)')

        args = parser.parse_args ( )
        config_path = os.path.join (args.path, 'config.yml')

        if not os.path.exists (args.path):
            os.makedirs (args.path)

        config = Config ('F:\\work\\paper3\\GAN\\misf-main\\checkpoints\\config.yml')

        # train mode
        if mode == None or 1:
            config.MODE = 1

        # test mode
        elif mode == 2:
            config.MODE = 2
        return config
    config = load_config()
    print(config.kernel_size[3])
    gen =InpaintGenerator_SS(config=config).cuda()
    print(config.kernel_size,config.in_chs)
    a = torch.randn (4, config.in_chs, 256, 256).cuda ( )
    aa = gen(a)
    print(a.shape)
    # gen2 = InpaintGenerator_SS(config=config).cuda()
    # flops_ssif,para_SSif =profile(gen2,inputs=(a,))
    # flops_ssf, para_ssf = profile(gen,inputs=(a,))
    # print('SSIFpara.{}'.format(para_SSif),'    SSIFflops:{}'.format(flops_ssif))
    # print('SSFpara.{}'.format(para_ssf),'    SSFflops:{}'.format(flops_ssf))


