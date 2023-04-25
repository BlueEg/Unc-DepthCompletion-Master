import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from model.Resnet import make_layer


class ResBlock_CH(nn.Module):
    def __init__(self, inplanes, planes, stride, kernel):
        super(ResBlock_CH, self).__init__()
        if kernel == 3:
            padding = 1
        elif kernel == 7:
            padding = 3
        elif kernel == 9:
            padding = 4
        elif kernel == 5:
            padding = 2
        else:
            padding = 0

        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel, 1, padding, bias=False),
                                   nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel, 1, padding, bias=False))

        self.ds = nn.Sequential(nn.Conv2d(inplanes, planes, kernel, stride, padding, bias=False))
        init_weights(self.conv1)
        init_weights(self.conv2)
        init_weights(self.ds)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.ds(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, inplanes, kernel):
        super(ResBlock, self).__init__()
        if kernel == 3:
            padding = 1
        elif kernel == 7:
            padding = 3
        elif kernel == 9:
            padding = 4
        elif kernel == 5:
            padding = 2
        else:
            padding = 0

        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel, 1, padding, bias=False),
                                   nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel, 1, padding, bias=False))
        init_weights(self.conv1)
        init_weights(self.conv2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=False, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=False))
    layers = nn.Sequential(*layers)

    for m in layers.modules():
        init_weights(m)

    return layers


def convt_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bn=False, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    for m in layers.modules():
        init_weights(m)

    return layers


class CompletionBlock(nn.Module):
    def __init__(self, inplanes, outplanes, Ucert=True):
        super(CompletionBlock, self).__init__()
        self.Ucert = Ucert
        self.enconv0 = nn.Sequential(conv_bn_relu(inplanes, 32, 9, stride=1, padding=4, bn=False, relu=False),
                                     conv_bn_relu(32, 64, 7, stride=1, padding=3, bn=False, relu=True), ResBlock(64, 3))

        pretrained_model = resnet.__dict__['resnet34'](pretrained=False)
        self.enconv1 = pretrained_model._modules['layer2']
        self.enconv2 = pretrained_model._modules['layer3']
        del pretrained_model  # clear memory

        self.middle = ResBlock(256, 3)
        self.SDconv0 = nn.Sequential(ResBlock_CH(1, 32, 1, 7))
        self.SDconv1 = nn.Sequential(ResBlock_CH(32, 32, 1, 5))
        self.SDconv2 = nn.Sequential(ResBlock_CH(32, 64, 2, 3))
        self.SDconv3 = nn.Sequential(ResBlock_CH(64, 128, 2, 3))

        self.RGBconv0 = nn.Sequential(ResBlock_CH(3, 32, 1, 7))
        self.RGBconv1 = nn.Sequential(ResBlock_CH(32, 32, 1, 5))
        self.RGBconv2 = nn.Sequential(ResBlock_CH(32, 64, 2, 3))
        self.RGBconv3 = nn.Sequential(ResBlock_CH(64, 128, 2, 3))

        self.deRes3 = nn.Sequential(ResBlock(128, 3))
        self.deRes2 = nn.Sequential(ResBlock(64, 3))

        self.mix1 = conv_bn_relu(64 + 32, 64, 3, stride=1, padding=1, bn=False, relu=False)
        self.mix2 = conv_bn_relu(128 + 64, 128, 3, stride=1, padding=1, bn=False, relu=False)
        self.mix3 = conv_bn_relu(256 + 128, 256, 3, stride=1, padding=1, bn=False, relu=False)

        self.deconv3 = convt_bn_relu(in_channels=256 + 256 + 128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1)
        self.deconv2 = convt_bn_relu(in_channels=256 + 64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1)

        self.deconv1 = conv_bn_relu(128 + 32, 64, 3, 1, 1, bn=False, relu=False)

        if self.Ucert == True:
            self.deconvS = nn.Sequential(ResBlock(64, 3), conv_bn_relu(64, 16, 3, 1, 1, bn=False, relu=False),
                                         conv_bn_relu(16, outplanes, 3, 1, 1, bn=False, relu=True))

        self.deconv0 = nn.Sequential(ResBlock(64, 3), conv_bn_relu(64, 32, 3, 1, 1, bn=False, relu=False),
                                     conv_bn_relu(32, outplanes, 3, 1, 1, bn=False, relu=False))

    def forward(self, x, rgb, sd):
        # SD Feature
        conv0_SD = self.SDconv0(sd)
        conv1_SD = self.SDconv1(conv0_SD)  # 32
        conv2_SD = self.SDconv2(conv1_SD)  # 64
        conv3_SD = self.SDconv3(conv2_SD)  # 128

        # RGB Feature
        conv0_RGB = self.RGBconv0(rgb)
        conv1_RGB = self.RGBconv1(conv0_RGB)  # 32
        conv2_RGB = self.RGBconv2(conv1_RGB)  # 64
        conv3_RGB = self.RGBconv3(conv2_RGB)  # 128

        # Encoder
        conv0_en = self.enconv0(x)
        conv0_en_1 = self.mix1(torch.cat((conv0_en, conv1_RGB), 1))

        conv2_en = self.enconv1(conv0_en_1)
        conv2_en_1 = self.mix2(torch.cat((conv2_en, conv2_SD), 1))

        conv3_en = self.enconv2(conv2_en_1)
        conv3_en_1 = self.mix3(torch.cat((conv3_en, conv3_RGB), 1))

        conv3_en_1 = self.middle(conv3_en_1)

        # decoder
        deconv3 = self.deconv3(torch.cat((conv3_en_1, conv3_en, conv3_SD), 1))
        deconv3 = self.deRes3(deconv3)
        deconv2 = self.deconv2(torch.cat((deconv3, conv2_en, conv2_RGB), 1))
        deconv2 = self.deRes2(deconv2)
        deconv1 = self.deconv1(torch.cat((deconv2, conv0_en, conv1_SD), 1))
        deconv0 = self.deconv0(deconv1)

        if self.Ucert == True:
            deconvS = self.deconvS(deconv1)

        if self.Ucert == True:
            return deconv0, deconvS
        else:
            return deconv0


class MDCnet(nn.Module):
    def __init__(self):
        super(MDCnet, self).__init__()
        self.encoder_1 = CompletionBlock(2, 1)
        self.encoder_2 = CompletionBlock(2, 1)
        self.encoder_3 = CompletionBlock(2, 1)
        self.encoder_4 = CompletionBlock(2, 1)

        self.dwnscale = nn.Sequential(conv_bn_relu(1, 1, 9, stride=2, padding=4, bn=False, relu=False),
                                      conv_bn_relu(1, 1, 7, stride=2, padding=3, bn=False, relu=False),
                                      conv_bn_relu(1, 1, 3, stride=2, padding=1, bn=False, relu=False))

        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        dpth_1 = x['d']
        dpth_2 = self.pool(dpth_1)
        dpth_4 = self.pool(dpth_2)
        dpth_8 = self.pool(dpth_4)

        dpth_8_conv = self.dwnscale(x['d'])

        # scale_1/8
        out_1, s_1 = self.encoder_1(torch.cat((dpth_8_conv, dpth_8), 1), x['rgb_8'], dpth_8)
        out_1_1 = F.upsample(out_1, (out_1.size()[2] * 2, out_1.size()[3] * 2), mode='bicubic')

        # scale_1/4
        out_2, s_2 = self.encoder_2(torch.cat((out_1_1, dpth_4), 1), x['rgb_4'], dpth_4)
        out_2_1 = F.upsample(out_2, (out_2.size()[2] * 2, out_2.size()[3] * 2), mode='bicubic')

        out_3, s_3 = self.encoder_3(torch.cat((out_2_1, dpth_2), 1), x['rgb_2'], dpth_2)
        out_3_1 = F.upsample(out_3, (out_3.size()[2] * 2, out_3.size()[3] * 2), mode='bicubic')

        out_4, s_4 = self.encoder_4(torch.cat((out_3_1, dpth_1), 1), x['rgb_1'], dpth_1)

        if self.training:
            return 100 * out_1, 100 * out_2, 100 * out_3, 100 * out_4, s_1, s_2, s_3, s_4
        else:
            min_distance = 0.9
            return F.relu(out_4 * 100 - min_distance) + min_distance, s_4


class AIRnet(nn.Module):
    def __init__(self):
        super(AIRnet, self).__init__()
        self.RESnet = ResNet_L1()

    def forward(self, x, sd):
        Res = self.RESnet(x, sd)
        out = Res + x
        return Res, out

class ResNet_B(nn.Module):
    def __init__(self):
        super(ResNet_B, self).__init__()


        self.enconv0 = nn.Sequential(conv_bn_relu(4, 16, 3, stride=1, padding=1, bn=False, relu=False),conv_bn_relu(16, 64, 3, stride=1, padding=1, bn=False, relu=True),ResBlock(64,3))

        self.enconv1 = make_layer(64,128,2,2)
        self.enconv2 = make_layer(128+ 64,256,3,1)

        # self.SDconv0 = nn.Sequential(ResBlock_CH(3,32,1,3))
        # self.SDconv1 = nn.Sequential(ResBlock_CH(32,32,1,3))
        # self.SDconv2 = nn.Sequential(ResBlock_CH(32, 64, 2,3))
        # self.SDconv3 = nn.Sequential(ResBlock_CH(64, 128, 1,3))

        self.SDconv0 = nn.Sequential(conv_bn_relu(1, 8, 9, stride=1, padding=4, bn=False, relu=False),conv_bn_relu(8, 32, 5, stride=1, padding=2, bn=False, relu=False),ResBlock(32,3))
        self.SDconv1 = nn.Sequential(conv_bn_relu(32, 64, 3, stride=2, padding=1, bn=False, relu=False),ResBlock(64,3))
        self.SDconv2 = nn.Sequential(conv_bn_relu(64, 128, 3, stride=1, padding=1, bn=False, relu=False),ResBlock(128,3))

        # self.RGBconv0 = nn.Sequential(conv_bn_relu(3, 8, 9, stride=1, padding=4, bn=False, relu=False),conv_bn_relu(8, 16, 5, stride=1, padding=2, bn=False, relu=False))
        # self.RGBconv1 = nn.Sequential(conv_bn_relu(16, 32, 3, stride=1, padding=1, bn=False, relu=False),ResBlock(32,3))
        # self.RGBconv2 = nn.Sequential(conv_bn_relu(32, 64, 3, stride=2, padding=1, bn=False, relu=False),ResBlock(64,3))

        #self.deRes3 = conv_bn_relu(256, 128, 3, stride=1, padding=1, bn=False, relu=False)

        self.deconv2  = convt_bn_relu(in_channels=256 + 128 + 128,
                                    out_channels= 256,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.deRes2 = ResBlock(256,3)
        self.deconv1  = nn.Conv2d(256  + 32,  128, 3, 1, 1,bias = False)
        self.deconv0 = nn.Sequential(ResBlock(128,3), nn.Conv2d(128,64,3,1,1,bias = False),nn.Conv2d(64,16,3,1,1,bias = False), nn.Conv2d(16,1,3,1,1,bias = False))

    def forward(self, x,sd):

        # conv0_RGB = self.RGBconv0(sd['rgb_1'])  #32
        # conv1_RGB = self.RGBconv1(conv0_RGB)   #64
        # conv2_RGB = self.RGBconv2(conv1_RGB)   #128
        #encoder
        conv0_SD = self.SDconv0(sd['d'])  #32
        conv1_SD = self.SDconv1(conv0_SD)   #64
        conv2_SD = self.SDconv2(conv1_SD)   #128

        conv0_en = self.enconv0(torch.cat((x,sd['rgb_1']),1))

        conv1_en = self.enconv1(conv0_en)

        conv2_en = self.enconv2(torch.cat((conv1_en,conv1_SD),1))

        #deconv3 = self.deRes3(conv2_en)

        deconv2 = self.deconv2(torch.cat((conv2_en,conv1_en, conv2_SD), 1))
        deconv2 = self.deRes2(deconv2)
        deconv1 = self.deconv1(torch.cat((deconv2,conv0_SD), 1))
        out = self.deconv0(deconv1)

        return out

class ResNet_L1(nn.Module):
    def __init__(self):
        super(ResNet_L1, self).__init__()

        self.enconv0 = nn.Sequential(conv_bn_relu(4, 16, 3, stride=1, padding=1, bn=False, relu=False),
                                     conv_bn_relu(16, 64, 3, stride=1, padding=1, bn=False, relu=True), ResBlock(64, 3))
        self.enconv1 = make_layer(64, 128, 2, 2)
        self.enconv2 = make_layer(128 + 64, 128, 3, 1)

        self.SDconv0 = nn.Sequential(conv_bn_relu(1, 8, 9, stride=1, padding=4, bn=False, relu=False),
                                     conv_bn_relu(8, 32, 5, stride=1, padding=2, bn=False, relu=False), ResBlock(32, 3))
        self.SDconv1 = nn.Sequential(conv_bn_relu(32, 64, 3, stride=2, padding=1, bn=False, relu=False),
                                     ResBlock(64, 3))
        self.SDconv2 = nn.Sequential(conv_bn_relu(64, 128, 3, stride=1, padding=1, bn=False, relu=False),
                                     ResBlock(128, 3))

        self.deconv2 = convt_bn_relu(in_channels=128 + 128 + 128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1)
        self.deRes2 = ResBlock(128, 3)
        self.deconv1 = nn.Conv2d(128 + 32, 64, 3, 1, 1, bias=False)
        self.deconv0 = nn.Sequential(ResBlock(64, 3), nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                                     nn.Conv2d(32, 16, 3, 1, 1, bias=False), nn.Conv2d(16, 1, 3, 1, 1, bias=False))

    def forward(self, x, sd):
        # encoder
        conv0_SD = self.SDconv0(sd['d'])  # 32
        conv1_SD = self.SDconv1(conv0_SD)  # 64
        conv2_SD = self.SDconv2(conv1_SD)  # 128

        conv0_en = self.enconv0(torch.cat((x, sd['rgb_1']), 1))
        conv1_en = self.enconv1(conv0_en)
        conv2_en = self.enconv2(torch.cat((conv1_en, conv1_SD), 1))

        deconv2 = self.deconv2(torch.cat((conv2_en, conv1_en, conv2_SD), 1))
        deconv2 = self.deRes2(deconv2)
        deconv1 = self.deconv1(torch.cat((deconv2, conv0_SD), 1))
        out = self.deconv0(deconv1)

        return out