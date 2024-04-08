import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ChannelCompress(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = 48
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(num_bottleneck, out_ch)]
        add_block += [nn.BatchNorm1d(out_ch)]
        # add_block += [nn.ReLU()]
        # add_block += [nn.Linear(500, out_ch)]
        # add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x


# class ChannelCompress(nn.Module):
#     def __init__(self, in_ch=2048, out_ch=256):
#         """
#         reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
#         out_ch could be 512, 256, 128
#         """
#         super(ChannelCompress, self).__init__()
#         num_bottleneck = 1000
#         add_block = []
#         add_block += [nn.Linear(in_ch, num_bottleneck)]
#         add_block += [nn.BatchNorm1d(num_bottleneck)]
#         add_block += [nn.ReLU()]
#         add_block += [nn.Linear(num_bottleneck, 500)]
#         add_block += [nn.BatchNorm1d(500)]
#         add_block += [nn.ReLU()]
#         add_block += [nn.Linear(500, out_ch)]
#         add_block += [nn.BatchNorm1d(out_ch)]
#
#         add_block = nn.Sequential(*add_block)
#         add_block.apply(weights_init_kaiming)
#         self.model = add_block
#
#     def forward(self, x):
#         x = self.model(x)
#         return x


class VIB(nn.Module):
    def __init__(self, in_ch=2048, z_dim=256, num_class=395):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB, maybe modified later.
        # classifier = []
        # classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        # classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        # classifier += [nn.LeakyReLU(0.1)]
        # classifier += [nn.Dropout(0.5)]
        # classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        # classifier = nn.Sequential(*classifier)
        # self.classifier = classifier
        self.classifier = nn.Linear(self.out_ch, self.num_class)
        self.classifier.apply(weights_init_classifier)

    def forward(self, v):
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return p_y_given_z, z_given_v