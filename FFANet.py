import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from attention import PAM_Module
from attention import CAM_Module


__all__ = ['VoVNet', 'vovnet27_slim', 'vovnet39', 'vovnet57']


model_urls = {
    'vovnet39': 'https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1',
    'vovnet57': 'https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1'
}


def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        #sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        #sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        #output = [sasc_output]
        #output.append(sa_output)
        #output.append(sc_output)
        return sasc_output


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=True):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        self.eca = eca_layer(in_ch)
        self.concat2 = nn.Sequential(
            OrderedDict(conv1x1(in_ch, concat_ch, module_name, 'concat')))


        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))


    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)

        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.eca(xt)
        identity_feat = self.concat2(identity_feat)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name,
            _OSA_module(in_ch,
                        stage_ch,
                        concat_ch,
                        layer_per_block,
                        module_name))
        for i in range(block_per_stage-1):
            module_name = f'OSA{stage_num}_{i+2}'
            self.add_module(module_name,
                _OSA_module(concat_ch,
                            stage_ch,
                            concat_ch,
                            layer_per_block,
                            module_name,
                            identity=True))


class VoVNet(nn.Module):
    def __init__(self, 
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 num_classes=5):
        super(VoVNet, self).__init__()
        self.num_classes = num_classes

        # Stem module
        stem = conv3x3(1,   64, 'stem', '1', 2)
        stem += conv3x3(64,  64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4): #num_stages
            name = 'stage%d' % (i+2)
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block,
                                       i+2))

        #self.classifier = nn.Linear(config_concat_ch[-1], num_classes)


        self.convcat = nn.Conv2d(2560, 256, kernel_size=1)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)

        #final conv
        self.semantic_branch = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        # self.gn1 = nn.GroupNorm(128, 128)
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(64, 64)
        self.danet = DANetHead(384, num_classes, nn.BatchNorm2d)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


    def forward(self, x):
        x0 = self.stem(x)
        out = []
        x1 = x0


        for name in self.stage_names:
            x0 = getattr(self, name)(x0)
            out.append(x0)

        '''out1 = self.attention(256, out[0])#64*64,128
        out2 = self.attention(512, out[1])#32*32,256
        out3 = self.attention(768, out[2])#16*16,768
        out4 = self.attention(1024, out[3])#8*8,1024'''

        out1 = out[0]
        out2 = out[1]
        out3 = out[2]
        out4 = out[3]


        _, _, H, W = x.size()
        _, _, h0, w0 = x1.size()
        _, _, h1, w1 = out1.size()

        up2 = self.upsample(out2, h1, w1)
        up3 = self.upsample(out3, h1, w1)
        up4 = self.upsample(out4, h1, w1)

        out_cat = torch.cat([out1, up2, up3, up4], dim=1)

        fuse_feature = self.convcat(out_cat)# channle = 256
        #final_up = self.upsample(fuse_feature, h0, w0)
        final_cat = torch.cat([x1, fuse_feature], dim=1)
        s = self.danet(final_cat)

        '''s5 = self.upsample(F.relu(self.gn2(self.conv2(s))), h0, w0)
        # sDA5 = self._upsample(F.relu(self.gn2(self.conv2(DA5))), h, w)
        # 256->256
        s5 = self.upsample(F.relu(self.gn2(self.conv2_1(s5))), h0, w0)
        # sDA5 = self._upsample(F.relu(self.gn2(self.conv2(sDA5))), h, w)
        # 256->128
        s5 = self.upsample(F.relu(self.gn1(self.semantic_branch(s5))), h0, w0)
        # sDA5 = self._upsample(F.relu(self.gn1(self.semantic_branch(sDA5))), h, w)
        s5 = self.upsample(self.conv3(s5), H, W)'''

        s = self.upsample(s, H, W)

        #x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        #x = self.classifier(x)
        return s


def _vovnet(arch,
            config_stage_ch,
            config_concat_ch,
            block_per_stage,
            layer_per_block,
            pretrained,
            progress,
            **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch,
                   block_per_stage, layer_per_block,
                   **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def FFANet(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-57 model as described in 
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet57', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,4,3], 5, pretrained, progress, **kwargs)


def vovnet39(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet39', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,2,2], 5, pretrained, progress, **kwargs)


def vovnet27_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet27_slim', [64, 80, 96, 112], [128, 256, 384, 512],
                    [1,1,1,1], 5, pretrained, progress, **kwargs)
