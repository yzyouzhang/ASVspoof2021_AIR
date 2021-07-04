import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np
from pytorch_model_summary import summary

## Adapted from https://github.com/joaomonteirof/e2e_antispoofing
## https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/newfunctions/


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        #self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0)==1:
            attentions = F.softmax(torch.tanh(weights),dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5*torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = torch.cat((avg_repr,std_repr),1)

            return representations

class ConvNet(nn.Module):
    def __init__(self, num_classes=2, num_nodes=512, enc_dim=2, subband_attention=False):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 2), stride=(2, 3), bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(1, 2), dilation=(1, 1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(num_nodes, 3), padding=(0, 1), dilation=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(num_nodes, 256)
        self.fc2 = nn.Linear(256, enc_dim)
        self.fc3 = nn.Linear(enc_dim, num_classes)
        # self.dropout = nn.Dropout(0.2)

        self.subband_attention = subband_attention
        if self.subband_attention:
            self.attention = SelfAttention(128)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        # print(h.shape)
        if self.subband_attention:
            # print(h.shape)
            h = self.layer5(h)
            h = h.squeeze(2)
            # print(h.shape)
            h = self.attention(h.permute(0, 2, 1).contiguous())
            # print(h.shape)
            out = h
        else:
            h = h.reshape(h.size(0), -1)
            out = self.fc1(h)
        # h = self.dropout(h)
        out1 = self.fc2(out)
        out = self.fc3(out1)

        return out1, out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }

class ResNet(nn.Module):
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2):
        self.in_planes = 16
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = SelfAttention(256)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.activation(self.bn5(x)).squeeze(2)

        stats = self.attention(x.permute(0, 2, 1).contiguous())

        feat = self.fc(stats)

        mu = self.fc_mu(feat)

        return feat, mu


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=1000, loss='softmax', **kwargs):
        self.inplanes = 16
        super(Res2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])#64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)#128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)#256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)#512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Linear(128*block.expansion, num_classes)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale))

        return nn.Sequential(*layers)

    def _forward(self, x):
        #x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print('maxpool: ', x.size())

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        feat = x
        # print('flatten: ', x.size())
        x = self.cls_layer(x)
        out = F.log_softmax(x, dim=-1)
        
        return feat, out

    def extract(self, x):
        # x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        return x
    # Allow for accessing forward method in a inherited class
    forward = _forward


def se_res2net50_v1b(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model


class SEBottle2neck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale=4,
                 stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(SEBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=16)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        #print('x: ', x.size())
        out = self.conv1(x)
        #print('conv1: ', out.size())
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        #print('conv2: ', out.size())
        out = self.conv3(out)
        #print('conv3: ', out.size())
        out = self.bn3(out)
        out = self.se(out)
        #print('se :', out.size())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # print('se reduction: ', reduction)
        # print(channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):   # x: B*C*D*T
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)    
    
class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)
    MaxFeatureMap2D(max_dim=1)
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super(MaxFeatureMap2D, self).__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


class LCNN(nn.Module):
    def __init__(self, num_nodes, enc_dim, nclasses=2):
        super(LCNN, self).__init__()
        self.num_nodes = num_nodes
        self.enc_dim = enc_dim
        self.nclasses = nclasses
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, (5, 5), 1, padding=(2, 2)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv5 = nn.Sequential(nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(64, affine=False))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv8 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv9 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 1, padding=[1, 1]),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.out = nn.Sequential(nn.Dropout(0.7),
                                 nn.Linear((750 // 16) * (num_nodes // 16) * 32, 160),
                                 MaxFeatureMap2D(),
                                 nn.Linear(80, self.enc_dim))
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        feat = torch.flatten(x, 1)
        feat = self.out(feat)
        out = self.fc_mu(feat)

        return feat, out

class Subband(nn.Module):
    def __init__(self, num_nodes=512, enc_dim=2, num_classes=2, subband_num=4):
        super(Subband, self).__init__()
        self.subband_num = subband_num
        self.enc_dim = enc_dim
        for i in range(self.subband_num):
            if i == 0:
                setattr(self,
                        'sub'+str(i+1),
                        LCNN(num_nodes, enc_dim // subband_num + enc_dim % subband_num, nclasses=num_classes)
                        )
            else:
                setattr(self,
                        'sub' + str(i + 1),
                        LCNN(num_nodes, enc_dim // subband_num, nclasses=num_classes)
                        )
        self.fc_out = nn.Linear(enc_dim, num_classes)

    def forward(self, x):
        subs = torch.split(x, x.shape[2] // self.subband_num, dim=2)
        feat_loader = []
        for i in range(self.subband_num):
            feati, _ = getattr(self, 'sub'+str(i+1))(subs[i])
            feat_loader.append(feati)
        #print(feat_loader)
        #feat = torch.cat(feat_loader, dim=-1)
        #out_loader = self.fc_out(feat_loader)
        return feat_loader


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)  # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()

        self.device = device

        self.Sinc_conv = SincConv(device=self.device,
                                  out_channels=d_args['filts'][0],
                                  kernel_size=d_args['first_conv'],
                                  in_channels=d_args['in_channels']
                                  )

        self.first_bn = nn.BatchNorm1d(num_features=d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1], first=True))
        self.block1 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features=d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size=d_args['filts'][2][-1],
                          hidden_size=d_args['gru_node'],
                          num_layers=d_args['nb_gru_layer'],
                          batch_first=True)

        self.fc1_gru = nn.Linear(in_features=d_args['gru_node'],
                                 out_features=d_args['nb_fc_node'])

        self.fc2_gru = nn.Linear(in_features=d_args['nb_fc_node'],
                                 out_features=d_args['nb_classes'], bias=True)

        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):

        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)  # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1)  # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)  # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2  # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1)  # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3  # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)  # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4  # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1)  # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5  # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        output = self.logsoftmax(x)

        return output

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []

        l_fc.append(nn.Linear(in_features=in_features,
                              out_features=l_out_features))

        return nn.Sequential(*l_fc)

    def _make_layer(self, nb_blocks, nb_filts, first=False):
        layers = []
        # def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts=nb_filts,
                                         first=first))
            if i == 0: nb_filts[0] = nb_filts[1]

        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn=None):
        if print_fn == None: printfn = print
        model = self

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()

        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)

# class RawNet2(nn.Module):
#     def __init__(self):


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(summary(se_res2net50_v1b(pretrained=False, num_classes=2), torch.randn((1, 1, 60, 750)), show_input=False))
    lfcc = torch.randn(1, 1, 60, 750)
    res2net = se_res2net50_v1b(pretrained=False, num_classes=2)
    feat, out = res2net(lfcc)
    print(feat.shape)
    print(out.shape)
    # # cqcc = torch.randn((32,1,90,788)).cuda()
    # # resnet = ResNet(3, 2, resnet_type='18', nclasses=2).cuda()
    # # _, output = resnet(cqcc)
    # # print(output.shape)
    # lfcc = torch.randn((1, 1, 60, 750))
    # # lcnn = LCNN(4, 2, nclasses=2).cuda()
    # subband = Subband(1, 256, '18', 2, 3)
    # # feat, output = resnet(lfcc)
    # feat_loader = torch.randn((3,256))
    # #out_loader = torch.randn((1,256))
    # feat_loader = subband(lfcc)
    # feat = torch.cat(feat_loader, dim=-1)
    # output = nn.Linear(256, 1)(feat)
    # #output = lfcc.fc_out(feat)
    # print(feat.shape)
    # print(output.shape)
    # # cnn = ConvNet(num_classes = 2, num_nodes = 47232, enc_dim = 256).cuda()
    # # _, output = cnn(lfcc)
    # # print(output.shape)
