"""
Challenge 2021 1st model (ResNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
from einops import rearrange


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(interChannels)
        self.conv2 = nn.Conv1d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(functional.relu(self.bn1(x)))
        out = self.conv2(functional.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(functional.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, down=False):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.down = down

    def forward(self, x):
        out = self.conv1(functional.relu(self.bn1(x)))
        if self.down:
            out = functional.avg_pool1d(out, 2)
        return out


class MyResidualUBlock(nn.Module):
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super(MyResidualUBlock, self).__init__()
        self.downsample = downsampling

        self.conv1 = nn.Conv2d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=(3, 9),
                               padding=(1, 4),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for idx in range(layers):
            if idx == 0:
                self.encoders.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_ch,
                        out_channels=mid_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        bias=False
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.encoders.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=mid_ch,
                        out_channels=mid_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        bias=False
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))

            if idx == layers - 1:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=mid_ch * 2,
                        out_channels=out_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        output_padding=(0, 1),
                        bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=mid_ch * 2,
                        out_channels=mid_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        output_padding=(0, 1),
                        bias=False
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))

            self.bottleneck = nn.Sequential(
                nn.Conv2d(
                    in_channels=mid_ch,
                    out_channels=mid_ch,
                    kernel_size=(3, 9),
                    padding=(1, 4),
                    bias=False
                ),
                nn.BatchNorm2d(mid_ch),
                nn.LeakyReLU()
            )

            if self.downsample:
                self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
                self.idfunc_1 = nn.Conv2d(in_channels=out_ch,
                                          out_channels=out_ch,
                                          kernel_size=1,
                                          bias=False)

    def forward(self, x):
        x_in = functional.leaky_relu(self.bn1(self.conv1(x)))

        out = x_in
        encoder_out = []
        for idx, layer in enumerate(self.encoders):
            out = layer(out)
            encoder_out.append(out)
        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            out = layer(torch.cat([out, encoder_out[-1 - idx]], dim=1))

        out = out[..., :x_in.size(-1)]
        out += x_in

        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out


class ResDenseNet(nn.Module):
    def __init__(self, nOUT, in_ch=1, out_ch=128, mid_ch=64, lead=12):
        super(ResDenseNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=(3, 15),
                              padding=(1, 7),
                              stride=(1, 2),
                              bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

        self.rub_0 = MyResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=6)
        self.rub_1 = MyResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=5)
        self.rub_2 = MyResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_3 = MyResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)

        self.conv2 = nn.Conv1d(in_channels=lead * out_ch,
                               out_channels=out_ch,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        growthRate = 12
        reduction = 0.5
        nChannels = out_ch
        nDenseBlocks = 16

        self.dense1 = self._make_dense(nChannels, growthRate=12, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        self.mha_small = nn.MultiheadAttention(nOutChannels, 8)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate=12, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        self.trans2 = Transition(nChannels, out_ch, down=True)

        self.mha = nn.MultiheadAttention(out_ch, 8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc_1 = nn.Linear(out_ch, nOUT)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = functional.leaky_relu(self.bn(self.conv(x)))

        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.rub_2(x)
        x = self.rub_3(x)

        x = rearrange(x, 'b c l t -> b (c l) t')
        x = functional.leaky_relu(self.bn2(self.conv2(x)))

        x = self.trans1(self.dense1(x))
        x = x.permute(2, 0, 1)
        x, s = self.mha_small(x, x, x)
        x = x.permute(1, 2, 0)
        x = self.trans2(self.dense2(x))

        x = functional.dropout(x, p=0.5, training=self.training)

        x = x.permute(2, 0, 1)
        x, s = self.mha(x, x, x)
        x = x.permute(1, 2, 0)
        x = self.pool(x).squeeze(2)

        x = self.fc_1(x)
        x = x.squeeze()

        return x

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
