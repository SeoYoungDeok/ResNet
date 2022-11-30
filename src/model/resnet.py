import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Type, List, Optional, Union


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_features: int, out_features: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        self.stride = stride

        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_features)

        self.conv2 = nn.Conv2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                stride=self.stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_features: int, out_features: int, stride: int = 1):
        super(BottleneckBlock, self).__init__()
        self.stride = stride
        self.in_features = in_features
        self.out_features = out_features

        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=self.stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_features)

        self.conv2 = nn.Conv2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.conv3 = nn.Conv2d(
            in_channels=out_features,
            out_channels=out_features * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_features * self.expansion)

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features * self.expansion,
                kernel_size=1,
                stride=self.stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_features * self.expansion),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride != 1 or self.in_features != self.out_features * self.expansion:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, BottleneckBlock]],
        block_repaat: List[int],
        stride: List[int],
        class_num: int = 10,
    ):
        super(ResNet, self).__init__()

        self.in_features = 64

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_repaat[0], stride=stride[0])
        self.layer2 = self._make_layer(block, 128, block_repaat[1], stride=stride[1])
        self.layer3 = self._make_layer(block, 256, block_repaat[2], stride=stride[2])
        self.layer4 = self._make_layer(block, 512, block_repaat[3], stride=stride[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, BottleneckBlock]],
        out_features: int,
        block_repaat: int,
        stride: int = 1,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (block_repaat - 1)
        layers = []
        for i in range(block_repaat):
            layers.append(block(self.in_features, out_features, strides[i]))
            self.in_features = block.expansion * out_features
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], [1, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], [1, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3], [1, 2, 2, 2])


def ResNet101():
    return ResNet(BottleneckBlock, [3, 4, 23, 3], [1, 2, 2, 2])


def ResNet152():
    return ResNet(BottleneckBlock, [3, 8, 36, 3], [1, 2, 2, 2])
