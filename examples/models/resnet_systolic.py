import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ---- AdaPT systolic layers ----

from adapt.approx_layers.layers_systolic import (
    AdaPT_Conv2d_Systolic as SConv2d,
    AdaPT_Linear_Systolic as SLinear,
)

__all__ = [
    "ResNetSystolic",
    "resnet18_systolic",
    "resnet34_systolic",
    "resnet50_systolic",
]

# global knobs (mirrors your style)
_systolic_axx_mult  = "mul8s_acc"
_systolic_use_exact = True   # True → exact multiply, False → LUT approx


def conv3x3_systolic(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 systolic convolution with padding."""
    return SConv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation,
        axx_mult=_systolic_axx_mult, use_exact=_systolic_use_exact
    )

def conv1x1_systolic(in_planes, out_planes, stride=1):
    """1x1 systolic convolution."""
    return SConv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False,
        axx_mult=_systolic_axx_mult, use_exact=_systolic_use_exact
    )


class BasicBlockSystolic(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlockSystolic supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlockSystolic")

        self.conv1 = conv3x3_systolic(inplanes, planes, stride)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_systolic(planes, planes)
        self.bn2   = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class BottleneckSystolic(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1_systolic(inplanes, width)
        self.bn1   = norm_layer(width)
        self.conv2 = conv3x3_systolic(width, width, stride, groups, dilation)
        self.bn2   = norm_layer(width)
        self.conv3 = conv1x1_systolic(width, planes * self.expansion)
        self.bn3   = norm_layer(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNetSystolic(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must be 3 elements")

        self.groups     = groups
        self.base_width = width_per_group

        # CIFAR-style stem (3x3, stride 1, pad 1)
        self.conv1 = SConv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False,
            axx_mult=_systolic_axx_mult, use_exact=_systolic_use_exact
        )
        self.bn1   = norm_layer(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Systolic linear head
        self.fc = SLinear(512 * block.expansion, num_classes,
                          bias=True, axx_mult=_systolic_axx_mult, use_exact=_systolic_use_exact)

        # Init like your original
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, SConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckSystolic):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockSystolic):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_systolic(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---- factory helpers (mirror your API) --------------------------------------

def _resnet_systolic(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNetSystolic(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict, strict=False)  # BN stats etc. are safe; systolic layers match shapes
    return model


def resnet18_systolic(pretrained=False, progress=True, device="cpu",
                      axx_mult="mul8s_acc", use_exact=True, **kwargs):
    global _systolic_axx_mult, _systolic_use_exact
    _systolic_axx_mult  = axx_mult
    _systolic_use_exact = use_exact
    return _resnet_systolic("resnet18", BasicBlockSystolic, [2, 2, 2, 2],
                            pretrained, progress, device, **kwargs)


def resnet34_systolic(pretrained=False, progress=True, device="cpu",
                      axx_mult="mul8s_acc", use_exact=True, **kwargs):
    global _systolic_axx_mult, _systolic_use_exact
    _systolic_axx_mult  = axx_mult
    _systolic_use_exact = use_exact
    return _resnet_systolic("resnet34", BasicBlockSystolic, [3, 4, 6, 3],
                            pretrained, progress, device, **kwargs)


def resnet50_systolic(pretrained=False, progress=True, device="cpu",
                      axx_mult="mul8s_acc", use_exact=True, **kwargs):
    global _systolic_axx_mult, _systolic_use_exact
    _systolic_axx_mult  = axx_mult
    _systolic_use_exact = use_exact
    return _resnet_systolic("resnet50", BottleneckSystolic, [3, 4, 6, 3],
                            pretrained, progress, device, **kwargs)
