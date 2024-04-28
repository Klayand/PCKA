import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mobilenetv1_imagenet", "mobilenetv1_imagenet_fakd"]


def _initialize_weight_goog(m):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(0)  # fan-out
        init_range = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-init_range, init_range)
        if m.bias is not None:
            m.bias.data.zero_()


class MobileNetV1(nn.Module):
    def __init__(self, ch_in=3, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            _initialize_weight_goog(m)

    def forward(self, x, is_feat=False):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        if is_feat:
            return 0, x
        else:
            return x


class MobileNetV1_FAKD(nn.Module):
    def __init__(self, ch_in=3, num_classes=1000, **kwargs):
        super(MobileNetV1_FAKD, self).__init__()
        if "dirac_ratio" not in kwargs:
            kwargs["dirac_ratio"] = 1.0
        if "weight" not in kwargs:
            kwargs["weight"] = [1, 1, 1, 0]
        if "flow_loss_type" not in kwargs:
            kwargs["flow_loss_type"] = None
        dirac_ratio = kwargs["dirac_ratio"]
        weight = kwargs["weight"]
        flow_loss_type = kwargs["flow_loss_type"]
        self.teacher_features = kwargs["teacher_features"]
        self.teacher_sizes = kwargs["teacher_sizes"]
        self.encoder_type = kwargs["encoder_type"]

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.ModuleList([
            nn.Sequential(
                conv_bn(ch_in, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2)),
            nn.Sequential(
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2)),
            nn.Sequential(
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2)),
            nn.Sequential(
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1)),
            nn.AdaptiveAvgPool2d(1)])
        shapes = [28, 14, 7]
        from .common import FlowAlignModule
        self.flow1 = FlowAlignModule(self.teacher_features[0], 256, "feature_based",
                                     self.teacher_sizes[0],
                                     shapes[0], dirac_ratio=dirac_ratio, weight=weight[0], loss_type=flow_loss_type,
                                     encoder_type=self.encoder_type)
        self.flow2 = FlowAlignModule(self.teacher_features[1], 512, "feature_based",
                                     self.teacher_sizes[1],
                                     shapes[1], dirac_ratio=dirac_ratio, weight=weight[1], loss_type=flow_loss_type,
                                     encoder_type=self.encoder_type)
        self.flow3 = FlowAlignModule(self.teacher_features[2], 1024, "feature_based",
                                     self.teacher_sizes[2],
                                     shapes[2], dirac_ratio=dirac_ratio, weight=weight[2], loss_type=flow_loss_type,
                                     encoder_type=self.encoder_type)

        if weight[3] != 0:
            self.logit_based = True
        else:
            self.logit_based = False
        if self.logit_based:
            self.flow4 = FlowAlignModule(num_classes, 1024, "logit_based", self.teacher_sizes[2],
                                         shapes[2], dirac_ratio=dirac_ratio,
                                         weight=weight[3], loss_type=flow_loss_type, encoder_type=self.encoder_type)
        else:
            self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            _initialize_weight_goog(m)

    def forward(self, x, teacher_features=None, is_feat=False, inference_sampling=4, **kwargs):
        out = []
        flow_loss = 0
        if teacher_features == None:
            teacher_features = [None] * 4
        if len(teacher_features) >= 5:
            teacher_features = teacher_features[-4:]
        x = self.model[0](x)
        out.append(x)
        x = self.model[1](x)
        out.append(x)
        loss, x = self.flow1(x, teacher_features[0], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        x = self.model[2](x)
        out.append(x)
        loss, x = self.flow2(x, teacher_features[1], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        x = self.model[3](x)
        out.append(x)
        loss, x = self.flow3(x, teacher_features[2], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss

        if self.logit_based:
            loss, x = self.flow4(x, teacher_features[3], inference_sampling=inference_sampling, **kwargs)
            flow_loss += loss
        else:
            x = self.model[4](x)
            x = x.view(-1, 1024)
            x = self.fc(x)

        if is_feat:
            return out, x, flow_loss
        else:
            return x, flow_loss


def mobilenetv1_imagenet(**kwargs):
    return MobileNetV1(**kwargs)


def mobilenetv1_imagenet_fakd(**kwargs):
    return MobileNetV1_FAKD(**kwargs)