import torch
import torch.nn as nn
from .common import FlowAlignModule, OnlineFlowAlignModule

__all__ = ['resnet18_imagenet', 'resnet18_imagenet_aux', 'resnet34_imagenet',
           'resnet34_imagenet_aux', 'resnet50_imagenet', 'resnet50_imagenet_aux', 'resnet18_imagenet_fakd',
           'resnet50_imagenet_fakd', 'online_resnet18_imagenet_fakd', 'online_resnet34_imagenet_fakd']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if is_feat:
            return [f1, f2, f3, f4], x
        else:
            return x


class ResNet_FAKD(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, **kwargs):
        super(ResNet_FAKD, self).__init__()
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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        img_size = 56
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        img_size = img_size // 2
        self.flow1 = FlowAlignModule(self.teacher_features[0], 128, "feature_based",
                                     self.teacher_sizes[0],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[0], loss_type=flow_loss_type,
                                     encoder_type=self.encoder_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        img_size = img_size // 2
        self.flow2 = FlowAlignModule(self.teacher_features[1], 256, "feature_based",
                                     self.teacher_sizes[1],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[1], loss_type=flow_loss_type,
                                     encoder_type=self.encoder_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        img_size = img_size // 2
        self.flow3 = FlowAlignModule(self.teacher_features[2], 512, "feature_based",
                                     self.teacher_sizes[2],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[2], loss_type=flow_loss_type,
                                     encoder_type=self.encoder_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if weight[3] != 0:
            self.logit_based = True
        else:
            self.logit_based = False
        if self.logit_based:
            self.flow4 = FlowAlignModule(num_classes, 512 * block.expansion, "logit_based", self.teacher_sizes[2],
                                         img_size,
                                         dirac_ratio=dirac_ratio,
                                         weight=weight[3], loss_type=flow_loss_type, encoder_type=self.encoder_type)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, teacher_features=None, is_feat=False, inference_sampling=4, **kwargs):
        flow_loss = 0
        if teacher_features == None:
            teacher_features = [None] * 4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        loss, x = self.flow1(x, teacher_features[0], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        x = self.layer3(x)
        f3 = x
        loss, x = self.flow2(x, teacher_features[1], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        x = self.layer4(x)
        loss, x = self.flow3(x, teacher_features[2], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        f4 = x

        if self.logit_based:
            loss, x = self.flow4(x, teacher_features[3], inference_sampling=inference_sampling, **kwargs)
            flow_loss += loss
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        if is_feat:
            return [f1, f2, f3, f4], x, flow_loss
        else:
            return x, flow_loss


class Online_ResNet_FAKD(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, **kwargs):
        super(Online_ResNet_FAKD, self).__init__()
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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        img_size = 56
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        img_size = img_size // 2
        self.flow1 = OnlineFlowAlignModule(self.teacher_features[0], 128, "feature_based",
                                           self.teacher_sizes[0],
                                           img_size, dirac_ratio=dirac_ratio, weight=weight[0],
                                           loss_type=flow_loss_type, encoder_type=self.encoder_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        img_size = img_size // 2
        self.flow2 = OnlineFlowAlignModule(self.teacher_features[1], 256, "feature_based",
                                           self.teacher_sizes[1],
                                           img_size, dirac_ratio=dirac_ratio, weight=weight[1],
                                           loss_type=flow_loss_type, encoder_type=self.encoder_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        img_size = img_size // 2
        self.flow3 = OnlineFlowAlignModule(self.teacher_features[2], 512, "feature_based",
                                           self.teacher_sizes[2],
                                           img_size, dirac_ratio=dirac_ratio, weight=weight[2],
                                           loss_type=flow_loss_type, encoder_type=self.encoder_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if weight[3] != 0:
            self.logit_based = True
        else:
            self.logit_based = False
        if self.logit_based:
            self.flow4 = OnlineFlowAlignModule(num_classes, 512 * block.expansion, "logit_based", self.teacher_sizes[2],
                                               img_size,
                                               dirac_ratio=dirac_ratio,
                                               weight=weight[3], loss_type=flow_loss_type,
                                               encoder_type=self.encoder_type)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, inference_sampling=4, **kwargs):
        flow_loss = 0

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        loss, x = self.flow1(x, inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        x = self.layer3(x)
        f3 = x
        loss, x = self.flow2(x, inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        x = self.layer4(x)
        loss, x = self.flow3(x, inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        f4 = x

        if self.logit_based:
            loss, x = self.flow4(x, inference_sampling=inference_sampling, **kwargs)
            flow_loss += loss
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        if is_feat:
            return [f1, f2, f3, f4], x, flow_loss
        else:
            return x, flow_loss


class Auxiliary_Classifier(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Auxiliary_Classifier, self).__init__()

        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.block_extractor1 = nn.Sequential(*[self._make_layer(block, 128, layers[1], stride=2),
                                                self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 128 * block.expansion
        self.block_extractor2 = nn.Sequential(*[self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 256 * block.expansion
        self.block_extractor3 = nn.Sequential(*[self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 512 * block.expansion
        self.block_extractor4 = nn.Sequential(*[self._make_layer(block, 512, layers[3], stride=1)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1

            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = getattr(self, 'fc' + str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class ResNet_Auxiliary(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_Auxiliary, self).__init__()
        self.backbone = ResNet(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)
        self.auxiliary_classifier = Auxiliary_Classifier(block, layers, num_classes=num_classes * 4,
                                                         zero_init_residual=zero_init_residual)

    def forward(self, x, grad=False):
        if grad is False:
            feats, logit = self.backbone(x, is_feat=True)
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        else:
            feats, logit = self.backbone(x, is_feat=True)

        ss_logits = self.auxiliary_classifier(feats)
        return logit, ss_logits


def resnet18_imagenet(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_imagenet_fakd(**kwargs):
    return ResNet_FAKD(BasicBlock, [2, 2, 2, 2], **kwargs)


def online_resnet18_imagenet_fakd(**kwargs):
    return Online_ResNet_FAKD(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_imagenet_aux(**kwargs):
    return ResNet_Auxiliary(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_imagenet(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def online_resnet34_imagenet_fakd(**kwargs):
    return Online_ResNet_FAKD(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet34_imagenet_aux(**kwargs):
    return ResNet_Auxiliary(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_imagenet(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50_imagenet_fakd(**kwargs):
    return ResNet_FAKD(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50_imagenet_aux(**kwargs):
    return ResNet_Auxiliary(Bottleneck, [3, 4, 6, 3], **kwargs)


if __name__ == '__main__':
    net = resnet34_imagenet_aux(num_classes=1000)
    from C100.utils import cal_param_size, cal_multi_adds

    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 224, 224)) / 1e6))
