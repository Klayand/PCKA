import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import FlowAlignModule

__all__ = ['wrn', 'wrn_40_2_aux', 'wrn_16_2_aux', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
           'wrn_40_1_aux', 'wrn_16_2_spkd', 'wrn_40_1_spkd', 'wrn_40_2_spkd', 'wrn_40_1_crd', 'wrn_16_2_crd',
           'wrn_40_2_crd', 'wrn_16_2_sskd', 'wrn_40_1_sskd', 'wrn_40_2_sskd', "wrn_16_2_fakd", "wrn_40_1_fakd",
           "wrn_40_2_fakd"]


class Normalizer4CRD(nn.Module):
    def __init__(self, linear, power=2):
        super().__init__()
        self.linear = linear
        self.power = power

    def forward(self, x):
        x = x.flatten(1)
        z = self.linear(x)
        norm = z.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = z.div(norm)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0,last_encoder=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.last_encoder = last_encoder
        if self.last_encoder is None:
            self.le = nn.Identity()
        else:
            from .common import Meta_Encoder
            self.le = Meta_Encoder(self.last_encoder)(nChannels[3])
            print(f"Apply Last Encoder: {self.last_encoder}!")
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.le(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)
        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
            return [f1, f2, f3], out
        else:
            return out


class Auxiliary_Classifier(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(Auxiliary_Classifier, self).__init__()
        self.nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        block = BasicBlock
        n = (depth - 4) // 6
        self.block_extractor1 = nn.Sequential(*[NetworkBlock(n, self.nChannels[1], self.nChannels[2], block, 2),
                                                NetworkBlock(n, self.nChannels[2], self.nChannels[3], block, 2), ])
        self.block_extractor2 = nn.Sequential(*[NetworkBlock(n, self.nChannels[2], self.nChannels[3], block, 2)])
        self.block_extractor3 = nn.Sequential(*[NetworkBlock(n, self.nChannels[3], self.nChannels[3], block, 1)])

        self.bn1 = nn.BatchNorm2d(self.nChannels[3])
        self.bn2 = nn.BatchNorm2d(self.nChannels[3])
        self.bn3 = nn.BatchNorm2d(self.nChannels[3])

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.nChannels[3], num_classes)
        self.fc2 = nn.Linear(self.nChannels[3], num_classes)
        self.fc3 = nn.Linear(self.nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        ss_logits = []
        ss_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            out = self.relu(getattr(self, 'bn' + str(idx))(out))
            out = self.avg_pool(out)
            out = out.view(-1, self.nChannels[3])
            ss_feats.append(out)
            out = getattr(self, 'fc' + str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class WideResNet_Auxiliary(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_Auxiliary, self).__init__()
        self.backbone = WideResNet(depth, num_classes, widen_factor=widen_factor)
        self.auxiliary_classifier = Auxiliary_Classifier(depth=depth, num_classes=num_classes * 4,
                                                         widen_factor=widen_factor)

    def forward(self, x, grad=False):
        feats, logit = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_logits = self.auxiliary_classifier(feats)

        return logit, ss_logits


class WideResNet_SPKD(WideResNet):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_SPKD, self).__init__(depth, num_classes, widen_factor, dropRate)

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)
        return f4, out


class WideResNet_SSKD(WideResNet):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_SSKD, self).__init__(depth, num_classes, widen_factor, dropRate)
        self.ss_module = nn.Sequential(
            nn.Linear(self.nChannels, self.nChannels),
            nn.ReLU(inplace=True),
            nn.Linear(self.nChannels, self.nChannels)
        )

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        f4 = self.ss_module(out)
        out = self.fc(out)
        return f4, out


class WideResNet_CRD(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_CRD, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        linear = nn.Linear(nChannels[3], 128, bias=True)
        self.normalizer = Normalizer4CRD(linear, power=2)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        crdout = out
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        crdout = self.normalizer(crdout)
        return crdout, out


class WideResNet_FAKD(nn.Module):
    def __init__(self, depth, num_classes, teacher_features, teacher_sizes, widen_factor=1, dropRate=0.0,
                 **kwargs):
        super(WideResNet_FAKD, self).__init__()
        if "dirac_ratio" not in kwargs:
            kwargs["dirac_ratio"] = 1.0
        if "weight" not in kwargs:
            kwargs["weight"] = [1,1,1,0]
        if "flow_loss_type" not in kwargs:
            kwargs["flow_loss_type"] = None
        if "encoder_type" not in kwargs:
            kwargs["encoder_type"] = None
        dirac_ratio = kwargs["dirac_ratio"]
        weight = kwargs["weight"]
        flow_loss_type = kwargs["flow_loss_type"]
        encoder_type = kwargs["encoder_type"]
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        print(f"Layer Weight are {weight}")
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        img_size = 32
        self.teacher_features = teacher_features
        self.teacher_sizes = teacher_sizes
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.flow1 = FlowAlignModule(self.teacher_features[0], nChannels[1], "feature_based", self.teacher_sizes[0],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[0],loss_type=flow_loss_type,encoder_type=encoder_type,)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        img_size = img_size // 2
        self.flow2 = FlowAlignModule(self.teacher_features[1], nChannels[2], "feature_based", self.teacher_sizes[1],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[1],loss_type=flow_loss_type,encoder_type=encoder_type)

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        img_size = img_size // 2
        self.flow3 = FlowAlignModule(self.teacher_features[2], nChannels[3], "feature_based", self.teacher_sizes[2],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[2],loss_type=flow_loss_type,encoder_type=encoder_type)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        if weight[3]!=0:
            self.logit_based = True
        else:
            self.logit_based = False
        if self.logit_based:
            self.flow4 = FlowAlignModule(num_classes, nChannels[3], "logit_based",self.teacher_sizes[2], img_size, dirac_ratio=dirac_ratio,
                                         weight=weight[3],loss_type=flow_loss_type,encoder_type=encoder_type)
        else:
            self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def forward(self, x, teacher_features=None, is_feat=False, preact=False, inference_sampling=4, **kwargs):
        flow_loss = 0
        if teacher_features == None:
            teacher_features = [None] * 4
        out = self.conv1(x)
        out = self.block1(out)
        loss, out = self.flow1(out, teacher_features[0], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        f1 = out
        out = self.block2(out)
        loss, out = self.flow2(out, teacher_features[1], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        f2 = out
        out = self.block3(out)
        loss, out = self.flow3(out, teacher_features[2], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        f3 = out
        if self.logit_based:
            out = self.bn1(out)
            loss, out = self.flow4(out, teacher_features[3], inference_sampling=inference_sampling, **kwargs)
            flow_loss += loss
        else:
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size(3))
            out = out.view(-1, self.nChannels)
            out = self.fc(out)
        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
            return [f1, f2, f3], out, flow_loss
        else:
            return out, flow_loss


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_2_aux(**kwargs):
    model = WideResNet_Auxiliary(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_2_spkd(**kwargs):
    model = WideResNet_SPKD(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_2_sskd(**kwargs):
    model = WideResNet_SSKD(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_2_crd(**kwargs):
    model = WideResNet_CRD(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_2_fakd(**kwargs):
    model = WideResNet_FAKD(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_1(**kwargs):
    model = WideResNet(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_40_1_aux(**kwargs):
    model = WideResNet_Auxiliary(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_40_1_spkd(**kwargs):
    model = WideResNet_SPKD(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_40_1_crd(**kwargs):
    model = WideResNet_CRD(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_40_1_sskd(**kwargs):
    model = WideResNet_SSKD(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_40_1_fakd(**kwargs):
    model = WideResNet_FAKD(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_16_2(**kwargs):
    model = WideResNet(depth=16, widen_factor=2, **kwargs)
    return model


def wrn_16_2_aux(**kwargs):
    model = WideResNet_Auxiliary(depth=16, widen_factor=2, **kwargs)
    return model


def wrn_16_2_spkd(**kwargs):
    model = WideResNet_SPKD(depth=16, widen_factor=2, **kwargs)
    return model


def wrn_16_2_crd(**kwargs):
    model = WideResNet_CRD(depth=16, widen_factor=2, **kwargs)
    return model


def wrn_16_2_sskd(**kwargs):
    model = WideResNet_SSKD(depth=16, widen_factor=2, **kwargs)
    return model


def wrn_16_2_fakd(**kwargs):
    model = WideResNet_FAKD(depth=16, widen_factor=2, **kwargs)
    return model


def wrn_16_1(**kwargs):
    model = WideResNet(depth=16, widen_factor=1, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = wrn_40_2_aux(num_classes=100)
    logit, ss_logits = net(x)
    print(logit.size())
