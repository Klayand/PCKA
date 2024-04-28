import torch.nn as nn
import torch.nn.functional as F
import math
from .common import FlowAlignModule

__all__ = ['vgg13_bn_aux', 'vgg13_bn', 'vgg13_bn_spkd', 'vgg13_bn_crd', "vgg8_fakd", "vgg8_bn_fakd","vgg13"]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


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


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x

        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x

        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x

        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f3 = x

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3], x
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Auxiliary_Classifier(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=100):
        super(Auxiliary_Classifier, self).__init__()

        self.block_extractor1 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[1], batch_norm, cfg[0][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[2], batch_norm, cfg[1][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.block_extractor2 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[2], batch_norm, cfg[1][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.block_extractor3 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2),
                                                self._make_layers(cfg[3], batch_norm, cfg[2][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.block_extractor4 = nn.Sequential(*[self._make_layers(cfg[3], batch_norm, cfg[4][-1]),
                                                nn.ReLU(inplace=True),
                                                self._make_layers(cfg[4], batch_norm, cfg[3][-1]),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1))])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            out = out.view(-1, 512)
            out = getattr(self, 'fc' + str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class VGG_Auxiliary(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=100):
        super(VGG_Auxiliary, self).__init__()
        self.backbone = VGG(cfg, batch_norm=batch_norm, num_classes=num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(cfg, batch_norm=batch_norm, num_classes=num_classes * 4)

    def forward(self, x, grad=False):
        feats, logit = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_logits = self.auxiliary_classifier(feats)
        return logit, ss_logits


class VGG_SPKD(VGG):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG_SPKD, self).__init__(cfg, batch_norm, num_classes)

    def forward(self, x):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x

        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x

        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x

        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f3 = x

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return f3, x


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


class VGG_CRD(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG_CRD, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(512, num_classes)
        linear = nn.Linear(512, 128, bias=True)
        self.normalizer = Normalizer4CRD(linear, power=2)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x):
        h = x.shape[2]
        x = F.relu(self.block0(x))

        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)

        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)

        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)

        x = self.pool4(x)
        crdout = x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        crdout = self.normalizer(crdout)
        return crdout, x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_FAKD(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000, **kwargs):
        super(VGG_FAKD, self).__init__()
        if "dirac_ratio" not in kwargs:
            kwargs["dirac_ratio"] = 1.0
        if "weight" not in kwargs:
            kwargs["weight"] = [1, 1, 1, 0]
        if "flow_loss_type" not in kwargs:
            kwargs["flow_loss_type"] = None
        if "encoder_type" not in kwargs:
            kwargs["encoder_type"] = None
        dirac_ratio = kwargs["dirac_ratio"]
        weight = kwargs["weight"]
        flow_loss_type = kwargs["flow_loss_type"]
        encoder_type = kwargs["encoder_type"]
        self.teacher_features = kwargs["teacher_features"]
        self.teacher_sizes = kwargs["teacher_sizes"]
        img_size = 32
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        img_size = img_size // 2
        self.flow1 = FlowAlignModule(self.teacher_features[0], cfg[1][-1], "feature_based",
                                     self.teacher_sizes[0],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[0], loss_type=flow_loss_type,encoder_type=encoder_type)
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        img_size = img_size // 4
        self.flow2 = FlowAlignModule(self.teacher_features[1], cfg[2][-1], "feature_based",
                                     self.teacher_sizes[1],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[1], loss_type=flow_loss_type,encoder_type=encoder_type)
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])
        img_size = img_size // 2
        self.flow3 = FlowAlignModule(self.teacher_features[2], cfg[4][-1], "feature_based",
                                     self.teacher_sizes[2],
                                     img_size, dirac_ratio=dirac_ratio, weight=weight[2], loss_type=flow_loss_type,encoder_type=encoder_type)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        if weight[3] != 0:
            self.logit_based = True
        else:
            self.logit_based = False
        if self.logit_based:
            self.flow4 = FlowAlignModule(num_classes, 512, "logit_based", self.teacher_sizes[2], img_size,
                                         dirac_ratio=dirac_ratio, weight=weight[3], loss_type=flow_loss_type,encoder_type=encoder_type)
        else:
            self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, teacher_features=None, is_feat=False, inference_sampling=4, **kwargs):
        flow_loss = 0
        if teacher_features == None:
            teacher_features = [None] * 4
        if len(teacher_features)>4:
            teacher_features = teacher_features[1:]
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x

        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        loss, x = self.flow1(x, teacher_features[0], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss

        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        loss, x = self.flow2(x, teacher_features[1], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss

        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)

        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f3 = x
        loss, x = self.flow3(x, teacher_features[2], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss

        if self.logit_based:
            loss, x = self.flow4(x, teacher_features[3], inference_sampling=inference_sampling, **kwargs)
            flow_loss += loss
        else:
            x = F.relu(x)
            x = self.pool4(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        if is_feat:
            return [f0, f1, f2, f3], x, flow_loss
        else:
            return x, flow_loss

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3,last_ignore=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-last_ignore]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg8_fakd(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_FAKD(cfg['S'], **kwargs)
    return model


def vgg8_bn_fakd(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_FAKD(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg8_bn_aux(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_Auxiliary(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg8_bn_spkd(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_SPKD(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg8_bn_crd(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_CRD(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg13_bn_aux(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_Auxiliary(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg13_bn_spkd(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_SPKD(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg13_bn_crd(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_CRD(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = vgg13_bn_aux(num_classes=100)
    logit, ss_logits = net(x)
    print(logit.size())
