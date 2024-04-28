
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .common import FlowAlignModule


__all__ = ['ShuffleV1_aux', 'ShuffleV1','ShuffleV1_fakd']
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.stride = stride

        mid_planes = int(out_planes/4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        preact = torch.cat([out, res], 1) if self.stride == 2 else out+res
        out = F.relu(preact)
        # out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out+res)
        return out


class ShuffleNet_FAKD(nn.Module):
    def __init__(self, cfg, num_classes=10,**kwargs):
        super(ShuffleNet_FAKD, self).__init__()
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
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']
        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        img_size = 32
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        img_size = img_size // 2
        self.flow1 =  FlowAlignModule(self.teacher_features[0], out_planes[0], "feature_based",
                     self.teacher_sizes[0],img_size, dirac_ratio=dirac_ratio, weight=weight[0],
                                      loss_type=flow_loss_type,encoder_type=encoder_type)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        img_size = img_size // 2
        self.flow2 =  FlowAlignModule(self.teacher_features[1], out_planes[1], "feature_based",
                     self.teacher_sizes[1], img_size, dirac_ratio=dirac_ratio, weight=weight[1],
                                      loss_type=flow_loss_type,encoder_type=encoder_type)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        img_size = img_size // 2
        self.flow3 =  FlowAlignModule(self.teacher_features[2], out_planes[2], "feature_based",
                     self.teacher_sizes[2],img_size, dirac_ratio=dirac_ratio, weight=weight[2],
                                      loss_type=flow_loss_type,encoder_type=encoder_type)
        if weight[3] != 0:
            self.logit_based = True
        else:
            self.logit_based = False
        if self.logit_based:
            self.flow4 = FlowAlignModule(num_classes, out_planes[2], "logit_based", self.teacher_sizes[2], img_size,
                                         dirac_ratio=dirac_ratio,weight=weight[3],
                                         loss_type=flow_loss_type,encoder_type=encoder_type)
        else:
            self.linear = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes,
                                     stride=stride,
                                     groups=groups,
                                     is_last=(i == num_blocks - 1)))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        raise NotImplementedError('ShuffleNet currently is not supported for "Overhaul" teacher')

    def forward(self, x, teacher_features=None, is_feat=False, inference_sampling=4, **kwargs):
        flow_loss = 0
        if teacher_features == None:
            teacher_features = [None] * 4

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        f1 = out
        loss, out = self.flow1(out, teacher_features[0], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        out = self.layer2(out)
        f2 = out
        loss, out = self.flow2(out, teacher_features[1], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        out = self.layer3(out)
        f3 = out
        loss, out = self.flow3(out, teacher_features[2], inference_sampling=inference_sampling, **kwargs)
        flow_loss += loss
        if self.logit_based:
            loss, out = self.flow4(out, teacher_features[3], inference_sampling=inference_sampling, **kwargs)
            flow_loss += loss
        else:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        if is_feat:
            return [f1, f2, f3], out, flow_loss
        else:
            return out, flow_loss

class ShuffleNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes,
                                     stride=stride,
                                     groups=groups,
                                     is_last=(i == num_blocks - 1)))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        raise NotImplementedError('ShuffleNet currently is not supported for "Overhaul" teacher')

    def forward(self, x, is_feat=False, preact=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        f1 = out
        out = self.layer2(out)
        f2 = out
        out = self.layer3(out)
        f3 = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        f4 = out
        out = self.linear(out)

        if is_feat:
            return [f1, f2, f3], out
        else:
            return out

class Auxiliary_Classifier(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(Auxiliary_Classifier, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.in_planes = out_planes[0]
        self.block_extractor1 = nn.Sequential(*[self._make_layer(out_planes[1], num_blocks[1], groups),
                                                self._make_layer(out_planes[2], num_blocks[2], groups)])
        self.in_planes = out_planes[1]
        self.block_extractor2 = nn.Sequential(*[self._make_layer(out_planes[2], num_blocks[2], groups)])

        self.inplanes = out_planes[2]
        self.block_extractor3 = nn.Sequential(*[self._make_layer(out_planes[2], num_blocks[2], groups, downsample=False)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(out_planes[2], num_classes)
        self.fc2 = nn.Linear(out_planes[2], num_classes)
        self.fc3 = nn.Linear(out_planes[2], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_layer(self, out_planes, num_blocks, groups, downsample=True):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 and downsample is True else 1
            cat_planes = self.in_planes if i == 0  and downsample is True else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes,
                                     stride=stride,
                                     groups=groups,
                                     is_last=(i == num_blocks - 1)))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

                               
    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor'+str(idx))(x[i])
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = getattr(self, 'fc'+str(idx))(out)
            ss_logits.append(out)
        return ss_logits


class ShuffleNet_Auxiliary(nn.Module):
    def __init__(self, cfg, num_classes=100):
        super(ShuffleNet_Auxiliary, self).__init__()
        self.backbone = ShuffleNet(cfg, num_classes=num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(cfg, num_classes=num_classes * 4)
        
    def forward(self, x, grad=False):
        feats, logit = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_logits = self.auxiliary_classifier(feats)
        return logit, ss_logits


def ShuffleV1(**kwargs):
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet(cfg, **kwargs)

def ShuffleV1_fakd(**kwargs):
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet_FAKD(cfg, **kwargs)

def ShuffleV1_aux(**kwargs):
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet_Auxiliary(cfg, **kwargs)


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = ShuffleV1_aux(num_classes=100)
    logit, ss_logits = net(x)
    print(logit.size())
