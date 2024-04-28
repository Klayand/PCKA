from torch import Tensor
import torch

from mmdet.models import RetinaNet
from mmrazor.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from typing import List, Tuple, Union, Dict
from mmrazor.models.architectures.necks import FlowAlignmentFPN
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

@MODELS.register_module()
class FlowAlignmentRetinaNet(RetinaNet):
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             teacher_fpn_outputs: Tuple[Tensor]) -> Union[dict, list]:
        x, flow_loss = self.extract_feat(batch_inputs, teacher_fpn_outputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        losses["flow_loss"] = flow_loss
        return losses
    
    def extract_feat(self, batch_inputs: Tensor, teacher_fpn_outputs: Tuple[Tensor] = None):
        x = self.backbone(batch_inputs)
        if self.with_neck:
            if isinstance(self.neck, FlowAlignmentFPN):
                x, flow_loss = self.neck(x, teacher_fpn_outputs)
                if self.training:
                    return x, flow_loss
                else:
                    return x
            else:
                x = self.neck(x)
                return x

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                teacher_fpn_outputs : Tuple[Tensor] = None,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples,teacher_fpn_outputs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')