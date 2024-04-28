import torch, copy
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from mmdet.models import FasterRCNN
from mmrazor.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from typing import List, Tuple, Union, Dict
from mmrazor.models.architectures.necks import FlowAlignmentFPN
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

@MODELS.register_module()
class FlowAlignmentFasterRCNN(FasterRCNN):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             teacher_fpn_outputs: Tuple[Tensor]) -> dict:
        x, flow_loss = self.extract_feat(batch_inputs, teacher_fpn_outputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
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