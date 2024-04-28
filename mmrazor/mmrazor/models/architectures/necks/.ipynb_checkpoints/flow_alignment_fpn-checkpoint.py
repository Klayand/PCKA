import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from mmdet.models.necks import FPN
from mmrazor.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from typing import List, Tuple, Union, Dict

from .common import FlowAlignModule


@MODELS.register_module()
class FlowAlignmentFPN(FPN):
    def __init__(
            self,
            fakd_kwargs: Dict,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            start_level: int = 0,
            end_level: int = -1,
            add_extra_convs: Union[bool, str] = False,
            relu_before_extra_convs: bool = False,
            no_norm_on_lateral: bool = False,
            conv_cfg: OptConfigType = None,
            norm_cfg: OptConfigType = None,
            act_cfg: OptConfigType = None,
            upsample_cfg: ConfigType = dict(mode='nearest'),
            init_cfg: MultiConfig = dict(
                type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(in_channels, out_channels, num_outs, start_level, end_level,
                         add_extra_convs, relu_before_extra_convs, no_norm_on_lateral, conv_cfg, norm_cfg, act_cfg,
                         upsample_cfg, init_cfg)
        """The only specific hyperparameter is fakd_kwargs"""

        check_key_names = ["dirac_ratio", "weight", "encoder_type", "teacher_features"]
        for check_key_name in check_key_names:
            assert check_key_name in fakd_kwargs, f"Must ensure {check_key_name} in fakd_kwargs"

        self.teacher_features = fakd_kwargs["teacher_features"]
        self.dirac_ratio = fakd_kwargs["dirac_ratio"]
        self.weight = fakd_kwargs["weight"]
        self.flow_loss_type = "pkd"
        self.encoder_type = fakd_kwargs["encoder_type"]

        self.flows = []
        for i in range(num_outs):
            flow = FlowAlignModule(
                teacher_channel=self.teacher_features[i],
                teacher_size=None,
                type="feature_based",
                student_channel=out_channels,
                student_size=None,
                sampling=8,
                dirac_ratio=self.dirac_ratio,
                weight=self.weight[i],
                encoder_type=self.encoder_type)
            self.flows.append(flow)
        self.flows = nn.ModuleList(self.flows)

    def forward(self, inputs: Tuple[Tensor], teacher_fpn_outputs: Tuple[Tensor], inference_sampling=4,
                **kwargs) -> tuple:
        """Get the student outputs by calling parent class's forward"""
        student_fpn_outputs = super().forward(inputs)
        # return student_fpn_outputs,Tensor([0.]).to(inputs[0].device)
        # Now all fpn output's sizes are [batch size, channels, scales[i], scales[i]]
        new_student_fpn_outputs = []
        total_loss = Tensor([0.]).to(inputs[0].device)

        if teacher_fpn_outputs is None:
            teacher_fpn_outputs = [None] * len(self.flows)
        for student_fpn_output, teacher_fpn_output, flow in zip(student_fpn_outputs, teacher_fpn_outputs, self.flows):
            loss, new_student_fpn_output = flow(student_fpn_output, teacher_fpn_output, inference_sampling=inference_sampling)
            total_loss += loss
            new_student_fpn_outputs.append(new_student_fpn_output)
        start_len = min(len(teacher_fpn_outputs),len(self.flows))
        end_len = len(student_fpn_outputs)
        for i in range(start_len,end_len):
            new_student_fpn_outputs.append(student_fpn_outputs[i])
        return tuple(new_student_fpn_outputs), total_loss
