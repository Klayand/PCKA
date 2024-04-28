# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional,Dict,Union
import torch

from mmengine.structures import BaseDataElement
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS

from ...base import LossResults
from .fpn_teacher_distill import FpnTeacherDistill


@MODELS.register_module()
class FlowAlignmentFpnTeacherDistill(FpnTeacherDistill):

    def loss(
            self,
            batch_inputs: torch.Tensor,
            data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()
        # If the `override_data` of a delivery is False, the delivery will
        # record the origin data.
        self.distiller.set_deliveries_override(False)

        # Unlike ``SingleTeacherDistill``, teacher will only execute
        # back + neck, not head, so there will be no loss.
        if self.teacher_trainable:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                _ = self.teacher.extract_feat(batch_inputs)
        else:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                with torch.no_grad():
                    _ = self.teacher.extract_feat(batch_inputs)

        # We need to obtain all teacher fpn's output throughing Recorders...
        teacher_fpn_outputs = []
        for loss_name, forward_mappings in self.distiller.loss_forward_mappings.items():
            for forward_key, record in forward_mappings.items():
                if forward_key == "teacher_feature":
                    forward_var = self.distiller.get_record(**record)
                    teacher_fpn_outputs.append(forward_var)

        # If the `override_data` of a delivery is True, the delivery will
        # override the origin data with the recorded data.
        self.distiller.set_deliveries_override(True)
        with self.distiller.deliveries:
            student_losses = self.student(
                batch_inputs, data_samples, teacher_fpn_outputs, mode='loss')
        losses.update(add_prefix(student_losses, 'student'))
        return losses
