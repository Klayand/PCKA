# Copyright (c) OpenMMLab. All rights reserved.
from .datafree_distillation import (DAFLDataFreeDistillation,
                                    DataFreeDistillation)
from .fpn_teacher_distill import FpnTeacherDistill
from .overhaul_feature_distillation import OverhaulFeatureDistillation
from .self_distill import SelfDistill
from .single_teacher_distill import SingleTeacherDistill
from .fpn_teacher_fakd_distill import FlowAlignmentFpnTeacherDistill

__all__ = [
    'SelfDistill', 'SingleTeacherDistill', 'FpnTeacherDistill',
    'DataFreeDistillation', 'DAFLDataFreeDistillation',
    'OverhaulFeatureDistillation', 'FlowAlignmentFpnTeacherDistill'
]
