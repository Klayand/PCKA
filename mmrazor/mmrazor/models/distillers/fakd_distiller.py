# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from inspect import signature
from typing import Dict, List, Optional, Union

from mmengine.model import BaseModel
from torch import nn

from mmrazor.registry import MODELS
from ..algorithms.base import LossResults
from ..task_modules import DistillDeliveryManager, RecorderManager
from .base_distiller import BaseDistiller


@MODELS.register_module()
class SingleFlowAlignmentDistiller(BaseDistiller):
    def __init__(self,
                 teacher_recorders: Optional[Dict[str, Dict]] = None,
                 distill_deliveries: Optional[Dict[str, Dict]] = None,
                 connectors: Optional[Dict[str, Dict]] = None,
                 loss_forward_mappings: Optional[Dict[str, Dict]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        # The recorder manager is just constructed, but not really initialized
        # yet. Recorder manager initialization needs to input the corresponding
        # model.
        self.teacher_recorders = RecorderManager(teacher_recorders)

        self.deliveries = DistillDeliveryManager(distill_deliveries)

        self.connectors = self.build_connectors(connectors)

        if loss_forward_mappings:
            self.loss_forward_mappings = loss_forward_mappings
        else:
            self.loss_forward_mappings = dict()

    def set_deliveries_override(self, override: bool) -> None:
        """Set the `override_data` of all deliveries."""
        self.deliveries.override_data = override

    def prepare_from_teacher(self, model: nn.Module) -> None:
        """Initialize teacher recorders."""
        self.teacher_recorders.initialize(model)

    def prepare_from_student(self, model: nn.Module) -> None:
        pass

    def build_connectors(
        self,
        connectors: Optional[Union[Dict[str, List], Dict[str, Dict]]] = None,
    ) -> nn.ModuleDict:
        """Initialize connectors."""

        distill_connecotrs = nn.ModuleDict()
        if connectors:
            for connector_name, connector_cfg in connectors.items():
                if isinstance(connector_cfg, dict):
                    connector = MODELS.build(connector_cfg)
                    distill_connecotrs[connector_name] = connector
                else:
                    assert isinstance(connector_cfg, list)
                    module_list = []
                    for cfg in connector_cfg:
                        connector = MODELS.build(cfg)
                        module_list.append(connector)
                    distill_connecotrs[connector_name] = nn.Sequential(
                        *module_list)

        return distill_connecotrs

    def get_record(self,
                   recorder: str,
                   from_student: bool,
                   record_idx: int = 0,
                   data_idx: Optional[int] = None,
                   connector: Optional[str] = None,
                   connector_idx: Optional[int] = None) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``."""

        if from_student:
            raise NotImplementedError("In FAKD, we do not need to implement the student recorder!")
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)
        record_data = recorder_.get_record_data(record_idx, data_idx)

        if connector:
            record_data = self.connectors[connector](record_data)
        if connector_idx is not None:
            record_data = record_data[connector_idx]

        return record_data

    def compute_distill_losses(self) -> LossResults:
        pass