_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

# default_scope = 'mmrazor'
teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::gfl/gfl_r50_fpn_1x_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::gfl/gfl_r101_fpn_ms-2x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_cwd_fpn0=dict(
                type='CenterKernelAlignmentRKD', intra_weight=10, inter_weight=10),
            loss_cwd_fpn1=dict(
                type='CenterKernelAlignmentRKD', intra_weight=10, inter_weight=10),
            loss_cwd_fpn2=dict(
                type='CenterKernelAlignmentRKD', intra_weight=10, inter_weight=10),
            loss_cwd_fpn3=dict(
                type='CenterKernelAlignmentRKD', intra_weight=10, inter_weight=10)),
        loss_forward_mappings=dict(
            loss_cwd_fpn0=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=0),
                teacher_logits=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_cwd_fpn1=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=1),
                teacher_logits=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_cwd_fpn2=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=2),
                teacher_logits=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_cwd_fpn3=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=3),
                teacher_logits=dict(from_student=False, recorder='fpn', data_idx=3)),
                )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
