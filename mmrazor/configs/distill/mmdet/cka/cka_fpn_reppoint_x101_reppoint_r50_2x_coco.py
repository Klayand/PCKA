_base_ = [
    '../../../_base_/datasets/mmdet/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::reppoints/reppoints-moment_r50_fpn-gn_head-gn_2x_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::reppoints/reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='CenterKernelAlignmentRKD', intra_weight=10),
            loss_pkd_fpn1=dict(type='CenterKernelAlignmentRKD', intra_weight=10),
            loss_pkd_fpn2=dict(type='CenterKernelAlignmentRKD', intra_weight=10),
            loss_pkd_fpn3=dict(type='CenterKernelAlignmentRKD', intra_weight=10)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=0),
                teacher_logits=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=1),
                teacher_logits=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=2),
                teacher_logits=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_pkd_fpn3=dict(
                student_logits=dict(from_student=True, recorder='fpn', data_idx=3),
                teacher_logits=dict(from_student=False, recorder='fpn',
                             data_idx=3)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
