_base_ = [
    '../../../_base_/datasets/mmdet/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_2x_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::retinanet/retinanet_x101-64x4d_fpn_1x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='OFDLoss', loss_weight=1.0),
            loss_pkd_fpn1=dict(type='OFDLoss', loss_weight=1.0),
            loss_pkd_fpn2=dict(type='OFDLoss', loss_weight=1.0),
            # loss_pkd_fpn3=dict(type='ATLoss_patch', loss_weight=1.0)
        ),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                s_feature=dict(from_student=True, recorder='fpn', data_idx=0),
                t_feature=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                s_feature=dict(from_student=True, recorder='fpn', data_idx=1),
                t_feature=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                s_feature=dict(from_student=True, recorder='fpn', data_idx=2),
                t_feature=dict(from_student=False, recorder='fpn', data_idx=2)),
            # loss_pkd_fpn3=dict(
            #     s_feature=dict(from_student=True, recorder='fpn', data_idx=3),
            #     t_feature=dict(from_student=False, recorder='fpn',
            #                  data_idx=3))
        )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))