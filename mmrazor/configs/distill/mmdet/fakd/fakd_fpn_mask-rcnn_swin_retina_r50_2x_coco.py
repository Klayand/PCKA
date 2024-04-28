_base_ = [
    '../../../_base_/datasets/mmdet/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]
teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FlowAlignmentFpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::configs/mmdet/retinanet/retinanet_r50_fpn_2x_coco_fakd.py',
        pretrained=False),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet::configs/mmdet/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
        distiller=dict(
        type='SingleFlowAlignmentDistiller',
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        loss_forward_mappings=dict(
            loss_fakd_fpn0=dict(
                student_feature=dict(from_student=True, recorder='fpn', data_idx=0),
                teacher_feature=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_fakd_fpn1=dict(
                student_feature=dict(from_student=True, recorder='fpn', data_idx=1),
                teacher_feature=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_fakd_fpn2=dict(
                student_feature=dict(from_student=True, recorder='fpn', data_idx=2),
                teacher_feature=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_fakd_fpn3=dict(
                student_feature=dict(from_student=True, recorder='fpn', data_idx=3),
                teacher_feature=dict(from_student=False, recorder='fpn',data_idx=3))))
    )
find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))

# dataset
val_evaluator = dict(metric=['bbox'])
test_evaluator = val_evaluator

