_base_ = [
    '../../../_base_/datasets/mmdet/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FlowAlignmentFpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::configs/mmdet/faster_rcnn/faster-rcnn_r50_fpn_2x_coco_fakd.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::configs/mmdet/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py',
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
                teacher_feature=dict(from_student=False, recorder='fpn',data_idx=3)))))

find_unused_parameters = True
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=2.5))