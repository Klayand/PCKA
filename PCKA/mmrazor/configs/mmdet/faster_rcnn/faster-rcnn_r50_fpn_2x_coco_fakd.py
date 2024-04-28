_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FlowAlignmentFasterRCNN',
    neck=dict(
        type='FlowAlignmentFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        fakd_kwargs=dict(
            teacher_features=[256] * 5,
            dirac_ratio=1.0,
            weight=[6] * 4 + [0] * 1,
            encoder_type="convnext_detection")))
