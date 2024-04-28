_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]

model = dict(
    type='FlowAlignmentRetinaNet',
    neck=dict(
        type='FlowAlignmentFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=5,
        add_extra_convs='on_input',
        fakd_kwargs=dict(
            teacher_features=[256] * 5,
            dirac_ratio=1,
            weight=[10] * 4 + [0] * 1,
            encoder_type="conv_detection")))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
