_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# training schedule for 2x
train_cfg = dict(max_epochs=24)
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
            encoder_type="convnext_detection")))

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
