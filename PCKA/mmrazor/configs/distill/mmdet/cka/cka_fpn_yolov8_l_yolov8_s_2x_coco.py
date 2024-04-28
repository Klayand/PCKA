_base_ = "mmyolo::/home/chenhuanran2022/work/mm/mmrazor/configs/mmyolo/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"
#"/home/chenhuanran2022/work/mm/mmrazor/configs/mmyolo/_base_/default_runtime.py",
#"/home/chenhuanran2022/work/mm/mmrazor/configs/mmyolo/_base_/det_p5_tta.py"


# dataset settings
dataset_type = 'YOLOv5CocoDataset'
# dd_data_root = '/root/dataset/dd_coco/'
data_root = '/data1/chenhuanran2022/COCO2017/'

train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 4
# persistent_workers must be False if num_workers is 0
persistent_workers = True
# the number of epochs used for stage2
num_epochs_stage2 = 160
# the number of total epochs
max_epochs = 500  # Maximum training epochs
# learning rate
base_lr = 0.01

weight_decay = 0.0005


# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

img_scale = (640, 640)

random_resize_ratio_range = (0.5, 1.0)

# train_pipeline = [
#     dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
#     dict(_scope_='mmyolo', type='LoadAnnotations', with_bbox=True),
#     dict(
#         _scope_='mmyolo',
#         img_scale=(
#             640,
#             640,
#         ),
#         pad_val=114.0,
#         pre_transform=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True),
#         ],
#         # use_cached=True,
#         # max_cached_images=40,
#         type='Mosaic'),
#     dict(
#         _scope_='mmyolo',
#         border=(
#             -320,
#             -320,
#         ),
#         border_val=(
#             114,
#             114,
#             114,
#         ),
#         max_rotate_degree=0.0,
#         max_shear_degree=0.0,
#         scaling_ratio_range=(
#             0.5,
#             1.5,
#         ),
#         type='YOLOv5RandomAffine'),
#     dict(
#         _scope_='mmyolo',
#         bbox_params=dict(
#             format='pascal_voc',
#             label_fields=[
#                 'gt_bboxes_labels',
#                 'gt_ignore_flags',
#             ],
#             type='BboxParams'),
#         keymap=dict(gt_bboxes='bboxes', img='image'),
#         transforms=[
#             dict(p=0.01, type='Blur'),
#             dict(p=0.01, type='MedianBlur'),
#             dict(p=0.01, type='ToGray'),
#             dict(p=0.01, type='CLAHE'),
#         ],
#         type='mmdet.Albu'),
#     dict(_scope_='mmyolo', type='YOLOv5HSVRandomAug'),
#     dict(_scope_='mmyolo', prob=0.5, type='mmdet.RandomFlip'),
#     dict(
#         _scope_='mmyolo',
#         meta_keys=(
#             'img_id',
#             'img_path',
#             'ori_shape',
#             'img_shape',
#             'flip',
#             'flip_direction',
#         ),
#         type='mmdet.PackDetInputs'),
# ]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    collate_fn=dict(_delete_=True, type='yolov5_collate'),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(min_size=32),
        pipeline=_base_.train_pipeline))

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
#     dict(type='YOLOv5KeepRatioResize', scale=img_scale),
#     dict(
#         type='LetterResize',
#         scale=img_scale,
#         allow_scale_up=False,
#         pad_val=dict(img=114)),
#     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'pad_param'))
# ]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=_base_.test_pipeline,
        batch_shapes_cfg=_base_.batch_shapes_cfg))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

teacher_ckpt = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'

norm_cfg = dict(type='BN')

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::/home/chenhuanran2022/work/mm/mmrazor/configs/mmyolo/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmyolo::/home/chenhuanran2022/work/mm/mmrazor/configs/mmyolo/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        # `recorders` are used to record various intermediate results during
        # the model forward.
        student_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck'),
            # fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            # fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2')
        ),
        teacher_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck'),
            # fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            # fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2')
        ),
        # `connectors` are adaptive layers which usually map teacher's and
        # students features to the same dimension.
        connectors=dict(
            fpn0_s=dict(
                type='ConvModuleConnector',
                in_channel=128,
                out_channel=256,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn0_t=dict(
                type='NormConnector', in_channels=256, norm_cfg=norm_cfg),
            fpn1_s=dict(
                type='ConvModuleConnector',
                in_channel=256,
                out_channel=512,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn1_t=dict(
                type='NormConnector', in_channels=512, norm_cfg=norm_cfg),
            fpn2_s=dict(
                type='ConvModuleConnector',
                in_channel=512,
                out_channel=512,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn2_t=dict(
                type='NormConnector', in_channels=512, norm_cfg=norm_cfg)
        ),
        distill_losses=dict(
            loss_fpn0=dict(type='CenterKernelAlignmentRKD', intra_weight=15),
            #loss_fpn1=dict(type='CenterKernelAlignmentRKD', intra_weight=15),
            # loss_fpn2=dict(type='CenterKernelAlignmentRKD', intra_weight=15)
        ),
        # `loss_forward_mappings` are mappings between distill loss forward
        # arguments and records.
        loss_forward_mappings=dict(
            loss_fpn0=dict(
                student_logits=dict(
                    from_student=True, recorder='fpn0', connector='fpn0_s', data_idx=0),
                teacher_logits=dict(
                    from_student=False, recorder='fpn0', connector='fpn0_t', data_idx=0)),
            #loss_fpn1=dict(
                #student_logits=dict(
                    #from_student=True, recorder='fpn0', connector='fpn1_s', data_idx=1),
                #teacher_logits=dict(
                    #from_student=False, recorder='fpn0', connector='fpn1_t', data_idx=1)),
            # loss_fpn2=dict(
            #     student_logits=dict(
            #         from_student=True, recorder='fpn0', connector='fpn2_s', data_idx=2),
            #     teacher_logits=dict(
            #         from_student=False, recorder='fpn0', connector='fpn2_t', data_idx=2
            #         ))
        )))

find_unused_parameters = True

save_epoch_intervals = 10
val_interval_stage2 = 1

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        val_interval_stage2)])

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='SGD',
#         lr=base_lr,
#         momentum=0.937,
#         weight_decay=weight_decay,
#         nesterov=True,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#     constructor='YOLOv5OptimizerConstructor')

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=_base_.base_lr,
        momentum=0.937,
        weight_decay=_base_.weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

# learning rate
param_scheduler = None

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]


val_cfg = dict(
    _delete_=True,
    type='mmrazor.SingleTeacherDistillValLoop'
)
