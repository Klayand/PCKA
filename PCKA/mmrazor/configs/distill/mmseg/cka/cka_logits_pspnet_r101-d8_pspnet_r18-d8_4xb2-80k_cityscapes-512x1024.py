_base_ = [
    'mmseg::_base_/datasets/cityscapes.py',
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'  # noqa: E501
teacher_cfg_path = 'mmseg::pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py'  # noqa: E501
student_cfg_path = 'mmseg::pspnet/pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_cwd=dict(type='CenterKernelAlignmentSeg', intra_weight=10, inter_weight=10)),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                student_logits=dict(from_student=True, recorder='logits'),
                teacher_logits=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
