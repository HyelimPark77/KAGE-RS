_base_ = './lae_dino_swin-t_finetune_DIOR_kage.py'

load_from = '../checkpoints/lae_dino_pretrain.pth'

max_epochs = 1
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[],
        gamma=0.1)
]

model = dict(
    kage_cfg=dict(
        descriptor_path='../work_dirs/kage_descriptor_stats/dior_descriptors_merged.json',
        stats_path='../work_dirs/kage_descriptor_stats/dior_stats_updated_1ep.json'))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'))
