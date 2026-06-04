_base_ = './lae_dino_swin-t_finetune_DIOR.py'

model = dict(
    kage_cfg=dict(
        descriptor_path='../tools/kage/dior_descriptors.json',
        roi_output_size=7,
        expand_scale=1.5,
        topk=3,
        loss_weight=0.01,
        score_beta=0.2,
        stats_path='../work_dirs/kage_descriptor_stats/dior_stats.json',
        stats_interval=100))
