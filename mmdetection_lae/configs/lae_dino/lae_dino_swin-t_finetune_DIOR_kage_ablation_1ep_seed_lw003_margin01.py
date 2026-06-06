_base_ = './lae_dino_swin-t_finetune_DIOR_kage_ablation_1ep_seed.py'

model = dict(
    kage_cfg=dict(
        loss_weight=0.03,
        margin=0.2,
        margin_weight=0.1,
        score_beta=0.1,
        stats_path='../work_dirs/kage_descriptor_stats/dior_stats_seed_lw003_margin01_1ep.json'))
