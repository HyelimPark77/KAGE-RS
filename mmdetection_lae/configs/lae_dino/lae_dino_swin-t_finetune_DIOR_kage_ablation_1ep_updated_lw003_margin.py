_base_ = './lae_dino_swin-t_finetune_DIOR_kage_ablation_1ep_updated.py'

model = dict(
    kage_cfg=dict(
        loss_weight=0.03,
        margin=0.2,
        margin_weight=0.5,
        score_beta=0.1,
        stats_path='../work_dirs/kage_descriptor_stats/dior_stats_updated_lw003_margin_1ep.json'))
