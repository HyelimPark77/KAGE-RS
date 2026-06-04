_base_ = './lae_dino_swin-t_finetune_DIOR.py'

model = dict(
    kage_cfg=dict(
        descriptor_path='tools/kage/dior_descriptors.json',
        roi_output_size=7,
        expand_scale=1.5,
        topk=3,
        loss_weight=1.0))
