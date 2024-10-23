_base_ = 'grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='DistillDinoV2Adapter',
        backbone=dict(
            type='DinoV2Adapter',
            img_size=518,
            patch_size=14,
            init_values=1.0,
            embed_dim=768,
            ffn_layer='mlp',
            block_chunks=0,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            pretrained='pretrained/dinov2_vitb14_reg4_pretrain.pth',
            adapter_drop_path_rate=0.3,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=12,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
            freeze_backbone=True,
            normalize_interaction=True,
            normalize_interaction_has_grad=True,
            interpolation_up=True,
            extra_cls_token=True,
            adapter_norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        ),
        teachers=dict(
            gdino8x=dict(
                type='SequentialNecks',
                necks=[
                    dict(
                        type='IndexSelect',
                        index=1,
                    ),
                    dict(
                        type='Permute',
                        dims=(0, 2, 3, 1),
                    ),
                    dict(
                        type='MultiLayersPerception',
                        input_size=768,
                        hidden_size=1280,
                        output_size=256,
                    ),
                    dict(
                        type='Permute',
                        dims=(0, 3, 1, 2),
                    )
                ]
            ),
            gdino16x=dict(
                type='SequentialNecks',
                necks=[
                    dict(
                        type='IndexSelect',
                        index=2,
                    ),
                    dict(
                        type='Permute',
                        dims=(0, 2, 3, 1),
                    ),
                    dict(
                        type='MultiLayersPerception',
                        input_size=768,
                        hidden_size=1280,
                        output_size=512,
                    ),
                    dict(
                        type='Permute',
                        dims=(0, 3, 1, 2),
                    )
                ]
            ),
            gdino32x=dict(
                type='SequentialNecks',
                necks=[
                    dict(
                        type='IndexSelect',
                        index=3,
                    ),
                    dict(
                        type='Permute',
                        dims=(0, 2, 3, 1),
                    ),
                    dict(
                        type='MultiLayersPerception',
                        input_size=768,
                        hidden_size=1280,
                        output_size=1024,
                    ),
                    dict(
                        type='Permute',
                        dims=(0, 3, 1, 2),
                    )
                ]
            )
        )
    )
)

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='Pad', size_divisor=32),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, return_classes=True))
test_dataloader = val_dataloader