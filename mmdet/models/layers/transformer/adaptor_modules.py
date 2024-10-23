from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from ..ffn import ConvFFN
from ..ms_deform_attn import MSDeformAttn
from timm.models.layers import DropPath

from dinov2_layers.attention import MemEffCrossAttention


def get_reference_points(spatial_shapes, device, dtype):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5,
                H_ - 0.5,
                H_,
                dtype=dtype,
                device=device,
            ),
            torch.linspace(
                0.5,
                W_ - 0.5,
                W_,
                dtype=dtype,
                device=device,
            ),
            indexing='ij',
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // 16, w // 16)], x.device, x.dtype
    )
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor(
        [(h // 16, w // 16)], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        x.device,
        x.dtype,
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
        moe_cfg=None,
        adapter_use_deform_attn=True,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.adapter_use_deform_attn = adapter_use_deform_attn
        if self.adapter_use_deform_attn:
            self.attn = MSDeformAttn(
                d_model=dim,
                n_levels=n_levels,
                n_heads=num_heads,
                n_points=n_points,
                ratio=deform_ratio,
                moe_cfg=moe_cfg,
                attn_moe=(moe_cfg and moe_cfg.extractor_attn_moe),
            )
        else:
            self.attn = MemEffCrossAttention(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp

        self.moe_cfg = moe_cfg
        if (moe_cfg and moe_cfg.ffn_moe) or with_cffn:
            if moe_cfg and moe_cfg.ffn_moe:
                print('Use MoE FFN')
                hidden_features = dim
                if moe_cfg.use_conv_ffn:
                    hidden_features = int(dim * cffn_ratio)

                self.ffn = MoEFFN(
                    in_features=dim,
                    hidden_features=hidden_features,
                    drop=drop,
                    moe_cfg=moe_cfg,
                )
            else:
                self.ffn = ConvFFN(
                    in_features=dim,
                    hidden_features=int(dim * cffn_ratio),
                    drop=drop,
                )

            self.ffn_norm = norm_layer(dim)
            self.drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(
        self,
        query,
        reference_points,
        feat,
        spatial_shapes,
        level_start_index,
        H,
        W,
        **kwargs,
    ):
        def _inner_forward(query, feat):

            if self.adapter_use_deform_attn:
                attn = self.attn(
                    self.query_norm(query),
                    reference_points,
                    self.feat_norm(feat),
                    spatial_shapes,
                    level_start_index,
                    None,
                    **kwargs,
                )
            else:
                attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            query = query + attn

            if (self.moe_cfg and self.moe_cfg.ffn_moe) or self.with_cffn:
                query = query + self.drop_path(
                    self.ffn(self.ffn_norm(query), H, W, **kwargs)
                )
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
        with_cp=False,
        moe_cfg=None,
        adapter_use_deform_attn=True,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.adapter_use_deform_attn = adapter_use_deform_attn
        if self.adapter_use_deform_attn:
            self.attn = MSDeformAttn(
                d_model=dim,
                n_levels=n_levels,
                n_heads=num_heads,
                n_points=n_points,
                ratio=deform_ratio,
                moe_cfg=moe_cfg,
                attn_moe=(moe_cfg and moe_cfg.injector_attn_moe),
            )
        else:
            self.attn = MemEffCrossAttention(dim)
        self.gamma = nn.Parameter(
            init_values * torch.ones(dim), requires_grad=True
        )

    def forward(
        self,
        query,
        reference_points,
        feat,
        spatial_shapes,
        level_start_index,
        **kwargs,
    ):
        def _inner_forward(query, feat):

            if self.adapter_use_deform_attn:
                attn = self.attn(
                    self.query_norm(query),
                    reference_points,
                    self.feat_norm(feat),
                    spatial_shapes,
                    level_start_index,
                    None,
                    **kwargs,
                )
            else:
                attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            return query + self.gamma.to(query.dtype) * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
        normalize_interaction=False,
        normalize_interaction_has_grad=False,
        moe_cfg=None,
        adapter_use_deform_attn=True,
    ):
        super().__init__()

        self.normalize_interaction = normalize_interaction
        self.normalize_interaction_has_grad = normalize_interaction_has_grad

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
            moe_cfg=moe_cfg,
            adapter_use_deform_attn=adapter_use_deform_attn,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
            moe_cfg=moe_cfg,
            adapter_use_deform_attn=adapter_use_deform_attn,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                        moe_cfg=moe_cfg,
                        adapter_use_deform_attn=adapter_use_deform_attn,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(
        self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, **kwargs
    ):
        HW = H * W

        x_pre = x[:, :-HW]
        x_patches = x[:, -HW:]
        if self.normalize_interaction:
            if self.normalize_interaction_has_grad:
                NORM = x_patches.norm(dim=-1, keepdim=True).mean(
                    dim=1, keepdim=True
                ) / np.sqrt(x_patches.shape[-1])
            else:
                with torch.no_grad():
                    NORM = x_patches.norm(dim=-1, keepdim=True).mean(
                        dim=1, keepdim=True
                    ) / np.sqrt(x_patches.shape[-1])
            x_patches = x_patches / NORM
        x_patches = self.injector(
            query=x_patches,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
            **kwargs,
        )
        if self.normalize_interaction:
            x_patches = x_patches * NORM
        x = torch.cat((x_pre, x_patches), dim=1)

        for _, blk in enumerate(blocks):
            x = blk(x, H, W)

        x_patches = x[:, -HW:]
        if self.normalize_interaction:
            if self.normalize_interaction_has_grad:
                NORM = x_patches.norm(dim=-1, keepdim=True).mean(
                    dim=1, keepdim=True
                ) / np.sqrt(x_patches.shape[-1])
            else:
                with torch.no_grad():
                    NORM = x_patches.norm(dim=-1, keepdim=True).mean(
                        dim=1, keepdim=True
                    ) / np.sqrt(x_patches.shape[-1])
            x_patches = x_patches / NORM
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x_patches,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W,
            **kwargs,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x_patches,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                    **kwargs,
                )

        return x, c


class SpatialPriorModule(nn.Module):
    def __init__(
        self,
        inplanes=64,
        embed_dim=384,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ):
        super().__init__()

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(
                    3, inplanes, kernel_size=3, stride=2, padding=1, bias=False
                ),
                build_norm_layer(norm_cfg, inplanes)[-1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes,
                    inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, inplanes)[-1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes,
                    inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, inplanes)[-1],
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, 2 * inplanes)[-1],
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, 4 * inplanes)[-1],
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, 4 * inplanes)[-1],
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv2d(
            inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc2 = nn.Conv2d(
            2 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fc3 = nn.Conv2d(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fc4 = nn.Conv2d(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4