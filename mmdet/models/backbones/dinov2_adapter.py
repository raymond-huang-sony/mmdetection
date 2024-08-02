import math
import numpy as np
from typing import Any, Dict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from ..utils import as_tuple, to_length
from ..layers import (MSDeformAttn, InteractionBlock, 
                      SpatialPriorModule, deform_inputs)
from .dinov2 import DinoVisionTransformer
from mmengine.model import BaseModule, ModuleDict


class QKVTaskLoRA(nn.Module):
    def __init__(self, qkv: nn.Module, tasks, ranks, dropouts):
        super().__init__()
        self.qkv = qkv
        self.tasks = tasks
        self.ranks = ranks
        self.dropouts = dropouts
        self.dim = qkv.in_features
        self.active_task = 'task_agnostic'

        self.task_loras = nn.ModuleDict(
            {
                task: nn.ModuleDict(
                    {
                        'lora_q_drop': (
                            nn.Identity() if drop == 0 else nn.Dropout(drop)
                        ),
                        'lora_q_a': nn.Linear(self.dim, rank, bias=False),
                        'lora_q_b': nn.Linear(rank, self.dim, bias=False),
                        'lora_v_drop': (
                            nn.Identity() if drop == 0 else nn.Dropout(drop)
                        ),
                        'lora_v_a': nn.Linear(self.dim, rank, bias=False),
                        'lora_v_b': nn.Linear(rank, self.dim, bias=False),
                    }
                )
                for task, rank, drop in zip(tasks, ranks, dropouts)
            }
        )
        with torch.no_grad():
            for name, p in self.task_loras.named_parameters():
                if name.endswith('_a'):
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                elif name.endswith('_b'):
                    nn.init.zeros_(p)

    def forward(self, x):
        qkv = self.qkv(x)

        active_task = {'task_agnostic', self.active_task}
        tasks_with_lora = set(self.task_loras.keys())
        valid_tasks = active_task & tasks_with_lora

        for task in valid_tasks:
            lora_q = self.task_loras[task]['lora_q_b'](
                self.task_loras[task]['lora_q_a'](
                    self.task_loras[task]['lora_q_drop'](x)
                )
            )
            lora_v = self.task_loras[task]['lora_v_b'](
                self.task_loras[task]['lora_v_a'](
                    self.task_loras[task]['lora_v_drop'](x)
                )
            )
            qkv[:, :, : self.dim] += lora_q
            qkv[:, :, -self.dim :] += lora_v
        return qkv


def init_weights(model, pretrained=None, revise_keys=None):
    if isinstance(pretrained, str):
        load_checkpoint(
            model,
            pretrained,
            map_location='cpu',
            strict=False,
            **(
                {'revise_keys': revise_keys}
                if revise_keys is not None
                else {}
            ),
        )


@MODELS.register_module()
class DinoV2Adapter(DinoVisionTransformer):
    def __init__(
        self,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        adapter_init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        add_cls_token=False,
        return_cls_token=False,
        use_extra_extractor=True,
        freeze_backbone=False,
        freeze_adapter=False,
        adapter_patch_size=16,
        adapter_drop_path_rate=0,
        pretrained=None,
        revise_keys=None,
        task_embeddings=(),
        task_prompts=(),
        task_prompt_sizes=200,
        task_loras=(),
        task_lora_ranks=16,
        task_lora_dropouts=0.0,
        active_task='task_agnostic',
        normalize_interaction=False,
        interpolation_up=False,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.add_cls_token = add_cls_token
        self.return_cls_token = return_cls_token
        self.adapter_patch_size = adapter_patch_size
        self.adapter_drop_path_rate = adapter_drop_path_rate
        self.pretrained = pretrained
        self.revise_keys = revise_keys
        self.active_task = active_task
        self.normalize_interaction = normalize_interaction
        assert isinstance(task_embeddings, tuple)
        assert isinstance(task_prompts, tuple)
        self.task_prompt_sizes = to_length(
            as_tuple(task_prompt_sizes), len(task_prompts)
        )
        assert isinstance(task_loras, tuple)
        self.task_loras = task_loras
        self.task_lora_ranks = to_length(
            as_tuple(task_lora_ranks), len(task_loras)
        )
        self.task_lora_dropouts = to_length(
            as_tuple(task_lora_dropouts), len(task_loras)
        )

        init_weights(self, pretrained, revise_keys=revise_keys)

        if len(task_embeddings) > 0:
            self.task_embeddings = nn.ParameterDict(
                {
                    task: nn.Parameter(
                        torch.zeros(
                            self.patch_embed.num_patches
                            + 1
                            + self.num_register_tokens,
                            self.embed_dim,
                        )
                    )
                    for task in task_embeddings
                }
            )
        else:
            self.task_embeddings = None

        if len(task_prompts) > 0:
            self.task_prompts = nn.ParameterDict(
                {
                    task: nn.Parameter(
                        torch.zeros(
                            size,
                            self.embed_dim,
                        )
                    )
                    for task, size in zip(
                        task_prompts,
                        self.task_prompt_sizes,
                    )
                }
            )
            with torch.no_grad():
                for p in self.task_prompts.parameters():
                    trunc_normal_(p, std=0.02)
        else:
            self.task_prompts = None

        if len(task_loras) > 0:
            for block in self.blocks:
                block.attn.qkv = QKVTaskLoRA(
                    block.attn.qkv,
                    self.task_loras,
                    self.task_lora_ranks,
                    self.task_lora_dropouts,
                )

        embed_dim = self.embed_dim
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=embed_dim
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=adapter_init_values,
                    drop_path=self.adapter_drop_path_rate,
                    norm_layer=self.norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    normalize_interaction=self.normalize_interaction,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        if interpolation_up:
            self.up = lambda x: nn.functional.interpolate(
                x, scale_factor=2, mode='bilinear', align_corners=False
            )
        else:
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
            self.up.apply(self._init_weights)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        if freeze_backbone:
            self._freeze_backbone()
        if freeze_adapter:
            self._freeze_adapter()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 += self.level_embed[0]
        c3 += self.level_embed[1]
        c4 += self.level_embed[2]
        return c2, c3, c4

    def set_task_info(self, task_info: Dict[str, Any]):
        self.task_info = task_info
        self.active_task = task_info.get('task_name', 'task_agnostic')

        if len(self.task_loras) > 0:
            for block in self.blocks:
                block.attn.qkv.active_task = self.active_task

    def set_data_info(self, data_info: Dict[str, Any]):
        self.data_info = data_info

    def prepare_task_embeddings(self, x, H, W):
        if self.task_embeddings is None:
            return x

        # pos embed size
        M = int(math.sqrt(self.pos_embed.shape[1] - 1))
        dim = x.shape[-1]

        active_task = {'task_agnostic', self.active_task}
        tasks_with_embedding = set(self.task_embeddings.keys())
        valid_tasks = active_task & tasks_with_embedding

        for task in valid_tasks:
            pre_patch_task_embedding = self.task_embeddings[task][
                None, : -M * M
            ]
            patch_task_embedding = self.task_embeddings[task][-M * M :]

            patch_task_embedding = (
                nn.functional.interpolate(
                    patch_task_embedding.reshape(1, M, M, dim).permute(
                        0, 3, 1, 2
                    ),
                    size=(H, W),
                    mode='bicubic',
                    antialias=self.interpolate_antialias,
                )
                .permute(0, 2, 3, 1)
                .view(1, -1, dim)
            )

            task_embedding = torch.cat(
                (pre_patch_task_embedding, patch_task_embedding), dim=1
            )
            x = x + task_embedding

        return x

    def prepare_task_prompts(self, x):
        if self.task_prompts is None:
            return x

        active_task = {'task_agnostic', self.active_task}
        tasks_with_prompt = set(self.task_prompts.keys())
        valid_tasks = active_task & tasks_with_prompt

        for task in valid_tasks:
            task_prompt = self.task_prompts[task][None].expand(
                x.shape[0], -1, -1
            )
            x = torch.cat((task_prompt, x), dim=1)

        return x

    def forward(self, x, task=None, return_cls_token=None):
        if return_cls_token is None:
            return_cls_token = self.return_cls_token

        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        assert x.shape[-2] % self.adapter_patch_size == 0
        assert x.shape[-1] % self.adapter_patch_size == 0
        H = x.shape[-2] // self.adapter_patch_size
        W = x.shape[-1] // self.adapter_patch_size
        HW = H * W

        # resize img to fit dinov2
        if self.adapter_patch_size != self.patch_size:
            with torch.no_grad():
                x = F.interpolate(
                    x,
                    size=(
                        H * self.patch_size,
                        W * self.patch_size,
                    ),
                    mode='bicubic',
                    antialias=True,
                    align_corners=False,
                )
        x = self.prepare_tokens_with_masks(x)
        x = self.prepare_task_embeddings(x, H, W)
        x = self.prepare_task_prompts(x)
        B, _, C = x.shape

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
                active_task=self.active_task,
            )

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(B, C, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.normalize_interaction:
            with torch.no_grad():
                NORM = (x.norm(dim=-1).mean(dim=-1) / np.sqrt(x.shape[-1]))[
                    :, None, None, None
                ]

        if self.add_vit_feature:
            x3 = x[:, -HW:].transpose(1, 2).view(B, C, H, W).contiguous()
            if self.normalize_interaction:
                x3 = x3 / NORM
            x1 = F.interpolate(
                x3, scale_factor=4, mode='bilinear', align_corners=False
            )
            x2 = F.interpolate(
                x3, scale_factor=2, mode='bilinear', align_corners=False
            )
            x4 = F.interpolate(
                x3, scale_factor=0.5, mode='bilinear', align_corners=False
            )
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        cls_token = x[:, 0, :, None, None]
        if self.normalize_interaction:
            cls_token = cls_token / NORM

        if self.add_cls_token:
            c1, c2, c3, c4 = (
                c1 + cls_token,
                c2 + cls_token,
                c3 + cls_token,
                c4 + cls_token,
            )

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        if return_cls_token:
            return [f1, f2, f3, f4, cls_token]
        return [f1, f2, f3, f4]

    def _freeze_backbone(self):
        print('Freeze DinoV2')
        for blk in self.blocks:
            for param in blk.parameters():
                param.requires_grad = False
        for m in [self.patch_embed, self.norm]:
            if m is not None:
                for param in m.parameters():
                    param.requires_grad = False
        for p in [
            self.pos_embed,
            self.cls_token,
            self.mask_token,
            self.register_tokens,
        ]:
            if p is not None:
                p.requires_grad = False

    def _freeze_adapter(self):
        print('Freeze DinoV2 Adapter')
        for blk in self.interactions:
            for param in blk.parameters():
                param.requires_grad = False
        for m in [
            self.spm,
            self.up,
            self.norm1,
            self.norm2,
            self.norm3,
            self.norm4,
        ]:
            for param in m.parameters():
                param.requires_grad = False
        for p in [
            self.level_embed,
        ]:
            if p is not None:
                p.requires_grad = False

    def init_weights(self, pretrained=None, revise_keys=None):
        if isinstance(pretrained, str):
            load_checkpoint(
                self,
                pretrained,
                map_location='cpu',
                strict=False,
                **(
                    {'revise_keys': revise_keys}
                    if revise_keys is not None
                    else {}
                ),
            )


@MODELS.register_module()
class DistillDinoV2Adapter(BaseModule):

    class NeckWrapper(nn.Module):

        def __init__(self, neck):
            super().__init__()
            self.adapt_from = MODELS.build(neck)

    def __init__(
        self, 
        backbone, 
        teachers,
        pretrained=None,
        revise_keys=None,
    ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        # build teachers
        self.teachers = ModuleDict(
            {key: DistillDinoV2Adapter.NeckWrapper(val)
                for key, val in teachers.items()})
        init_weights(self, pretrained, revise_keys=revise_keys)

    def interup(self, x, size):
        return F.interpolate(x, size=size, mode='bicubic', align_corners=False)

    def interdown(self, x, size):
        return F.interpolate(x, size=size, mode='bicubic', 
                             align_corners=False, antialias=True)

    def forward(self, x, *args, **kwargs):
        _, _, raw_H, raw_W = x.shape
        patch_size = 32
        if raw_H % patch_size == 0 and raw_W % patch_size == 0:
            outs = self.backbone(x, *args, **kwargs)
        else:
            # scale images
            H = patch_size * int(math.ceil(raw_H / patch_size))
            W = patch_size * int(math.ceil(raw_W / patch_size))
            x = self.interup(x, (H, W))
            outs = self.backbone(x, *args, **kwargs)
            # scale feature maps
            outs = [self.interdown(out, 
                    size=(
                        int(math.ceil(raw_H / s)), int(math.ceil(raw_W / s))
                    ))
                    for s, out in zip((4, 8, 16, 32), outs)]
            import pdb;pdb.set_trace()
        outs = [teacher(outs) for _, teacher in self.teachers.items()]
        return outs
