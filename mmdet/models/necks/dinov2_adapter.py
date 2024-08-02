from typing import List, Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmengine.model import ModuleList


@MODELS.register_module()
class SequentialNecks(nn.Module):
    """Sequential necks

    Passes inputs through multiple necks one by one
    then return the outputs of the last neck
    """

    def __init__(self, necks: List[Dict]):
        super().__init__()
        self.necks = ModuleList([
            MODELS.build(neck) for neck in necks
        ])

    def forward(self, inputs):
        for neck in self.necks:
            inputs = neck(inputs)
        return inputs


@MODELS.register_module()
class MultiLayersPerception(nn.Module):

    def __init__(
            self,
            input_size: int, 
            hidden_size: int, 
            output_size: int,
            num_inner: int = 0,
            pre_norm: bool = False,
            post_norm: bool = False,
        ):
        super().__init__()

        self.pre_norm = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
        ) if pre_norm else nn.Identity()

        self.fc1 = nn.Linear(input_size, hidden_size)

        blocks = []
        for _ in range(num_inner):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        self.post_norm = nn.LayerNorm(output_size) if post_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)
        x = self.post_norm(x)
        return x


@MODELS.register_module()
class IndexSelect(nn.Module):

    def __init__(self, index: Union[int, str]):
        super().__init__()
        self.index = index

    def forward(self, inputs):
        if isinstance(self.index, (list, tuple)):
            return [inputs[i] for i in self.index]
        return inputs[self.index]


@MODELS.register_module()
class Reshape(nn.Module):

    def __init__(self, shape: Tuple[int]):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.reshape(*self.shape)


@MODELS.register_module()
class Transpose(nn.Module):

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, inputs):
        return inputs.transpose(self.dim0, self.dim1).contiguous()


@MODELS.register_module()
class Permute(nn.Module):

    def __init__(self, dims: Union[List[int], Tuple[int]]):
        super().__init__()
        self.dims = dims
    
    def forward(self, inputs):
        return torch.permute(inputs, self.dims).contiguous()


@MODELS.register_module()
class TorchNnModule(nn.Module):

    def __init__(self, module, *args, **kwargs):
        assert hasattr(nn, module), \
            f"Failed to find module {module} in torch.nn"
        super().__init__()
        self.module = getattr(nn, module)(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)