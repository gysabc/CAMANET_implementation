import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn
import numpy as np

class CAM:
    def __init__(self, normalized = False, relu = True):
        self.normalized = normalized
        self.relu = relu

    def compute_scores(self,patch_feats, fc_layer, class_idx):
        weights = self._get_weights(fc_layer, class_idx) # 获取分类器的权重[14,2048]
        with torch.no_grad():
            # 是 PyTorch 中的一个上下文管理器，用于禁用自动梯度计算
            # 在这个上下文中的任何计算都不会影响网络的参数更新
            # 同时，如果你在这个上下文中使用了网络的某些模块，那么这些模块的参数在反向传播时不会被更新
        # n_cam = weights.shape[0]
        #patch_feats = patch_feats.unsqueeze(1).expand(patch_feats.shape[0], n_cam, patch_feats.shape[1],patch_feats.shape[2])
            cams = torch.matmul(patch_feats, weights.transpose(-2,-1)).transpose(-2,-1) # 使用矩阵乘法计算类激活图
        # print(cams.shape)
        #
        #
        #     for weight, activation in zip(weights, patch_feats):
        #         # missing_dims = activation.ndim - weight.ndim  # type: ignore[union-attr]
        #         # weight = weight[(...,) + (None,) * missing_dims]
        #
        #         # Perform the weighted combination to get the CAM
        #         cam = torch.nansum(weight * activation, dim=1)  # type: ignore[union-attr]


            if self.relu:
                # 默认执行，用于将CAM中的负贡献值置为0(与我的代码的区别，确实应该这么做)
                cams = F.relu(cams, inplace=True)
        # Normalize the CAM
            if self.normalized:
                # 默认执行;具体为最大最小规范化
                cams = self._normalize(cams)

            #cams.append(cam)
        return cams

    @staticmethod
    @torch.no_grad()
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization."""
        # 最大最小规范化
        cams.sub_(cams.min(-1).values[(..., None)])
        cams_max = cams.max(-1).values[(..., None)]
        cams_max[cams_max<1e-12] = 1e-12
        cams.div_(cams_max)
        return cams


    @torch.no_grad()
    def _get_weights(self,fc_layer, class_idx):
        fc_weights = fc_layer.weight.data # 分类器的权重[14,2048]
        if fc_weights.ndim > 2:
            fc_weights = fc_weights.view(*fc_weights.shape[:2])
        # 根据class_idx的类型来选择不同的操作
        if isinstance(class_idx, int):
            # 如果class_idx是整数，那么就选择fc_weights的特定行并增加一个维度
            return fc_weights[class_idx, :].unsqueeze(0)
        else:
            # 如果class_idx不是整数（可能是一个列表或者数组），那么就选择fc_weights的多行。
            return fc_weights[class_idx, :]


