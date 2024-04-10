from torch import nn
import torch
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn
from copy import deepcopy as clone

class ForeBackLearning(nn.Module):
    def __init__(self, norm=None,dropout=None):
        super(ForeBackLearning, self).__init__()
        self.norm = norm
        self.dropout = dropout
        if norm:
            self.fore_norm = norm
            self.back_norm = clone(norm)
        if dropout:
            self.fore_dropout = dropout
            self.back_dropout = clone(dropout)

    def forward(self,patch_feats,cam,logits):
        # 根据预测的标签对cam进行选择
        logits = torch.sigmoid(logits)
        labels = (logits >= 0.5).float()
        cam = labels.unsqueeze(-1) * cam # [32,14,98]
        # 每个空间位置，找到在14个类别上的最大cam值,即这里的前景图
        fore_map, _ = torch.max(cam, dim=1, keepdim=True) # [32,1,98]
        fore_map = self._normalize(fore_map)# 规范化到01之间
        back_map = 1-fore_map
        fore_rep = torch.matmul(fore_map, patch_feats) #[32,1,2048]
        back_rep = torch.matmul(back_map, patch_feats)
        if self.norm:
            # 默认执行
            fore_rep = self.fore_norm(fore_rep)
            back_rep = self.back_norm(back_rep)
        if self.dropout:
            fore_rep = self.fore_dropout(fore_rep)
            back_rep = self.back_dropout(back_rep)
        return fore_rep, back_rep, fore_map.squeeze(1)

    def _normalize(self, cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization."""
        # 最大最小规范化
        cams.sub_(cams.min(-1).values[(..., None)]) # 减去最小值来将cams张量的范围映射到非负数
        cams_max = cams.max(-1).values[(..., None)] # 在上一句的基础上获取最大值
        cams_max[cams_max<1e-12] = 1e-12 # 如果最大值很小，就设置为1e-12，放置除0错误
        cams.div_(cams_max)
        return cams