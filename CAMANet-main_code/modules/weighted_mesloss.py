from torch.nn import MSELoss
import torch

class Weighted_MSELoss(torch.nn.Module):

    def __init__(self, weight = False):
        super().__init__()
        self.weight = weight
        if not self.weight:
            self.criterion = MSELoss()

    def forward(self, total_attn, fore_map, logits=None, labels = None):
        if self.weight:
            p_logits = torch.sigmoid(logits)
            scores = (p_logits * labels).sum(dim=-1)
            scale = labels.sum(dim=-1)
            weights = torch.empty_like(scores).fill_(0)
            index = (scale != 0).nonzero().squeeze(-1)
            weights [index] = scores[index] / scale[index]
            assert total_attn.shape == fore_map.shape
            assert total_attn.size(0) == weights.size(0)
            B, C = total_attn.size(0), total_attn.shape[-1]
            loss = torch.cdist(total_attn, fore_map, p=2).sum(dim=-1)
            loss = weights * loss
            loss = loss.sum() / (B * C)
            return loss
        else:
            # 默认执行这里
            scale = labels.sum(dim=-1) # 沿着最后一个维度的方向求和,即对每条数据的14个标签求和
            # index = (scale != 0).nonzero().squeeze(-1) # 统计非零元素的索引
            index=torch.nonzero((scale != 0),as_tuple=False).squeeze(-1)
            # fore_map是模型对图像的需要重点关注区域的提取;total_attn是模型对文本的重点关注区域的提取;
            # 计算两者之间的欧式距离,然后求和,再除以总的元素个数
            total_attn, fore_map = total_attn[index], fore_map[index]
            return self.criterion(total_attn, fore_map)

