from torch import nn
import torch
import torch.nn.functional as F
import math

class CamAttnCon(nn.Module):
    # cam attention consistency
    def __init__(self, method = 'mean', topk = 0.1, layer_id = 2, vis = False):
        super(CamAttnCon, self).__init__()
        # weighted sum or max
        self.method = method
        self.sim = nn.CosineSimilarity(dim=2)
        self.topk = topk
        self.layer_id = layer_id
        self.vis = vis

    def forward(self, fore_rep_encoded, target_embed, align_attns, targets):
        # fore_rep_encoded维度是[32,512];target_embed维度是[32,59,512]
        # align_attns包含3个维度是[32,8,59,98]的注意力权重张量;targets维度是[32,60]
        # how to process extra token
        targets = targets.clone()[:, :-1] # 去掉每条数据的最后一个值,[32,59]
        seq_mask = (targets.data > 0)
        seq_mask[:, 0] += True
        attns = align_attns[self.layer_id] # 获得解码器最后一个解码器层的注意力权重值[32,8,59,98]
        attns = torch.mean(attns, dim=1) # 8个头的注意力权重值求平均[32,59,98]
        # 目标文本本就是对图像的解读,前景表示是对图像中异常区域的突出;计算这两者之间的相似度;这里使用的是余弦相似度
        weights = self.sim(target_embed, fore_rep_encoded.unsqueeze(1)) # 计算目标嵌入与前景表示的余弦相似度[32,59]
        weights = weights.masked_fill(seq_mask == 0, -1) # 将无效的位置的相似度值置为-1
        _, idxs = torch.topk(weights, k = int(self.topk*weights.shape[1]), dim = 1) # 取前14个最大的相似度值的索引,idxs维度是[32,14]
        weights = weights.unsqueeze(-1) # [32,59,1]
        attns = F.relu(weights * attns) # 给注意力权重值乘上相似度值,并将负值置为0,相当于只保留了有效位置的注意力权重值
        seq_len = torch.sum(seq_mask, dim=1) # 计算每条数据的有效长度(包含开始符) # [32,]
        true_topk = seq_len * self.topk # 计算每条数据的有效长度乘上topk的值[32,]
        if self.method == 'mean':
            # scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            # weights = F.softmax(scores, dim=1).transpose(-1,-2)
            # total_attn = torch.matmul(weights, attns).squeeze(1)
            total_attn = [self._normalize(torch.mean(attn[idxs[i][:math.ceil(true_topk[i])]], dim=0)).unsqueeze(0) for i, attn in enumerate(attns)]
            #total_attn, _ = torch.mean(attns, dim=1)
        elif self.method == 'max':
            # 默认执行这里
            #scores = torch.matmul(target_embed, fore_rep_encoded.unsqueeze(-1))
            #weights = F.softmax(scores, dim=1)
            # total_attn = [self._normalize(torch.max(attn[idxs[i][:math.ceil(true_topk[i])]], dim=0).values).unsqueeze(0) for i, attn in
            #          enumerate(attns)]
            # 遍历每一条数据的注意力权重值,取前几个最大的值(根据有效长度计算的true_topk),然后取最大值(98个空间位置的元素值来自于这几个最大的目标序列位置里面最大的那个);
            # 然后进行最大最小规范化;结果存储在列表中,每个元素的维度为[1,98]
            total_attn = [self._normalize(torch.max(attn[idxs[i][:math.ceil(true_topk[i])]], dim=0).values).unsqueeze(0) for i, attn in
                     enumerate(attns)]
            total_attn = torch.cat(total_attn, dim=0) # [32,98];是每条数据综合了前景表示和目标文本,选出来的最重要的注意力权重值
            #total_attn, _ = torch.max(attns, dim = 1)
            # print(total_attn.shape, fore_map.shape)
            # print(total_attn[0], fore_map[0])
        else:
            raise NotImplementedError
        if self.vis:
            # 默认不执行
            return total_attn, [idxs[i][:math.ceil(true_topk[i])].detach().cpu() for i in range(len(attns))], align_attns
        return total_attn, None, None


    def _normalize(self, cams):
        """CAM normalization."""
        cams = cams - cams.min(-1, keepdim=True).values
        #cams.sub_(cams.min(-1).values[(..., None)])
        cams_max = cams.max(-1).values[(..., None)]
        cams_max[cams_max<1e-12] = 1e-12
        cams = cams / cams_max
        return cams

