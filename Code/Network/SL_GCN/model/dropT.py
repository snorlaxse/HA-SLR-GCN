import pdb
import torch
import torch.nn.functional as F
from torch import nn

class DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        
        self.keep_prob = keep_prob
        # print("self.keep_prob ", self.keep_prob)
        
        # 非训练阶段 不做任何drop处理
        if not self.training or self.keep_prob == 1:
            return input
        
        n,c,t,v = input.size()  # torch.Size([64, 128, 50, 27])

        # torch.Size([64, 128, 50, 27]) -> torch.Size([64, 128, 50]) -> torch.Size([64, 50])
        # dim=3: 对keypoints求均值
        # dim=1: 对channel求均值
        input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1).detach()
        
        # 百分比
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n,1,t)  # torch.Size([64, 1, 50])
        
        # 系数（数值较小）
        gamma = (1. - self.keep_prob) / self.block_size
        
        input1 = input.permute(0,1,3,2).contiguous().view(n,c*v,t)  # torch.Size([64, 3456, 50])
        # 从伯努利分布中提取二进制随机数（`0或1`）
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)    # torch.Size([64, 3456, 50])
        
        # maxpooling
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)   # torch.Size([64, 3456, 50])
        
        # mask 表示 要保留 的点 
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)    # torch.Size([64, 3456, 50])
        
        # view 
        # torch.Size([64, 3456, 50]) -> torch.Size([64, 128, 27, 50]) -> torch.Size([64, 128, 50, 27])
        return (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)
