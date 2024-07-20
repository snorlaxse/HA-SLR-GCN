import torch
import torch.nn.functional as F
from torch import nn
import warnings


class DropBlock_Ske(nn.Module):
    """ 
    Spatial DropGraph  --- 作用于 邻接矩阵
    """
    
    def __init__(self, num_point, block_size=7):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point

    def forward(self, input, keep_prob, A):  
        """
        n,c,t,v 
        
        n -- batchsize
        c -- channel
        t -- frame
        v -- keypoint
        """
        
        self.keep_prob = keep_prob
        
        # 非训练阶段 不做任何drop处理
        if not self.training or self.keep_prob == 1:
            return input   

        # input：attention map
        n, c, t, v = input.size()  # torch.Size([16, 128, 50, 27])

        # torch.Size([16, 128, 50, 27])  torch.abs() -- 求正值（绝对值）
        # → torch.Size([16, 128, 27])  torch.mean(, dim=2) -- 求均值
        # → torch.Size([16, 27]   torch.mean(, dim=1) -- 求均值 
        input_abs = torch.mean(torch.mean(torch.abs(input), dim=2), dim=1).detach() 

        # torch.numel: 获取张量元素个数numel()
        # 求 keypoints（均值）百分比
        # input_abs: the normalized attention map
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()

        # drop_size = 1 + \sum{i=1..K} B_i
        # γ = (1−keep prob) / drop_size  
        # γ: probability
        if self.num_point == 25:  # Kinect V2
            gamma = (1. - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:  # Kinect V1
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:   # maybe 27
            gamma = (1. - self.keep_prob) / (1 + 1.92)   # gamma： 找到删除哪些点的系数 gamma值极小
            warnings.warn('undefined skeleton graph')
            
        # torch.clamp 将大于max的 置为1        
        # torch.bernoulli: 从伯努利分布中提取二进制随机数（`0或1`），即 mask to drop
        #                  输入张量应为包含用于绘制二进制随机数的`概率`的张量  
        M_seed = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)     # torch.Size([16, 27])
        
        # 弱化 A 中边的连接强弱关系
        # M：Drop 之后的 A
        M = torch.matmul(M_seed, A)  

        # M是纯0，1矩阵
        # M值大于0.001的都保留
        # Bool is function setting non-zero element root to 1
        M[M > 0.001] = 1.0  
        M[M < 0.5] = 0.0   
        # 仅小于等于 0.001 的元素 设为0

        
        # mask 表示 要保留 的点 ---  drop mask
        mask = (1 - M).view(n, 1, 1, self.num_point)

        # input * mask：返回要保留的点 
        return input * mask / mask.sum() * mask.numel() 
