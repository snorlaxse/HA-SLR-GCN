import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .Adaptive_DropGraph import Adaptive_DropGraph
import os


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


# def conv_branch_init(conv):
#     weight = conv.weight
#     n = weight.size(0)
#     k1 = weight.size(1)
#     k2 = weight.size(2)
#     nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
#     nn.init.constant(conv.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        # dropGraph
        # self.dropS = DropBlock_Ske(num_point=num_point)
        # self.dropT = DropBlockT_1d(block_size=block_size)
        self.adrop = Adaptive_DropGraph(num_point=num_point, block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        # x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        x = self.adrop(x, keep_prob, A)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, A_hands, num_subset=3):
        super(unit_gcn, self).__init__()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.A_hands = torch.tensor(A_hands, dtype=torch.float32, requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32))
        self.PA_hands = nn.Parameter(torch.from_numpy(A_hands.astype(np.float32)))
        self.beta = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32))
        
        self.num_subset = num_subset

        self.conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv[i], self.num_subset)

    def forward(self, x):
        A = self.A.cuda(x.get_device())
        A_hands = self.A_hands.cuda(x.get_device())
        PA_hands = self.PA_hands.cuda(x.get_device())
        A = A + self.PA + A_hands * self.alpha + PA_hands * self.beta

        y = None
        for i in range(self.num_subset):
            f = self.conv[i](x)
            N, C, T, V = f.size()
            z = torch.matmul(f.view(N, C * T, V), A[i]).view(N, C, T, V)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.res(x)    # !!!
        return self.relu(y)

class TCN_GCN_unit(nn.Module):   # SL-GCN Block
    def __init__(self, in_channels, out_channels, A, A_hands, num_point, block_size, stride=1, residual=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        num_jpts = A.shape[-1]
        self.gcn1 = unit_gcn(in_channels, out_channels, A, A_hands)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride, num_point=num_point)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(in_channels, out_channels, kernel_size=1, stride=stride)
            
        # dropGraph
        # self.dropSke = DropBlock_Ske(num_point=num_point)
        # self.dropT_skip = DropBlockT_1d(block_size=block_size)
        self.adrop = Adaptive_DropGraph(num_point=num_point, block_size=block_size)
        
        self.attention = attention
        if attention:
            print('Attention Enabled!')
            self.sigmoid = nn.Sigmoid()
            # s attention 
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)  # 一维卷积
            nn.init.xavier_normal_(self.conv_sa.weight)  # Xavier 初始化
            nn.init.constant_(self.conv_sa.bias, 0)
            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)     # 一维卷积
            nn.init.constant_(self.conv_ta.weight, 0)  # torch.nn.init.constant_(tensor, val): 使用val的值来填充输入的Tensor
            nn.init.constant_(self.conv_ta.bias, 0)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)  # 
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0) 

    def forward(self, x, keep_prob):
        y = self.gcn1(x)  # N C T V  # ex. torch.Size([64, 64, 100, 27])
        if self.attention:  # STC Module
            # spatial attention
            se = y.mean(-2)  # N C V # 对时间维度做均值 # ex. torch.Size([64, 64, 27])
            se1 = self.sigmoid(self.conv_sa(se))  # se1:即注意力权重  # ex. torch.Size([64, 1, 27])   
            y = y * se1.unsqueeze(-2) + y  # *+ # y: torch.Size([64, 64, 100, 27])
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)     # 对空间维度做均值  # ex. torch.Size([64, 64, 100])
            se1 = self.sigmoid(self.conv_ta(se))    # ex. torch.Size([64, 1, 100])
            y = y * se1.unsqueeze(-1) + y  # *+ # se1.unsqueeze(-1): torch.Size([64, 1, 100, 1])
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)  # ex. torch.Size([64, 64])
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)  # torch.Size([64, 64, 1, 1])
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
             
        y = self.tcn1(y, keep_prob, self.A)
        # x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        # x_skip = self.adrop(self.residual(x), keep_prob, self.A)
        x_skip = self.residual(x)
        return self.relu(y + x_skip)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, A_hands=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)    
            self.graph = Graph(**graph_args)

        if A_hands is None:
            raise ValueError()
        else:
            Graph = import_class(A_hands)    
            self.A_hands = Graph(**graph_args)


        A = self.graph.A    # 邻接矩阵  (3, 27, 27)
        A_hands = self.A_hands.A    # 邻接矩阵  (3, 27, 27)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, A_hands, num_point, block_size, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A, A_hands, num_point, block_size)
        self.l3 = TCN_GCN_unit(64, 64, A, A_hands, num_point, block_size)
        self.l4 = TCN_GCN_unit(64, 64, A, A_hands, num_point, block_size)
        self.l5 = TCN_GCN_unit(64, 128, A, A_hands, num_point, block_size, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A, A_hands, num_point, block_size)
        self.l7 = TCN_GCN_unit(128, 128, A, A_hands, num_point, block_size)
        self.l8 = TCN_GCN_unit(128, 256, A, A_hands, num_point, block_size, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A, A_hands, num_point, block_size)
        self.l10 = TCN_GCN_unit(256, 256, A, A_hands, num_point, block_size)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):

        N, C, T, V, M = x.size()
        # print("(N, C, T, V, M) {}, {}, {}, {}, {} ".format( N, C, T, V, M))  # N, C, T, V, M  64 3 100 27 1
        # import pdb
        # pdb.set_trace()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # 64, 3, 100, 27

        x = self.l1(x, keep_prob=1.0)
        x = self.l2(x, keep_prob=1.0)
        x = self.l3(x, keep_prob=1.0)
        x = self.l4(x, keep_prob=1.0)
        x = self.l5(x, keep_prob=1.0)
        x = self.l6(x, keep_prob=1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)  
        # print("after l10 ", x.shape)   # torch.Size([64, 256, 25, 27])
        # N*M,C,T,V
        c_new = x.size(1)   # c_new:  256

        # print(x.size())
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        x = x.reshape(N, M, c_new, -1)  # torch.Size([64, 1, 256, 675])
        # print("x.shape ", x.shape)
        # import pdb
        # pdb.set_trace()
        x = x.mean(3).mean(1)  # torch.Size([64, 1, 256, 675]) → torch.Size([64, 1, 256])  → torch.Size([64, 256])
        # print(x.shape)  # torch.Size([64, 256])
        
        return self.fc(x)
