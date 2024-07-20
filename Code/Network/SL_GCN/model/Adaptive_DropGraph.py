import random
import torch
import torch.nn as nn
from .dropSke import DropBlock_Ske
from .dropT import DropBlockT_1d

class Adaptive_DropGraph(nn.Module):
    """ 
    import Gated mechanism on DropGraph
    """
    def __init__(self, num_point=27, block_size=41):
        super(Adaptive_DropGraph, self).__init__()
    
        # dropGraph
        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

        # Gated mechanism
        self.gamma = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32))  
        self.delta = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32)) 
        
    def forward(self, x, keep_prob, A):
        
        a1 = self.gamma * self.dropS(x, keep_prob, A)
        y = a1 + (1 - self.gamma) * x  
        
        a2 = self.delta * self.dropT(y, keep_prob) 
        y = a2 + (1 - self.delta) * y  
        
        return y
