from .tools import get_spatial_graph

num_node = 27
self_link = [(i, i) for i in range(num_node)]  # 自旋图
inward_ori_index = [
                    # 12-21 (-5) 7-16  左手
                    (12,13),(12,14),(12,16),(12,18),(12,20),
                    (14,15),(16,17),(18,19),(20,21),
                    
                    # 22-31 (-5) 17-26  右手
                    (22,23),(22,24),(22,26),(22,28),(22,30),
                    (24,25),(26,27),(28,29),(30,31),
                    
                    # 左右掌心
                    (12, 22)
                    ]     

bias = 5   # 偏移，inward_ori_index 最小序号   # 与 sign_27_cvpr.py 匹配
inward = [(i - bias, j - bias) for (i, j) in inward_ori_index]   # （方向）向内   
outward = [(j, i) for (i, j) in inward]   # （方向）向外
neighbor = inward + outward  # 双向（邻近）


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    A = Graph('spatial').get_adjacency_matrix()   # (3, 27, 27)
    
    # 逐层可视化
    for i in range(A.shape[0]):
        plt.imsave('./sign_27_A_hands_{}.png'.format(i), A[i]) 
        
    # 整体可视化
    plt.imsave('./sign_27_A_hands.png',A.transpose(1,2,0)) # (27, 27, 3)
    
