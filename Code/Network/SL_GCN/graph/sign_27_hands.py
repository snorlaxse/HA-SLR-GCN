import sys

from .tools import get_spatial_graph

num_node = 20
self_link = [(i, i) for i in range(num_node)]  # 自旋图
inward_ori_index = [
                    (12,13),(12,14),(12,16),(12,18),(12,20),
                    (14,15),(16,17),(18,19),(20,21),
                    (22,23),(22,24),(22,26),(22,28),(22,30),
                    (24,25),(26,27),(28,29),(30,31),
                    ]     # 5-31

bias = 12   # 偏移，inward_ori_index 最小序号
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

    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    plt.imsave('./check_hands.png', A.transpose(1,2,0))
