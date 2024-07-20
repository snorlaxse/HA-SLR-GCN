import sys

from .tools import get_spatial_graph

num_node = 7
self_link = [(i, i) for i in range(num_node)]  # 自旋图
inward_ori_index = [

                # (鼻子，肩膀)
                (5, 6), (5, 7),  

                # (肩膀，手肘)
                (6, 8), (7, 9), 

                # 手肘 - 手腕
                (8, 10), (9, 11)]     # 5-11

inward = [(i - 5, j - 5) for (i, j) in inward_ori_index]   # （方向）向内   # 偏移5，可能是最小下标是5
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
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        # plt.imshow(i, cmap='gray')
        # plt.show()
        plt.imsave('./check_27_body.png',i) 
    # print(A)
    # print(A.shape)  # (3, 27, 27)
