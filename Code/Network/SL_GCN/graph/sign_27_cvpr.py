from .tools import get_spatial_graph

num_node = 27
self_link = [(i, i) for i in range(num_node)]  # 自旋图
inward_ori_index = [
                # (鼻子，眼睛)
                (5, 6), (5, 7),  

                # (鼻子，肩膀)
                (5, 8), (5, 9), 

                # 肩膀 - 手肘
                (8, 10), (9, 11),
                
                # 12-21 (-5) 7-16  左手
                (12,13),(12,14),(12,16),(12,18),(12,20),
                (14,15),(16,17),(18,19),(20,21),

                # 22-31 (-5) 17-26  右手
                (22,23),(22,24),(22,26),(22,28),(22,30),
                (24,25),(26,27),(28,29),(30,31),

                # (手肘, 手掌)
                (10,12),(11,22)]     # 5-31

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

    A = Graph('spatial').get_adjacency_matrix() # (3, 27, 27)
    
    # 逐层可视化
    for i in range(A.shape[0]):
        plt.imsave('./sign_27_cvpr_{}.png'.format(i), A[i])  # (27, 27)
    
    # 整体可视化
    plt.imsave('./sign_27_cvpr.png',A.transpose(1,2,0)) # (27, 27, 3)
