"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random
sys.path.extend(['../'])
from feeders import tools

# 左右翻转
flip_index = np.concatenate(([0,2,1,4,3,6,5],   # 前7个是body
                             [17,18,19,20,21,22,23,24,25,26],  # 后20个是手
                             [7,8,9,10,11,12,13,14,15,16]), axis=0) 

node_name = 27

class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, random_mirror=False, random_mirror_p=0.5, is_vector=False):
        """
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence   随机选择
        :param random_shift (tools): If true, randomly pad zeros at the begining or end of sequence  # 随机偏移，偏移片段用0填充
        :param random_shift (args): 坐标数据的整体偏移
        :param random_move:  涉及 随机 旋转、平移、缩放
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples  前100个样本
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory  共享内存
        :param random_mirror:  随机镜像，增加 多样性   # 当前帧 是否翻转
        :param random_mirror_p:  随机镜像的阈值
        :param is_vector:  除joint外，bone、Jm、Bm 均为 True
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data()
        self.is_vector = is_vector
        if normalization:
            self.get_mean_map()
        print(len(self.label))

    def load_data(self):
        """
        读取 data_path、label_path
        """
        # data: N C V T M

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load label
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')


        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        """
        计算输入数据的 均值和方差，计划在归一化过程中使用 
        !!! However, 默认不使用
        """
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # 采样
        """
        如果 random_choose: False , 输出的长度不受self.window_size的限制？
        """
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        
        # 增强之翻转
        if self.random_mirror:
            if random.random() > self.random_mirror_p:
                assert data_numpy.shape[2] == node_name
                # （27）关键点顺序 左右body顺序替换，详见 flip_index
                data_numpy = data_numpy[:,:,flip_index,:]

                # 除joint外，bone\joint_motion\bone_motion 均认定为 is_vector: True
                # aka.  仅joint模态 被认定为为 is_vector: False
                if self.is_vector:
                    data_numpy[0,:,:,:] = - data_numpy[0,:,:,:]
                else:   # joint 坐标数据
                    data_numpy[0,:,:,:] = 512 - data_numpy[0,:,:,:]   # 512... 认为输入图大小为恒为512*512 ？？！
        
        # 数据处理之 归一化
        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            assert data_numpy.shape[0] == 3
            if self.is_vector:
                data_numpy[0,:,0,:] = data_numpy[0,:,0,:] - data_numpy[0,:,0,0].mean(axis=0)
                data_numpy[1,:,0,:] = data_numpy[1,:,0,:] - data_numpy[1,:,0,0].mean(axis=0)
            else:   # joint 坐标数据
                data_numpy[0,:,:,:] = data_numpy[0,:,:,:] - data_numpy[0,:,0,0].mean(axis=0)
                data_numpy[1,:,:,:] = data_numpy[1,:,:,:] - data_numpy[1,:,0,0].mean(axis=0)

        # 数据增强之 随机移动
        if self.random_shift:
            if self.is_vector:
                data_numpy[0,:,0,:] += random.random() * 20 - 10.0
                data_numpy[1,:,0,:] += random.random() * 20 - 10.0
            else:   # joint 坐标数据
                data_numpy[0,:,:,:] += random.random() * 20 - 10.0
                data_numpy[1,:,:,:] += random.random() * 20 - 10.0

        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)

        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # print("data_numpy ", data_numpy.shape)    # (3, 100, 27, 1)
        # import pdb
        # pdb.set_trace()
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


# def test(data_path, label_path, vid=None, graph=None, is_3d=False):
#     '''
#     vis the samples using matplotlib
#     :param data_path: 
#     :param label_path: 
#     :param vid: the id of sample
#     :param graph: 
#     :param is_3d: when vis NTU, set it True
#     :return: 
#     '''
#     import matplotlib.pyplot as plt
#     loader = torch.utils.data.DataLoader(
#         dataset=Feeder(data_path, label_path),
#         batch_size=64,
#         shuffle=False,
#         num_workers=2)

#     if vid is not None:
#         sample_name = loader.dataset.sample_name
#         sample_id = [name.split('.')[0] for name in sample_name]
#         index = sample_id.index(vid)
#         data, label, index = loader.dataset[index]
#         data = data.reshape((1,) + data.shape)

#         # for batch_idx, (data, label) in enumerate(loader):
#         N, C, T, V, M = data.shape

#         plt.ion()
#         fig = plt.figure()
#         if is_3d:
#             from mpl_toolkits.mplot3d import Axes3D
#             ax = fig.add_subplot(111, projection='3d')
#         else:
#             ax = fig.add_subplot(111)

#         if graph is None:
#             p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
#             pose = [
#                 ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
#             ]
#             ax.axis([-1, 1, -1, 1])
#             for t in range(T):
#                 for m in range(M):
#                     pose[m].set_xdata(data[0, 0, t, :, m])
#                     pose[m].set_ydata(data[0, 1, t, :, m])
#                 fig.canvas.draw()
#                 plt.pause(0.001)
#         else:
#             p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
#             import sys
#             from os import path
#             sys.path.append(
#                 path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
#             G = import_class(graph)()
#             edge = G.inward
#             pose = []
#             for m in range(M):
#                 a = []
#                 for i in range(len(edge)):
#                     if is_3d:
#                         a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
#                     else:
#                         a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
#                 pose.append(a)
#             ax.axis([-1, 1, -1, 1])
#             if is_3d:
#                 ax.set_zlim3d(-1, 1)
#             for t in range(T):
#                 for m in range(M):
#                     for i, (v1, v2) in enumerate(edge):
#                         x1 = data[0, :2, t, v1, m]
#                         x2 = data[0, :2, t, v2, m]
#                         if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
#                             pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
#                             pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
#                             if is_3d:
#                                 pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
#                 fig.canvas.draw()
#                 # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
#                 plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    # test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
