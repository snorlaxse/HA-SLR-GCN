import argparse
import pickle
from tqdm import tqdm
import sys
import numpy as np
import os
import pdb

sys.path.extend(['../'])

selected_joints = {
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],
                    [112,116,117,120,121,124,125,128,129,132]), axis=0), #27

    # 选取的27个关键骨骼点   # 0 - 132 顺序取点
    '27_2': np.concatenate(([0,5,6,7,8,9,10],   # noise\shoulders\elbows\wrists
                    [91,95,96,99,100,103,104,107,108,111],
                    [112,116,117,120,121,124,125,128,129,132]), axis=0), #27_2

    '27_cvpr': np.concatenate(([0,3,4,5,6,7,8],     # [0,1,2,3,4,5,6]  # noise\eyes\shoulders\elbows
                    [91,95,96,99,100,103,104,107,108,111],  # [7,8,9,10,11,12,13,14,15,16]  # 7-16 (+5)  12-21
                    [112,116,117,120,121,124,125,128,129,132]), axis=0), # [17,18,19,20,21,22,23,24,25,26]  # 27_cvpr 
    'hands':   np.concatenate(([91,95,96,99,100,103,104,107,108,111],  # [7,8,9,10,11,12,13,14,15,16]  # 7-16 (+5)  12-21
                    [112,116,117,120,121,124,125,128,129,132]), axis=0),
    'body_27':    [0,5,6,7,8,9,10]
}

max_body_true = 1  #...
max_frame = 150
num_channels = 3  #...

def gendata(data_path, label_path, out_path, part='train', config='27'):
    labels = []
    data=[]
    sample_names = []
    selected = selected_joints[config]
    num_joints = len(selected)
    label_file = open(label_path, 'r', encoding='utf-8')
    
    # 读取并使用csv标签文件
    for line in label_file.readlines():
        line = line.strip()
        line = line.split(',')

        sample_names.append(line[0])
        data.append(os.path.join(data_path, line[0] + '_color.mp4.npy'))
        # print(line[1])
        labels.append(int(line[1]))
        # print(labels[-1])

    fp = np.zeros((len(data), max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)
    print(fp.shape)  # ({split_num}, 150, , 3, 1)  xxx: 28142\4418\3742

    for i, data_path in enumerate(data):  # 每个i 代表1例视频

        # print(sample_names[i])
        skel = np.load(data_path)
        # print(skel.shape)  # (xx, 133, 3)
        skel = skel[:,selected,:]  # 仅选取关键点信息
        # print(skel.shape)   # (xx, 27, 3)
        # pdb.set_trace()

        if skel.shape[0] < max_frame:
            L = skel.shape[0]
            # :L
            fp[i,:L,:,:,0] = skel  # 填充前L帧信息
            
            rest = max_frame - L
            num = int(np.ceil(rest / L))  # 向上取整
            pad = np.concatenate([skel for _ in range(num)], 0)[:rest]  # 重复num次，循环填充
            # L:
            fp[i,L:,:,:,0] = pad  # 填充L帧后面的信息

        else:
            # 若视频帧数 大于 max_frame, 则截取前150帧信息即可
            L = skel.shape[0]
            fp[i,:,:,:,0] = skel[:max_frame,:,:]

        # print("video frame ", L)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, labels), f)

    fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print(fp.shape)   # (xx, 3, 150, 27, 1)  xx: 28142\4418\3742  
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sign Data Converter.')
    # parser.add_argument('--data_path', default='/data/sign/test_npy/npy3') #'train_npy/npy3', 'va_npy/npy3'
    # parser.add_argument('--label_path', default='../data/sign/27/test_labels_pseudo.csv') # 'train_labels.csv', 'val_gt.csv'
    parser.add_argument('--out_folder', default='../data/sign_autsl/')
    parser.add_argument('--points', default='27_cvpr')
    arg = parser.parse_args()

    # pointsl = ['body_27', 'hands']

    # for points in pointsl:

    out_path = os.path.join(arg.out_folder, arg.points)
    print(out_path) # ../data/sign/27

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    parts = ['train', 'val', 'test']
    # data_paths = ["../dataset/train_npy3", "../dataset/val_npy3", "../dataset/test_npy3"]
    data_paths = ["/raid/zhengjian/Isolated_SLR/dataset/train_npy3", 
                "/raid/zhengjian/Isolated_SLR/dataset/val_npy3", 
                "/raid/zhengjian/Isolated_SLR/dataset/test_npy3"]
    label_paths = ["../data/sign/train_labels.csv", 
                "../data/sign/val_labels.csv", 
                "../data/sign/ground_truth.csv"]
    for i in range(3):
        gendata(
            data_paths[i],
            label_paths[i],
            out_path,
            part=parts[i],
            config=arg.points)
