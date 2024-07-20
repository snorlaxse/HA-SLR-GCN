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

def gendata(data_path, label_path, out_path, is_unified_frames = True, part='train', config='27'):
    """
    
    """
    ####################### 标签 #####################
    data=[]
    sample_names = []
    labels = []
    label_file = open(label_path, 'r', encoding='utf-8')
    
    # 读取并使用csv标签文件
    for line in label_file.readlines():
        line = line.strip()
        line = line.split(',')

        sample_names.append(line[0])
        data.append(os.path.join(data_path, line[0] + '.npy'))
        # print(line[1])
        labels.append(int(line[1]))
        # print(labels[-1])
    
    # 记录 为
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, labels), f)

    ###################### 提取关键点并保存 #######################

    selected = selected_joints[config]
    num_joints = len(selected)

    if is_unified_frames:
        fp = np.zeros((len(data), max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)
        # print(fp.shape)  # ({split_num}, 150, , 3, 1)  xxx: 28142\4418\3742
    else: 
        # fp = np.asarray([])
        # print(fp.shape) 
        fp = []

    over_max_frames = 0
    for i, data_path in enumerate(data):  # 每个i 代表1例视频

        # print(sample_names[i])
        print("data_path ", data_path)
        skel = np.load(data_path)

        """
        选取关键点信息
        """
        # print(skel.shape)  # (xx, 133, 3)
        skel = skel[:,selected,:]  # 仅选取关键点信息
        # print(skel.shape)   # (xx, 27, 3)
        # pdb.set_trace()

        if is_unified_frames:
            """
            统一视频长度
            """
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
                over_max_frames += 1
                pdb.set_trace()
        else:
            fp.append(skel)

        # print("video frame ", L)
        # print("fp.shape ", fp.shape) 

    print("over_max_frames ", over_max_frames)
    fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print("fp.shape ", fp.shape)   # (xx, 3, 150, 27, 1)  xx: 28142\4418\3742  
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sign Data Converter.')
    # parser.add_argument('--data_path', default='/data/sign/test_npy/npy3') #'train_npy/npy3', 'va_npy/npy3'
    # parser.add_argument('--label_path', default='../data/sign/27/test_labels_pseudo.csv') # 'train_labels.csv', 'val_gt.csv'
    parser.add_argument('--out_folder', default='../data/sign_include_wo_unified_frames/')
    parser.add_argument('--is_unified_frames', default=True)
    # parser.add_argument('--points', default='27')
    arg = parser.parse_args()

    pointsl = ['27_cvpr']
    

    for points in pointsl:

        arg.points = points
        out_path = os.path.join(arg.out_folder, arg.points)
        print(out_path) # ../data/sign/27

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # parts = ['train', 'val', 'test']
        all_splits = ['train', 'test']
        data_root = "/raid/zhengjian/OpenHands/datasets/INCLUDE/all_npy/"

        for splits in all_splits:

            print(splits)
            label_path = "/raid/zhengjian/OpenHands/datasets/INCLUDE/INCLUDE_{}_wo_Second(Number).csv".format(splits)

            gendata(
                data_root,
                label_path,
                out_path,
                is_unified_frames=arg.is_unified_frames,
                part=splits,
                config=arg.points)
                
            pdb.set_trace()