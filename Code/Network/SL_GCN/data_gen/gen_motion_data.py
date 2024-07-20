from tqdm import tqdm
import os
import numpy as np
from numpy.lib.format import open_memmap

all_splits = {
    'train', 'val', 'test'  # autsl
    # 'train', 'test'  # include
}

# datasets = {
#     'sign/27_2'
# }

parts = {
    'joint', 'bone'
}

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Motion Data Converter.')
    parser.add_argument('--datasets', default='sign_gsl/27_cvpr')  # sign_autsl/27_cvpr sign/27_cvpr   sign/27_2  sign/hands  sign/body_27  sign_lsa64/27_cvpr
    arg = parser.parse_args()

    dataset = arg.datasets
    # for dataset in datasets:
    for splits in all_splits:
        for part in parts:
            print(dataset, set, part)
            data = np.load('../data/{}/{}_data_{}.npy'.format(dataset, splits, part))
            N, C, T, V, M = data.shape
            print(data.shape)
            fp_sp = open_memmap(
                '../data/{}/{}_data_{}_motion.npy'.format(dataset, splits, part),
                dtype='float32',
                mode='w+',
                shape=(N, C, T, V, M))
            for t in tqdm(range(T - 1)):
                fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]  # core code
            fp_sp[:, :, T - 1, :, :] = 0  # 最后一帧的偏移 设为0
