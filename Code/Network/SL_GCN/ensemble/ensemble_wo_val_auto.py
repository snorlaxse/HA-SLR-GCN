import argparse
import pickle

import numpy as np
from tqdm import tqdm
import pdb
import glob, os

"""
"""

tag = '27_baseline'      # '27'  'cvpr'
tag1l = ['acc', 'loss']   #  'acc'  'loss'
tag2l = ['val', 'test'] 

d1 = f"joint_{tag}_test"
d2 = f"bone_{tag}_test"
d3 = f"joint_motion_{tag}_test"
d4 = f"bone_motion_{tag}_test"
d = [d1, d2, d3, d4]

rl = [None, None, None, None]

save_dir = f"predictions_wo_val_{tag}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for tag1 in tag1l:
    for tag2 in tag2l:
        with open(f"predictions_wo_val_{tag}/ensemble_log.txt", 'a') as f:
            print(f"predictions_wo_val_{tag}_best_{tag1}_{tag2}", file=f)
            for i in range(4):
                for fp in glob.glob(os.path.join(f"../work_dir/{d[i]}/eval_results/", "*.pkl")):
                    if tag1 in fp and f"_{tag2}.pkl" in fp:
                        print(f"{i} {fp} ")
                        print(f"{i} {fp} ", file=f)
                        rl[i] = fp
                        break
        f.close()

        r1 = open(rl[0], 'rb')
        r1 = list(pickle.load(r1).items())
        r2 = open(rl[1], 'rb')
        r2 = list(pickle.load(r2).items())
        r3 = open(rl[2], 'rb')  # 'rb': 读取二进制文件
        r3 = list(pickle.load(r3).items())
        r4 = open(rl[3], 'rb')
        r4 = list(pickle.load(r4).items())

        label = open(f"{tag2}_label.pkl", 'rb')
        label = np.array(pickle.load(label))

        alpha = [1.0,0.9,0.5,0.5] # used in submission 1  # ensemble 权重

        right_num = total_num = right_num_5 = 0
        names = []
        preds = []
        scores = []
        mean = 0

        with open(f"predictions_wo_val_{tag}/predictions_wo_val_{tag}_best_{tag1}_{tag2}.csv", 'w') as f:

            for i in tqdm(range(len(label[0]))):
                name, l = label[:, i]
                names.append(name)
                name1, r11 = r1[i]
                name2, r22 = r2[i]
                name3, r33 = r3[i]
                name4, r44 = r4[i]
                assert name == name1 == name2 == name3 == name4

                mean += r11.mean()   # 有何意义
                score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum()
                # score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).mean()
                # score = r11*alpha[0] 
                rank_5 = score.argsort()[-5:]
                right_num_5 += int(int(l) in rank_5)  # 判断正确标签是否在rank_5中

                pred = np.argmax(score)
                scores.append(score)
                preds.append(pred)
                right_num += int(pred == int(l))

                total_num += 1

                f.write('{}, {}\n'.format(name, pred))

        acc = right_num / total_num
        acc5 = right_num_5 / total_num

        with open(f"predictions_wo_val_{tag}/ensemble_log.txt", 'a') as f:
            print(total_num, file=f)
            print("top1: {:.2f}%".format(100*acc), file=f)
            print("top5: {:.2f}%".format(100*acc5), file=f)
            print("", file=f)
        f.close()

        # print(mean/len(label[0]))
        # with open('predictions_wo_val_{tag}/val_pred.pkl', 'wb') as f:
        #     # score_dict = dict(zip(names, preds))
        #     score_dict = (names, preds)
        #     pickle.dump(score_dict, f)

        # with open(f"predictions_wo_val_{tag}/gcn_ensembled_{tag}_best_{tag1}_{tag2}.pkl", 'wb') as f:
        #     score_dict = dict(zip(names, scores))
        #     pickle.dump(score_dict, f)