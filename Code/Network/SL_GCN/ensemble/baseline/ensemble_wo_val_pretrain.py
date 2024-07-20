import argparse
import pickle

import numpy as np
from tqdm import tqdm
import pdb

"""
# test 
$ python ensemble_wo_val.py
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3742/3742 [00:00<00:00, 14660.78it/s]
3742
top1:  0.9647247461250668   # little worse than baseline_cvpr
top5:  0.9975948690539819   # little worse than baseline_cvpr

$ python ensemble_wo_val_pretrain.py  
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3742/3742 [00:00<00:00, 14694.59it/s]
3742
top1:  0.9647247461250668
top5:  0.9975948690539819
"""
# label = open('./test_labels_pseudo.pkl', 'rb')
label = open('../test_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./joint.pkl', 'rb')  
r1 = list(pickle.load(r1).items())  # len: 3742  # [(sample_name(str), (226, )(ndarray))]
r2 = open('./bone.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('./joint_motion.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('./bone_motion.pkl', 'rb')
r4 = list(pickle.load(r4).items())

alpha = [1.0,0.9,0.5,0.5] # used in submission 1  # ensemble 权重

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0

with open('predictions_wo_val.csv', 'w') as f:

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
        # pdb.set_trace()

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)

f.close()
# print(mean/len(label[0]))
# with open('./val_pred.pkl', 'wb') as f:
#     # score_dict = dict(zip(names, preds))
#     score_dict = (names, preds)
#     pickle.dump(score_dict, f)

with open('./gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)