import argparse
import pickle

import numpy as np
from tqdm import tqdm

"""
python ensemble_finetune.py 
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3742/3742 [00:00<00:00, 16548.39it/s]
3742
top1:  0.9690005344735435
top5:  0.9978621058257616
"""
label = open('../test_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('joint_finetuned.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('bone_finetuned.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('joint_motion_finetuned.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('bone_motion_finetuned.pkl', 'rb')
r4 = list(pickle.load(r4).items())
r5 = open('../baseline/gcn_ensembled.pkl', 'rb')  # 另一个评判维度
r5 = list(pickle.load(r5).items())

alpha = [1.2,1.2,0.5,0.5,1.5] # used in submissions 3

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0
# print(len(label[0]))

with open('predictions_finetuned.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        name5, r55 = r5[i]
        assert name == name1 == name2 == name3 == name4
        mean += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3] + r55*alpha[4]) / np.array(alpha).sum()
        # score = r11*alpha[0] 
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        pred = np.argmax(score)
        scores.append(score)
        preds.append(pred)
        right_num += int(pred == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, pred))
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

with open('./test_gcn_w_val_finetune.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)