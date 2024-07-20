# Skeleton Based Sign Language Recognition
## Data preparation
1. Extract whole-body keypoints data following the instruction in ./Preprocess
2. Run the following code to prepare the data for GCN.

        cd data_gen/
        python sign_gendata.py  --points 27_cvpr
        python gen_bone_data.py  --datasets sign/27_cvpr 
        python gen_motion_data.py  --datasets sign/27_cvpr 

## Usage

### Train:
```
python main.py --config config/sign_cvpr/train/train_joint.yaml

python main.py --config config/sign_cvpr/train/train_bone.yaml

python main.py --config config/sign_cvpr/train/train_joint_motion.yaml

python main.py --config config/sign_cvpr/train/train_bone_motion.yaml
```
### Finetune:
```
python main.py --config config/sign_cvpr/finetune/train_joint.yaml

python main.py --config config/sign_cvpr/finetune/train_bone.yaml

python main.py --config config/sign_cvpr/finetune/train_joint_motion.yaml

python main.py --config config/sign_cvpr/finetune/train_bone_motion.yaml
```
### Test:
```
python main.py --config config/sign_cvpr/test/test_joint.yaml

python main.py --config config/sign_cvpr/test/test_bone.yaml

python main.py --config config/sign_cvpr/test/test_joint_motion.yaml

python main.py --config config/sign_cvpr/test/test_bone_motion.yaml
```
### Test Finetuned:
```
python main.py --config config/sign_cvpr/test_finetuned/test_joint.yaml

python main.py --config config/sign_cvpr/test_finetuned/test_bone.yaml

python main.py --config config/sign_cvpr/test_finetuned/test_joint_motion.yaml

python main.py --config config/sign_cvpr/test_finetuned/test_bone_motion.yaml
```
### Multi-stream ensemble:
1. Copy the results .pkl files from all streams (joint, bone, joint motion and bone motion) to ../ensemble/gcn and renamed them correctly.
2. Follow the instruction in ../ensemble/gcn to obtained the results of multi-stream ensemble.

## 说明

### data_gen

- sign_gendata.py  （joint）

> 从133个关节点中 顺序提取 需要的 27个关键点

- gen_bone_data.py

> 由 joint 信息 计算bone数据

- gen_motion_data.py

> 由 joint 信息 计算motion数据

注释：
N, C, T, V, M = data.shape  ex. (28142 3 150 27 1)

- N: 视频样例数  ex. 28142\4418\3742
- C: sign_gendata.py/num_channels --- 3
- T: sign_gendata.py/max_frame --- 150
- V: 关节点数 27  xxx  7\20
- M: sign_gendata.py/max_body_true --- 1

### graph 

> 构建 关节点 的图关系（自环、单向边、双向边）

![](./graph/check.png)

### feed

> 数据输入预处理

### config

> 训练配置

- 实验名称
- feeder 训练集、验证集
- 预训练权重

### work_dir

> 存放 实验记录信息

注意：
- 使用同样的配置训练，训练所得的每次log信息一模一样，非常稳定

