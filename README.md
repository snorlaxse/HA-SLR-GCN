# HA-SLR-GCN
## Data preparation
The processed keypoints data can be downloaded [HERE]().

After downloading, put it in the `Code/Network/SL_GCN/data`.

## Usage

```
$ cd Code/Network/SL_GCN/

<!-- TRAIN -->
$ python main_base.py --config config/sign_cvpr_A_hands/AUTSL/train_joint_autsl.yaml
$ python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_include.yaml

<!-- TEST -->
$ python main_base.py --config config/sign_cvpr_A_hands/AUTSL/test_joint_autsl.yaml
$ python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_include.yaml
```