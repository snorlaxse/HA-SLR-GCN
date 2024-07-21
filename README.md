# HA-SLR-GCN
## Data preparation
The processed keypoints data can be downloaded [HERE](https://drive.google.com/drive/folders/1LM6gpmtgrcUvXdDkyKGVTQuDtqfViMKz?usp=drive_link).

After downloading and unpacking it, place it in the `Code/Network/SL_GCN/data/`.

## Requirements
Install some packages as follows.
```
torch
torchvision
torchaudio  
tqdm
tensorboard
pyyaml
pandas
```

## Usage

```
$ cd Code/Network/SL_GCN/

<!-- TRAIN -->
$ python main_base.py --config config/sign_cvpr_A_hands/AUTSL/train_joint_autsl.yaml
$ python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_include.yaml

<!-- TEST -->
$ python main_base.py --config config/sign_cvpr_A_hands/AUTSL/test_joint_autsl.yaml
$ python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/test_joint_include.yaml
```