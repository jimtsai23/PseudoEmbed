# Pseudo-Embedding for Generalized Few-Shot 3D Segmentation
---

## Point projection
S3DIS
```
python project.py
```

ScanNet
```
python project_scannet.py
```


## Segmentation and fuse
S3DIS
```
python zopenseg_base.py \
        --output_dir z_openseg \
        --dataset s3dis \
        --split train \
        --exp base0 \
        --openseg_model  openseg_exported_clip
```

ScanNet
```
python zopenseg_base.py \
        --output_dir z_openseg \
        --dataset scannet \
        --split train \
        --exp pt_pcd_camz01 \
        --openseg_model  openseg_exported_clip
```


## Generate pseudo masks
```
python zeval.py --generate_pseudo=True
```


## Data preprocess
```
python pretrain/preprocess/room2blocks.py --pseudo --exp
```


## Train
```
exp=pt_pcd
k_shot=1
python train.py  --save_path log_scannet/S0_K${k_shot}/$exp \
    --k_shot $k_shot --epochs 150 --phase train  \
    --pc_augm  --dataset scannet --total_classes 21 --cvfold 0  \
    --data_path datasets/ScanNet/blocks_bs1_s1 \
    --testing_data_path datasets/ScanNet/blocks_bs1_s1_test \
    --use_pretrain_weight --pretrain_checkpoint_path pretrain/log_scannet/log_pretrain_scannet_S0 \
    --bg_data_path pt_pcd
```
