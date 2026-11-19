# Colab 消融實驗（一個實驗一個 cell）

斷線復原：重跑 Cell 1，再從中斷的下一格繼續（已完成結果在 Drive）。

---

## Cell 1：設定

```python
from google.colab import drive
drive.mount('/content/drive')
!cp '/content/drive/MyDrive/2D_rdb_NA_NA_1000.h5' /content/data.h5
!cp '/content/drive/MyDrive/train_unet_v2.py' /content/train_unet_v2.py
!pip install -q h5py scikit-image tqdm
```

## Cell 2：噪聲 0.05

```python
!python -u train_unet_v2.py --data_path /content/data.h5 --model srcnn --batch_size 128 --epochs 60 --patience 10 --noise_std 0.05 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise005_srcnn

!python -u train_unet_v2.py --data_path /content/data.h5 --model miniunet --base 32 --batch_size 128 --epochs 60 --patience 10 --noise_std 0.05 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise005_miniunet

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 32 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.05 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise005_unet2d

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 64 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.05 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise005_unet2d_w
```

## Cell 3：噪聲 0.1

```python
!python -u train_unet_v2.py --data_path /content/data.h5 --model srcnn --batch_size 128 --epochs 60 --patience 10 --noise_std 0.1 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise010_srcnn

!python -u train_unet_v2.py --data_path /content/data.h5 --model miniunet --base 32 --batch_size 128 --epochs 60 --patience 10 --noise_std 0.1 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise010_miniunet

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 32 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.1 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise010_unet2d

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 64 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.1 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise010_unet2d_w
```

## Cell 4：噪聲 0.2

```python
!python -u train_unet_v2.py --data_path /content/data.h5 --model srcnn --batch_size 128 --epochs 60 --patience 10 --noise_std 0.2 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise020_srcnn

!python -u train_unet_v2.py --data_path /content/data.h5 --model miniunet --base 32 --batch_size 128 --epochs 60 --patience 10 --noise_std 0.2 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise020_miniunet

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 32 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.2 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise020_unet2d

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 64 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.2 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_noise020_unet2d_w
```

## Cell 5：參數外推（極端，無噪聲）

```python
!python -u train_unet_v2.py --data_path /content/data.h5 --model srcnn --batch_size 128 --epochs 60 --patience 10 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_extrap_srcnn

!python -u train_unet_v2.py --data_path /content/data.h5 --model miniunet --base 32 --batch_size 128 --epochs 60 --patience 10 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_extrap_miniunet

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 32 --batch_size 64 --epochs 60 --patience 10 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_extrap_unet2d

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 64 --batch_size 64 --epochs 60 --patience 10 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_extrap_unet2d_w
```

## Cell 6：時間外推 head（前 25% 推後 75%，無噪聲）

```python
!python -u train_unet_v2.py --data_path /content/data.h5 --model srcnn --batch_size 128 --epochs 60 --patience 10 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temphead_srcnn

!python -u train_unet_v2.py --data_path /content/data.h5 --model miniunet --base 32 --batch_size 128 --epochs 60 --patience 10 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temphead_miniunet

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 32 --batch_size 64 --epochs 60 --patience 10 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temphead_unet2d

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 64 --batch_size 64 --epochs 60 --patience 10 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temphead_unet2d_w
```

## Cell 7：時間外推 tail（後 25% 推前 75%，無噪聲）

```python
!python -u train_unet_v2.py --data_path /content/data.h5 --model srcnn --batch_size 128 --epochs 60 --patience 10 --temporal_split tail --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temptail_srcnn

!python -u train_unet_v2.py --data_path /content/data.h5 --model miniunet --base 32 --batch_size 128 --epochs 60 --patience 10 --temporal_split tail --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temptail_miniunet

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 32 --batch_size 64 --epochs 60 --patience 10 --temporal_split tail --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temptail_unet2d

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 64 --batch_size 64 --epochs 60 --patience 10 --temporal_split tail --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_temptail_unet2d_w
```

## Cell 8：混合（噪聲 0.1 + 參數外推 + 時間外推）

```python
!python -u train_unet_v2.py --data_path /content/data.h5 --model srcnn --batch_size 128 --epochs 60 --patience 10 --noise_std 0.1 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_mixed_srcnn

!python -u train_unet_v2.py --data_path /content/data.h5 --model miniunet --base 32 --batch_size 128 --epochs 60 --patience 10 --noise_std 0.1 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_mixed_miniunet

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 32 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.1 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_mixed_unet2d

!python -u train_unet_v2.py --data_path /content/data.h5 --model unet2d --base 64 --batch_size 64 --epochs 60 --patience 10 --noise_std 0.1 --split_mode extrapolate --train_radius_max 0.40 --test_radius_min 0.60 --temporal_split head --temporal_train_frac 0.25 --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --t_start 0 --t_end 20 --t_step 1 --out_dir /content/drive/MyDrive/ab_mixed_unet2d_w
```

---

## 斷線復原

1. 重跑 **Cell 1**（掛 Drive + 複製）。
2. 從**中斷的下一格**繼續（已完成結果在 Drive，不用重跑）。

## 說明

- batch：小模型（srcnn/miniunet）128，大模型（unet2d）64。
- 參數外推缺口：train r≤0.40，test r≥0.60（r∈[0.4,0.6] 不用）。
- 時間外推：0.25 訓練幀 / 0.75 測試幀。
- 混合：三個因素疊加（噪聲 + 參數外推 + 時間外推）。
