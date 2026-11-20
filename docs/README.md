# COMP4471 Project 文件

**Deep Learning for Super-Resolution of Scientific Simulation Data**

- 團隊：LI Muzi (21435533)、YEUNG Ho Yin (21042453)
- 課程：COMP4471

## 交接總覽

👉 **先看 [`../PDEBench/PROJECT.md`](../PDEBench/PROJECT.md)**：一頁摘要（快速開始、關鍵結果、資料交接）。

## 文件索引

| 文件 | 內容 |
|---|---|
| [01 - Project 目標](./01-project-overview.md) | 問題定義、目標、資料、方法總覽 |
| [02 - 技術解釋](./02-technical-explanation.md) | 資料、流程、U-Net 架構、損失、優化器、指標 |
| [03 - 進一步方向](./03-future-directions.md) | 方向 A/B/C/D/E 與優先級 |
| [04 - 目前進度](./04-progress.md) | 已完成工作、模型對照 + 噪聲消融結果、結論 |
| [Colab 資料生成](./colab_regenerate.md) | 在 Colab 重新生成 1000 樣本、3 通道資料（重置版） |
| [Colab 訓練](./colab_training.md) | 付費 GPU 上一次跑完 4 個模型對照 |
| [Colab 消融](./colab_ablation.md) | 噪聲 / 參數外推 / 時間外推 / 混合（4 實驗 × 4 模型） |

## 關鍵檔案位置

| 檔案 | 路徑 |
|---|---|
| 主訓練/評估腳本 | `../PDEBench/pdebench/train_unet_v2.py` |
| 本地補算 baseline | `../PDEBench/pdebench/run_eval.py` |
| 一次跑 4 模型 | `../PDEBench/pdebench/run_experiments.py` |
| 交接總覽 | `../PDEBench/PROJECT.md` |
| 資料（3 通道、1000 樣本） | `../PDEBench/pdebench/data/2D_rdb_NA_NA_.h5` |
| 消融結果 | `../PDEBench/pdebench/ab_*/`、`outputs*/` |
| Proposal | `../COMP4471_Project_Proposal.pdf` |

## 參數說明（`train_unet_v2.py`）

### 資料

| 參數 | 預設 | 說明 |
|---|---|---|
| `--data_path` | `data/2D_rdb_NA_NA.h5` | HDF5 資料路徑 |
| `--t_start` | 0 | 起始時間步 |
| `--t_end` | 100 | 結束時間步（含）。21 幀資料用 `--t_end 20` |
| `--t_step` | 1 | 時間步間隔（>1 降低相鄰幀相關性與成本） |
| `--noise_std` | 0.0 | 加到 LR 的量測雜訊標準差 σ（0=不加） |
| `--data_range` | None | 固定 PSNR 的 R（跨實驗可比；預設從訓練集算） |

### 下採樣

| 參數 | 預設 | 說明 |
|---|---|---|
| `--factor` | 4 | 下採樣倍率（2 / 4 / 8） |
| `--downsample` | `block` | `block`=反鋸齒平均；`pointwise`=隔行取樣（無反鋸齒，凸顯深度學習） |

### 切分

| 參數 | 預設 | 說明 |
|---|---|---|
| `--train_frac` | 0.9 | 訓練用多少比例的唯一模擬 |
| `--split_mode` | `random` | `random` / `extrapolate`（小半徑訓練、大半徑測試）/ `middle`（兩端訓練、中間測試） |
| `--train_radius_max` | None | 半徑門檻 1：extrapolate=訓練 ≤ 此值；middle=低訓練帶上界 |
| `--test_radius_min` | None | 半徑門檻 2：extrapolate=測試 ≥ 此值；middle=高訓練帶下界 |
| `--temporal_split` | `none` | `none` / `head`（前幾幀訓練推後面）/ `tail`（後幾幀推前面） |
| `--temporal_train_frac` | 0.25 | 時間切分時訓練用的幀比例（0.25 = 前 25% 幀訓練、後 75% 測試） |
| `--val_frac` | 0.1 | 訓練集內再切出的驗證比例 |

### 模型

| 參數 | 預設 | 說明 |
|---|---|---|
| `--model` | `miniunet` | `miniunet`（2 層 U-Net）/ `unet2d`（4 層）/ `srcnn`（3 層 CNN） |
| `--base` | 32 | 網路首層通道數（寬度） |

### 訓練

| 參數 | 預設 | 說明 |
|---|---|---|
| `--batch_size` | 32 | 批次大小（小模型可用 128，大模型用 64） |
| `--epochs` | 100 | 最大 epoch |
| `--lr` | 1e-3 | 學習率 |
| `--weight_decay` | 1e-4 | Adam 權重衰減 |
| `--loss` | `mse` | `mse`(L2) / `l1` / `charbonnier` |
| `--grad_loss_weight` | 0.0 | 梯度 loss（Sobel）權重 λ（0=不用；0.5 可銳化衝擊波） |
| `--seed` | 0 | 隨機種子（可復現） |
| `--patience` | 15 | 早停：連續多少 epoch 無改善即停 |
| `--lr_patience` | 5 | ReduceLROnPlateau 的 patience |

### 執行

| 參數 | 預設 | 說明 |
|---|---|---|
| `--out_dir` | `outputs_unet_v2` | 輸出目錄（結果 + checkpoint） |
| `--resume` | None | 從 checkpoint 繼續訓練 |
| `--eval_only` | False | 不訓練，只讀 checkpoint 做評估（補算 baseline） |
| `--ckpt` | None | 評估用 checkpoint（預設 `out_dir/model_best.pt`） |

### 常用命令

```bash
# 訓練（pointwise ×4 + L1 + 梯度損失，3 通道資料）
python train_unet_v2.py --data_path data/2D_rdb_NA_NA_.h5 --model miniunet \
    --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 \
    --t_start 0 --t_end 20 --t_step 1 --out_dir outputs_miniunet

# 只評估（不重訓，重算 bicubic / 1-NN / 線性內插 / LR 最近鄰 / U-Net）
python train_unet_v2.py --eval_only --data_path data/2D_rdb_NA_NA_.h5 --out_dir outputs_miniunet
```

## 最新結果摘要

### 模型對照（隨機切分，1000 樣本、3 通道）

| 方法 | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic | 37.13 | 0.950 |
| SRCNN | 51.03 | 0.996 |
| **MiniUNet** | **57.43** | 0.999 |
| UNet2d | 57.33 | 0.999 |
| UNet2d wide | 56.08 | 0.999 |
| 1-NN（oracle） | 57.54 | 0.998 |
| 線性內插（oracle） | **57.92** | 0.999 |

**核心結論**：參數外推時 U-Net（50.87 dB）遠勝 oracle（26.15 dB）；噪聲有「交叉點」（低噪聲 LR 最近鄰勝、高噪聲 U-Net 反超）；時間外推晚→早更難。詳見 [04-progress.md](./04-progress.md)。
