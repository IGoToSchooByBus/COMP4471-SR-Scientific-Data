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
資料: https://drive.google.com/file/d/1LUJn9pi8lpiMdJGrBJryhy4XN2XQEV47/view?usp=drive_link
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
| `--amp` / `--no-amp` | 開 | CUDA 混合精度（tensor cores，快 1.5–2 倍）；`--no-amp` 回到純 FP32 |

### 執行

| 參數 | 預設 | 說明 |
|---|---|---|
| `--out_dir` | `outputs_unet_v2` | 輸出目錄（結果 + checkpoint） |
| `--resume` | None | 從 checkpoint 繼續訓練 |
| `--eval_only` | False | 不訓練，只讀 checkpoint 做評估（補算 baseline） |
| `--skip_baselines` | False | 訓練時不算 baseline（同實驗 4 個模型共用同一組 baseline，只對第一個算，最後用 `run_eval.py` 補一次） |
| `--ckpt` | None | 評估用 checkpoint（預設 `out_dir/model_best.pt`） |
| `--ckpt_lean` | False | checkpoint 不存 optimizer/scheduler 狀態（檔案小 ~3 倍、寫 Drive 快；resume 時優化器重建、lr 重設） |
| `--num_workers` | Win 0 / 其他 2 | DataLoader 子程序數（Windows spawn 記憶體貴所以預設 0） |
| `--deterministic` | 關 | 關閉 cudnn.benchmark、強制 deterministic（慢但逐位元可復現；預設關 = 快） |

### 常用命令

```bash
# 訓練（pointwise ×4 + L1 + 梯度損失，3 通道資料）
python train_unet_v2.py --data_path data/2D_rdb_NA_NA_.h5 --model miniunet \
    --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 \
    --t_start 0 --t_end 20 --t_step 1 --out_dir outputs_miniunet

# 只評估（不重訓，重算 bicubic / 1-NN / 線性內插 / LR 最近鄰 / U-Net）
python train_unet_v2.py --eval_only --data_path data/2D_rdb_NA_NA_.h5 --out_dir outputs_miniunet
```

## 最新結果摘要（最終版，pointwise 對齊修正後）

### 模型對照（3 seeds，mean ± std）

| 模型 | PSNR | pooled_PSNR |
|---|---|---|
| SRCNN | 52.00 ± 0.19 | 52.58 ± 0.08 |
| MiniUNet | 58.29 ± 0.23 | 54.99 ± 0.20 |
| UNet2d | 57.75 ± 0.37 | 55.37 ± 0.29 |
| UNet2d wide | 58.95 ± 1.37 | 55.38 ± 1.06 |

三種 U-Net 統計上相當（誤差棒重疊）；SRCNN 明顯較弱（−6 dB）。pooled 比 per-sample 低 ~3 dB = t=0 幀灌水。

### 參數外推（train r≤0.40 → test r≥0.60，最關鍵）

| 方法 | MSE | pooled PSNR |
|---|---|---|
| 1-NN / inv_dist（查表） | 0.0086 | 30.7 |
| 線性外推（global/local） | 0.013~0.014 | 28.5~28.9 |
| LR 最近鄰 | 0.0072 | 31.5 |
| **U-Net（MiniUNet）** | 9.48e-5 | **50.31** |
| **Classical pipeline（參數反演 + Clawpack）** | **8.97e-5** | **≈50.5** |

**U-Net 贏查表 baseline 24 dB，但和物理古典 pipeline 打平**——深度學習在此資料的外推上沒有超越物理方法。

### 時間外推（MiniUNet）

| 方向 | PSNR |
|---|---|
| head（早→晚） | 44.49 |
| tail（晚→早） | 47.65 |

早→晚更難（初始壩體結構獨特）。

### 噪聲消融（MiniUNet，σ=0.2 未跑）

| σ=0 | σ=0.05 | σ=0.1 |
|---|---|---|
| 58.29 ± 0.23 | 45.67 | 41.28 |

### 混合（噪聲 0.1 + 參數外推 + 時間外推）

MiniUNet：28.95 dB（最難，仍勝 bicubic 24.07）。

### 失敗模式（t=0 破壞物理）

U-Net 的 t=0 幀獨佔 **78% 的 MSE**（預測非零 u/v，違反初始靜止條件）；classical pipeline 的物理天生免疫——U-Net 最大的失敗模式，恰是物理方法零成本擁有的優勢。

**核心結論**：
1. **模型容量**：三種 U-Net 相當（~58 dB）、SRCNN 弱、大模型變異大。
2. **外推**：U-Net >> 查表 baseline（+24 dB），但 **≈ 物理 classical pipeline（打平 ~50.5 dB）**。
3. **時間**：早→晚更難。
4. **失敗模式**：t=0 佔 78% MSE，物理方法免疫。
5. **根本洞察**：此平滑資料對深度學習過於簡單——物理古典方法與深度學習打平；深度學習的價值需在更高複雜度的場（湍流）才能彰顯。詳見 [04-progress.md](./docs/04-progress.md)。
