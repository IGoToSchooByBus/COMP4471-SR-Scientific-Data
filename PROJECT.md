# COMP4471 — Deep Learning for Super-Resolution of Scientific Simulation Data

- 團隊：LI Muzi (21435533)、YEUNG Ho Yin (21042453)
- 課程：HKUST COMP4471
- 摘要：用 U-Net（及 SRCNN / UNet2d 變體）把 PDEBench 淺水波（徑向壩潰）的低解析場（32×32）重建回高解析（128×128），並與 bicubic、1-NN/線性內插（oracle）、LR 最近鄰等基準比較。

---

## 快速開始

```bash
# 1. 裝依賴
pip install torch h5py scikit-image scikit-learn tqdm

# 2. 下載資料（3 通道 h/u/v、1000 樣本、21 幀）
#    方法 A：Google Drive 連結：<貼上你的分享連結>
#    方法 B：自己生成，見 docs/colab_regenerate.md

# 3. 訓練（點子範例）
python pdebench/train_unet_v2.py --data_path pdebench/data/2D_rdb_NA_NA_.h5 \
    --model miniunet --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 \
    --t_start 0 --t_end 20 --t_step 1 --out_dir outputs_miniunet
```

## 檔案結構

| 路徑 | 內容 |
|---|---|
| `pdebench/train_unet_v2.py` | **主訓練/評估腳本**（所有實驗的入口） |
| `pdebench/run_eval.py` | 對多個 checkpoint 跑 `--eval_only` 補算 baseline |
| `pdebench/run_experiments.py` | 一次跑 4 個模型對照 |
| `pdebench/data_gen/` | 修改過的資料生成（3 通道 + 第二參數 inner_height） |
| `docs/` | 專案文件（技術解釋、進度、Colab 流程） |
| `PROJECT.md` | 本文件（交接總覽） |

## 重現實驗

| 實驗 | 指引 |
|---|---|
| 模型對照（SRCNN/MiniUNet/UNet2d/UNet2d-wide） | `docs/colab_training.md` |
| 消融（噪聲 / 參數外推 / 時間外推 / 混合） | `docs/colab_ablation.md` |
| 資料生成（1000 樣本、3 通道） | `docs/colab_regenerate.md` |

## 關鍵結果摘要

### 模型對照（隨機切分，1000 樣本、3 通道、pointwise ×4）

| 方法 | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic | 37.13 | 0.950 |
| SRCNN | 51.03 | 0.996 |
| MiniUNet | **57.43** | 0.999 |
| UNet2d | 57.33 | 0.999 |
| UNet2d wide | 56.08 | 0.999 |
| 1-NN（oracle） | 57.54 | 0.998 |
| 線性內插（oracle） | **57.92** | 0.999 |

### 噪聲消融（PSNR dB）

| 方法 | σ=0 | σ=0.05 | σ=0.1 |
|---|---|---|---|
| Bicubic | 37.13 | 30.61 | 25.63 |
| MiniUNet | 57.43 | 45.34 | 40.91 |
| UNet2d | 57.33 | 47.34 | 43.35 |
| LR 最近鄰 | ~57.5 | **49.48** | **44.06** |
| 1-NN（oracle） | 57.54 | 57.54 | 57.54 |

## 主要結論

1. **U-Net 逼近但不超過「參數空間線性內插」oracle**（57.43 vs 57.92），顯示此平滑資料集上「內插」近乎最優。
2. **噪聲下，LR 最近鄰（非參數查表）勝過所有 U-Net**（σ=0.1：44.06 vs 43.52），顯示深度學習在此資料的抗噪優勢有限。
3. **噪聲下模型容量排序反轉**：σ=0 小模型（MiniUNet）最佳；σ>0 大模型（UNet2d/wide）反超（容量有助去噪）。
4. 這些發現共同指向：**此淺水波資料（2 參數、平滑）對深度學習而言過於簡單，需湍流等更高複雜度的場才能彰顯 SR 的價值**。

## 資料交接

- 原始 `.h5`（~4GB）**不上傳 GitHub**。用 Google Drive 分享連結，或照 `docs/colab_regenerate.md` 重生成。
- 分享連結：`<在此貼上 Drive 連結>`
