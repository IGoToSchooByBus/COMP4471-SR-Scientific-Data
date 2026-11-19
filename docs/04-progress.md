# 04 · 目前進度

## 目前資料

`2D_rdb_NA_NA_.h5`（Colab 重新生成）：
- **1000 樣本、3 通道（h/u/v）、2 參數（dam_radius × inner_height）、21 幀**（每 5 個時間步存一幀）。
- 所有 1000 場唯一（2 參數不同），無重複場 → 無 test-train 洩漏。

## 已完成的工作

1. **主訓練/評估腳本** `train_unet_v2.py`，支援：
   - 多模型 `--model {srcnn, miniunet, unet2d}`、寬度 `--base`。
   - 下採樣 `--downsample {block, pointwise}`、損失 `--loss {mse,l1,charbonnier}`、梯度損失 `--grad_loss_weight`。
   - 多時間步 `--t_start/--t_end/--t_step`、量測雜訊 `--noise_std`。
   - 切分 `--split_mode {random, extrapolate, middle}`、**時間切分 `--temporal_split {head,tail}`**。
   - 固定 `--data_range`（跨實驗可比）、`--resume` / `--eval_only`。
2. **基準線（baseline）**：
   - bicubic。
   - **1-NN / 線性內插**（參數空間 oracle：用真值 (r,h) 查表/內插乾淨 HR）。
   - **LR 最近鄰**（非參數：找噪聲 LR 最近的訓練樣本，用其乾淨 HR）。
3. **零變異數處理**：t=0 的 u/v 精確零場 → PSNR 設 NaN 跳過（不再算無限 dB）。
4. **group-level 統計**：每場模擬平均（避免把相關幀當獨立樣本）。
5. **分通道 PSNR**：`PSNR_per_channel = [h, u, v]`。

## 模型對照（隨機切分，1000 樣本、3 通道、pointwise×4、L1+grad）

| 方法 | PSNR (dB) | SSIM | MaxError |
|---|---|---|---|
| Bicubic | 37.13 | 0.950 | 0.254 |
| SRCNN | 51.03 | 0.996 | 0.101 |
| **MiniUNet** | **57.43** | 0.999 | 0.057 |
| UNet2d | 57.33 | 0.999 | 0.059 |
| UNet2d wide | 56.08 | 0.999 | 0.069 |
| 1-NN（oracle） | 57.54 | 0.998 | — |
| **線性內插（oracle）** | **57.92** | 0.999 | — |

## 噪聲消融（PSNR, dB）

| 方法 | σ=0 | σ=0.05 | σ=0.1 |
|---|---|---|---|
| Bicubic | 37.13 | 30.61 | 25.63 |
| SRCNN | 51.03 | 40.36 | 36.01 |
| MiniUNet | 57.43 | 45.34 | 40.91 |
| UNet2d | 57.33 | 47.34 | 43.35 |
| UNet2d wide | 56.08 | 46.80 | 43.52 |
| **LR 最近鄰** | （未算*） | **49.48** | **44.06** |
| 1-NN（oracle） | 57.54 | 57.54 | 57.54 |
| 線性內插（oracle） | 57.92 | 57.92 | 57.92 |

## 主要結論

1. **U-Net 逼近但不超過「參數空間線性內插」oracle**（57.43 vs 57.92）：此平滑資料上「內插」近乎最優，SR 給深度學習的發揮空間有限。
2. **噪聲下 LR 最近鄰勝過所有 U-Net**（σ=0.1：44.06 vs 43.52）：非參數查表先天去噪（輸出必為乾淨 HR），抗噪優於 U-Net。
3. **噪聲下模型容量排序反轉**：σ=0 小模型（MiniUNet）最佳（大模型過擬合）；σ>0 大模型（UNet2d/wide）反超（容量有助去噪）。
4. 共同指向：**此淺水波資料（2 參數、平滑）對深度學習過於簡單，需湍流等更高複雜度的場才能彰顯 SR 價值**。

\* σ=0 的 LR_1NN 需對 `outputs_*` 跑 `--eval_only` 補算（`run_eval.py` 目前只含 σ=0.05 / 0.1 的 checkpoint）。

## 待完成的實驗（命令已備好，見 `colab_ablation.md`）

1. **參數外推**（train r≤0.40 / test r≥0.60，極端缺口）。
2. **時間外推**（head：前 25% 幀推後 75%；tail：反過來）。
3. **混合**（噪聲 0.1 + 參數外推 + 時間外推）。
4. σ=0.2 噪聲點 + σ=0 的 LR_1NN（補完整曲線）。

## 下一步

1. 跑完上述外推實驗。
2. 畫「噪聲 vs PSNR」曲線、error map、長條圖。
3. 彙整最終報告（對照表 + 分通道指標 + 結論）。
