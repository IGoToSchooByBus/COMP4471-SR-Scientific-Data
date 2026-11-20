"""
PDEBench 2D Shallow Water: 超解析度 U-Net（方向 A + B + D）

方向 A（工程化）：
  1. 可復現性      —— 統一 seed（torch / numpy / cuda）
  2. 真殘差學習    —— pred = bicubic(lr) + model(bicubic(lr))
  3. 資料歸一化    —— 用訓練集 mean/std 做標準化，評估時再 denormalize
  4. validation    —— 900 筆再切 train/val，用 val loss 做早停
  5. checkpoint    —— 存最佳模型（含 optimizer / scheduler / 統計量），支援 --resume / --eval_only
  6. 結果輸出      —— 與 bicubic_baseline_revised.py 相同格式的 JSON / CSV / 圖

方向 B（讓任務更有鑑別力）：
  --downsample {block, pointwise}  下採樣方式（pointwise 無反鋸齒 → aliasing，難題）
  --factor {2,4,8}                  下採樣倍率（越大越難）
  --loss {mse, l1, charbonnier}     主損失（L1 保留邊緣較銳）
  --grad_loss_weight <λ>            梯度 loss 權重（Sobel 邊緣一致性，銳化 shock front）

方向 D（資料多樣性）：
  --t_start/--t_end/--t_step   使用多個時間步（取代單一 t=50）
  --noise_std <σ>              加到 LR 的量測雜訊（模擬感測器誤差）

資料切分：一律按「唯一模擬」切分（無 test-train 重疊）；--split_mode extrapolate 可做外推測試。

執行範例：
  # 方向 B 實驗：pointwise ×4 + L1 + gradient loss（隨機切唯一模擬）
  python train_unet_v2.py --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --out_dir outputs_b
  # 外推測試：小的壩半徑訓練、大的壩半徑測試
  python train_unet_v2.py --downsample pointwise --factor 4 --loss l1 --grad_loss_weight 0.5 --split_mode extrapolate --out_dir outputs_extrap
  # 只評估（不訓練，直接讀 out_dir/model_best.pt）
  python train_unet_v2.py --eval_only --out_dir outputs_b
"""
import argparse
import csv
import json
import os
import re

# 避免 Windows 上 torch 內建 OpenMP 與 numpy/MKL 的 OpenMP 各自載入 libiomp5md.dll，
# 造成 "OMP: Error #15 ... libiomp5md.dll already initialized"。必須在 import torch/numpy/scipy 之前設定。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# 1. 可復現性
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# 2. 資料載入與下採樣
# --------------------------------------------------------------------------- #
def load_swe_representative(path: str, t_index: int = 0):
    """載入「單一時間步」的所有模擬場（用於去重），回傳 ([N_sims, H, W, C], sorted keys)。

    方向 D 多通道：通道維自動偵測（舊資料 C=1 只有 h，新資料 C=3 為 h/u/v）。
    """
    with h5py.File(path, "r") as f:
        keys = sorted(list(f.keys()))
        arrs = [np.asarray(f[k]["data"][t_index], dtype=np.float32) for k in keys]  # [H, W, C]
    return np.stack(arrs, axis=0), keys


def load_swe_timeseries(path: str, sim_indices, t_start: int = 0, t_end: int = 100, t_step: int = 1):
    """只載入「指定模擬」的完整時間序列（省記憶體），回傳 [len(sim_indices), T_sel, H, W]。

    方向 D：1000 場都唯一（2 參數 dam_radius × inner_height）；每場 21 幀。
    """
    t_idx = list(range(t_start, t_end + 1, t_step))
    with h5py.File(path, "r") as f:
        keys = sorted(list(f.keys()))
        arrs = [np.asarray(f[keys[i]]["data"][t_idx], dtype=np.float32) for i in sim_indices]  # [T, H, W, C]
    return np.stack(arrs, axis=0)


def unique_group_ids(hr_all):
    """把「相同模擬」的樣本歸到同一組（用整張場的 bytes 當 key）。

    返回 (每筆樣本的 group id, 唯一場數)。radial dam break 的 dam_radius 在 128×128
    網格上被離散化，1000 筆其實只有 ~81 種不同的場；用場本身分組可自動偵測重複。
    """
    seen = {}
    gids = np.empty(len(hr_all), dtype=np.int64)
    for i, hr in enumerate(hr_all):
        key = hr.tobytes()
        if key not in seen:
            seen[key] = len(seen)
        gids[i] = seen[key]
    return gids, len(seen)


def load_dam_radius(path: str):
    """從每筆樣本的 HDF5 attrs['config']（YAML）讀出 dam_radius，回傳與 sorted keys 對齊的陣列。"""
    radii = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            cfg = f[key].attrs.get("config", "")
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8", errors="ignore")
            m = re.search(r"dam_radius:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cfg)
            radii.append(float(m.group(1)) if m else np.nan)
    return np.array(radii, dtype=np.float64)


def load_sim_params(path: str):
    """讀出每筆模擬的 (dam_radius, inner_height)，回傳 [N, 2]（供 1-NN / 線性內插 baseline 用）。"""
    params = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            cfg = f[key].attrs.get("config", "")
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8", errors="ignore")
            r = re.search(r"dam_radius:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cfg)
            h = re.search(r"inner_height:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cfg)
            params.append([float(r.group(1)) if r else np.nan, float(h.group(1)) if h else np.nan])
    return np.array(params, dtype=np.float64)


def block_average_downsample(hr: np.ndarray, factor: int = 4):
    """[H, W, (C)] -> [H//factor, W//factor, (C)]（反鋸齒的平均池化下採樣）。"""
    H, W = hr.shape[0], hr.shape[1]
    cropped = hr[: H - H % factor, : W - W % factor]
    return cropped.reshape(H // factor, factor, W // factor, factor, *cropped.shape[2:]).mean(axis=(1, 3))


def pointwise(hr: np.ndarray, factor: int = 4):
    """pointwise 隔行取樣：每 factor 個像素只取 1 個，無反鋸齒 → 產生 aliasing（方向 B 的難題）。
    取樣偏移 factor//2：對齊 bicubic(align_corners=False) 把 LR pixel i 放在 HR 座標 f·i + f/2 − 0.5，
    把全域位移從 f/2 − 0.5 降到 0.5 px（factor=4：1.5 px → 0.5 px）。
    （對齊測試：hr[::4] 的 bicubic 被位移懲罰 3.5 dB；hr[2::4] 回升 +3.52 dB。）"""
    off = factor // 2
    return hr[off::factor, off::factor].copy()


def get_downsample_fn(name: str):
    if name == "block":
        return block_average_downsample
    if name == "pointwise" or name == "subsample":  # 「subsample」保留為舊名相容
        return pointwise
    raise ValueError(f"未知的 downsampling 方法：{name}（可選 block / pointwise）")


class SWEDataset(Dataset):
    """回傳歸一化後的 (LR, HR) pair；歸一化統計量用訓練集計算。"""

    def __init__(self, hr_data, factor=4, mean=0.0, std=1.0, downsample="block", noise_std=0.0, seed=0):
        self.hr = hr_data
        ds_fn = get_downsample_fn(downsample)
        self.lr = np.stack([ds_fn(h, factor) for h in hr_data])
        if noise_std > 0:  # 方向 D：加量測雜訊（模擬感測器誤差），固定 seed 以可復現
            rng = np.random.default_rng(seed)
            self.lr = self.lr + rng.normal(0.0, noise_std, self.lr.shape).astype(np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):
        hr = self.hr[idx]   # [H, W, C]
        lr = self.lr[idx]   # [H_lr, W_lr, C]
        hr_n = (hr - self.mean) / self.std
        lr_n = (lr - self.mean) / self.std
        # channel-last [H,W,C] -> channel-first [C,H,W]
        return (
            torch.from_numpy(np.ascontiguousarray(lr_n)).permute(2, 0, 1).float(),
            torch.from_numpy(np.ascontiguousarray(hr_n)).permute(2, 0, 1).float(),
        )


# --------------------------------------------------------------------------- #
# 3. 模型（架構與 train_unet.py 相同，只是讓它只輸出「殘差」）
# --------------------------------------------------------------------------- #
class MiniUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d2 = self.dec2(torch.cat([self.up2(e2), e1], dim=1))
        return self.out(d2)


class UNet2d(nn.Module):
    """4 層 U-Net（比 MiniUNet 更深，可搭配 --base 64 更寬）。"""

    def __init__(self, in_ch=3, out_ch=3, base=32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_ch, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.enc4 = block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec4 = block(base * 8, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec3 = block(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec2 = block(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d4 = self.dec4(torch.cat([self.up4(e4), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        return self.out(d2)


class SRCNN(nn.Module):
    """3 層 CNN（最原始的 SR 網路，9-1-5 kernels），輸出殘差。"""

    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(base, base // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, out_ch, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x)


def make_model(name, in_ch, out_ch, base=32):
    if name == "miniunet":
        return MiniUNet(in_ch=in_ch, out_ch=out_ch, base=base)
    if name == "unet2d":
        return UNet2d(in_ch=in_ch, out_ch=out_ch, base=base)
    if name == "srcnn":
        return SRCNN(in_ch=in_ch, out_ch=out_ch, base=base)
    raise ValueError(f"未知 model：{name}（可選 miniunet / unet2d / srcnn）")


def upsample_and_predict(model, lr_img, target_size):
    """真殘差：pred = bicubic(lr) + model(bicubic(lr))。"""
    lr_up = F.interpolate(lr_img, size=target_size, mode="bicubic", align_corners=False)
    return lr_up + model(lr_up)


def bicubic_upsample_raw(lr: np.ndarray, target_shape):
    """raw bicubic 上採樣（供 baseline 對比）。支援 [H,W] 與 [H,W,C]。"""
    lr = np.ascontiguousarray(lr, dtype=np.float32)
    if lr.ndim == 2:
        x = torch.from_numpy(lr)[None, None]  # [1,1,H,W]
        with torch.no_grad():
            y = F.interpolate(x, size=target_shape, mode="bicubic", align_corners=False)
        return y[0, 0].numpy()  # [H,W]
    x = torch.from_numpy(lr).permute(2, 0, 1)[None]  # [1,C,H,W]
    with torch.no_grad():
        y = F.interpolate(x, size=target_shape, mode="bicubic", align_corners=False)
    return y[0].permute(1, 2, 0).numpy()  # [H,W,C]


# --------------------------------------------------------------------------- #
# 4. 損失函式（方向 B）
# --------------------------------------------------------------------------- #
def sobel(x):
    """Sobel 邊緣算子：輸出每個像素的梯度幅值（邊緣圖）。支援多通道（逐通道算）。"""
    C = x.shape[1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def make_criterion(loss_name: str):
    """主損失：mse(L2) / l1(MAE) / charbonnier（L1 的平滑版）。"""
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "charbonnier":
        return lambda pred, target: torch.sqrt((pred - target) ** 2 + 1e-6).mean()
    raise ValueError(f"未知 loss：{loss_name}（可選 mse / l1 / charbonnier）")


def gradient_loss(pred, target):
    """歸一化的 gradient loss（Sobel 邊緣一致性），數值 ~O(1)，與像素損失尺度可比較。"""
    gp = sobel(pred)
    gt = sobel(target)
    norm = (gt ** 2).mean().detach() + 1e-8
    return F.mse_loss(gp, gt) / norm


def compute_total_loss(pred, target, criterion, grad_weight):
    """主損失 + λ * gradient loss。"""
    loss = criterion(pred, target)
    if grad_weight > 0:
        loss = loss + grad_weight * gradient_loss(pred, target)
    return loss


def compute_metrics(gt, pred, data_range):
    """與 bicubic_baseline_revised.py 相同設定，並額外補 MAE / MaxError（邊緣指標）。"""
    err = np.abs(gt - pred)
    mse = float(mean_squared_error(gt, pred))
    if float(gt.max() - gt.min()) < 1e-12:
        # 常數場（如 t=0 的 u/v = 0）→ PSNR 無定義，設 NaN（聚合時跳過，不高估）
        psnr = float("nan")
    else:
        psnr = float(peak_signal_noise_ratio(gt, pred, data_range=data_range))
    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": float(
            structural_similarity(
                gt, pred, data_range=data_range,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
            )
        ),
        "MAE": float(err.mean()),
        "MaxError": float(err.max()),
    }


def compute_metrics_multichannel(gt, pred, data_range):
    """gt, pred: [C, H, W]（channel-first）。data_range: 標量或 [C]。回傳「跨通道平均」的指標。"""
    C = gt.shape[0]
    dr = np.asarray(data_range, dtype=np.float64).reshape(-1)
    if dr.size == 1:
        dr = np.full(C, dr[0])
    ms = {"MSE": [], "PSNR": [], "SSIM": [], "MAE": [], "MaxError": []}
    for c in range(C):
        m = compute_metrics(gt[c], pred[c], float(dr[c]))
        for k in ms:
            ms[k].append(m[k])
    out = {k: float(np.nanmean(v)) for k, v in ms.items()}   # nanmean：跳過零變異數場的 NaN PSNR
    out["PSNR_per_channel"] = [float(v) if np.isfinite(v) else float("nan") for v in ms["PSNR"]]
    return out


def compute_parameter_baselines(train_hr, train_params, test_hr, test_params, data_range):
    """1-NN 與線性內插 baseline（oracle：用參數空間 + 訓練場的 HR，逐幀算指標）。

    train_hr: [n_train, T, H, W, C]; train_params: [n_train, 2]（dam_radius, inner_height）
    test_hr: [n_test, T, H, W, C]; test_params: [n_test, 2]
    回傳 {'1NN': {...}, 'linear': {...}}，值是跨通道平均的 MSE/PSNR/SSIM。
    """
    pmin = train_params.min(axis=0)
    pmax = train_params.max(axis=0)
    tr_n = (train_params - pmin) / (pmax - pmin + 1e-12)
    te_n = (test_params - pmin) / (pmax - pmin + 1e-12)
    T = train_hr.shape[1]
    agg = {"1NN": {"MSE": [], "PSNR": [], "SSIM": []}, "inv_dist": {"MSE": [], "PSNR": [], "SSIM": []}}
    for i in range(len(test_params)):
        d = np.linalg.norm(tr_n - te_n[i], axis=1)
        order = np.argsort(d)
        j1, j2 = int(order[0]), int(order[1])
        w1 = 1.0 / (d[j1] + 1e-12); w2 = 1.0 / (d[j2] + 1e-12)
        w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
        for t in range(T):
            gt = test_hr[i, t].transpose(2, 0, 1)
            nn = train_hr[j1, t].transpose(2, 0, 1)
            lin = (w1 * train_hr[j1, t] + w2 * train_hr[j2, t]).transpose(2, 0, 1)
            mn = compute_metrics_multichannel(gt, nn, data_range)
            ml = compute_metrics_multichannel(gt, lin, data_range)
            for k in ("MSE", "PSNR", "SSIM"):
                agg["1NN"][k].append(mn[k]); agg["inv_dist"][k].append(ml[k])
    return {name: {k: float(np.nanmean(v)) for k, v in agg[name].items()} for name in ("1NN", "inv_dist")}


def compute_linear_extrapolation_baseline(train_hr, train_params, test_hr, test_params, data_range):
    """線性外推 baseline：對每個時間步、每個像素獨立擬合「場 ≈ w0 + w1·dam_radius + w2·inner_height」，
    再外推到測試參數。和凸組合（1-NN/線性內插）不同，它可以「超出」訓練參數範圍（真外推）。

    train_hr: [n_train, T, H, W, C]; train_params: [n_train, 2]
    test_hr: [n_test, T, H, W, C]; test_params: [n_test, 2]
    回傳 {'global_lin_extrap': {...}}（MSE/PSNR/SSIM）。
    """
    n_train = len(train_params)
    T = train_hr.shape[1]
    shape = train_hr.shape[2:]   # (H, W, C)
    D = int(np.prod(shape))

    # 設計矩陣（截距 + 兩參數）
    X = np.column_stack([np.ones(n_train), train_params[:, 0], train_params[:, 1]])   # [n_train, 3]
    XtX_inv_Xt = np.linalg.solve(X.T @ X + 1e-10 * np.eye(3), X.T)   # [3, n_train]，小正則避免奇異

    lin_mse, lin_psnr, lin_ssim = [], [], []
    for t in range(T):
        Y = train_hr[:, t].reshape(n_train, D).astype(np.float64)   # [n_train, D]
        W = XtX_inv_Xt @ Y   # [3, D]（每像素最小平方解）
        for i in range(len(test_params)):
            x = np.array([1.0, test_params[i, 0], test_params[i, 1]], dtype=np.float64)
            pred = (x @ W).reshape(shape)   # [H, W, C]
            gt = test_hr[i, t]
            m = compute_metrics_multichannel(gt.transpose(2, 0, 1), pred.transpose(2, 0, 1), data_range)
            lin_mse.append(m["MSE"]); lin_psnr.append(m["PSNR"]); lin_ssim.append(m["SSIM"])
    return {"global_lin_extrap": {
        "MSE": float(np.nanmean(lin_mse)),
        "PSNR": float(np.nanmean(lin_psnr)),
        "SSIM": float(np.nanmean(lin_ssim)),
    }}


def compute_local_linear_extrapolation_baseline(train_hr, train_params, test_hr, test_params, data_range, k=32):
    """局部線性外推 baseline：用參數接近測試點的訓練場擬合線性模型再外推。

    - 測試點在訓練範圍「內」→ 用 k 個最近鄰（leverage 低，安全）。
    - 測試點在訓練範圍「外」（extrap）→ 改用 r 最上面的 band（r ≥ r_max − 0.05），
      避免 kNN 全擠在邊界導致 leverage 爆炸（reviewer 實測：kNN k=32 的 leverage = 311，
      band r≥r_max−0.05 的 leverage ≈ 2.8）。
    - fit 用歸一化參數 + ridge（λ=1e-2），並在結束時印預測範圍檢查數值爆炸。
    """
    pmin = train_params.min(axis=0)
    pmax = train_params.max(axis=0)
    tr_n = (train_params - pmin) / (pmax - pmin + 1e-12)
    te_n = (test_params - pmin) / (pmax - pmin + 1e-12)
    r_max = float(train_params[:, 0].max())
    T = train_hr.shape[1]
    shape = train_hr.shape[2:]
    D = int(np.prod(shape))
    k = min(k, len(train_params))

    # 外推用的 band：所有 test sim 共用同一個 sel → fit 只算 T 次（不重複 N_test 次）
    band_sel = np.where(train_params[:, 0] >= r_max - 0.05)[0]
    X_band = np.column_stack([np.ones(len(band_sel)), tr_n[band_sel, 0], tr_n[band_sel, 1]])
    XtX_inv_Xt_band = np.linalg.solve(X_band.T @ X_band + 1e-2 * np.eye(3), X_band.T)
    W_band = [XtX_inv_Xt_band @ train_hr[band_sel, t].reshape(len(band_sel), D).astype(np.float64)
              for t in range(T)]   # 每個 t 一個 [3, D]

    mse_list, psnr_list, ssim_list = [], [], []
    pred_min_all, pred_max_all = np.inf, -np.inf
    for i in range(len(test_params)):
        x = np.array([1.0, te_n[i, 0], te_n[i, 1]], dtype=np.float64)
        if test_params[i, 0] > r_max:   # 外推：用預先 fit 好的 band W
            for t in range(T):
                pred = (x @ W_band[t]).reshape(shape)
                pred_min_all = min(pred_min_all, float(pred.min()))
                pred_max_all = max(pred_max_all, float(pred.max()))
                m = compute_metrics_multichannel(test_hr[i, t].transpose(2, 0, 1), pred.transpose(2, 0, 1), data_range)
                mse_list.append(m["MSE"]); psnr_list.append(m["PSNR"]); ssim_list.append(m["SSIM"])
        else:                            # 內插：kNN（每個 test sim 的 sel 不同，只能在迴圈內 fit）
            d = np.linalg.norm(tr_n - te_n[i], axis=1)
            sel = np.argsort(d)[:k]
            X = np.column_stack([np.ones(len(sel)), tr_n[sel, 0], tr_n[sel, 1]])
            XtX_inv_Xt = np.linalg.solve(X.T @ X + 1e-2 * np.eye(3), X.T)
            for t in range(T):
                Y = train_hr[sel, t].reshape(len(sel), D).astype(np.float64)
                W = XtX_inv_Xt @ Y
                pred = (x @ W).reshape(shape)
                pred_min_all = min(pred_min_all, float(pred.min()))
                pred_max_all = max(pred_max_all, float(pred.max()))
                m = compute_metrics_multichannel(test_hr[i, t].transpose(2, 0, 1), pred.transpose(2, 0, 1), data_range)
                mse_list.append(m["MSE"]); psnr_list.append(m["PSNR"]); ssim_list.append(m["SSIM"])
    print(f"[local_lin_extrap] pred range: [{pred_min_all:.3f}, {pred_max_all:.3f}]（真值應約在 [-1, 2.5]；超出過多 = 數值爆炸）")
    return {"local_lin_extrap": {
        "MSE": float(np.nanmean(mse_list)),
        "PSNR": float(np.nanmean(psnr_list)),
        "SSIM": float(np.nanmean(ssim_list)),
    }}


def compute_lr_nn_baseline(train_lr, train_hr, test_lr, test_hr, data_range):
    """LR 最近鄰 baseline（非參數）：找「噪聲 LR」最接近的訓練樣本，用其乾淨 HR 當預測。
    此 baseline 和 U-Net 一樣看得到 LR（含噪聲），所以會受噪聲影響，適合噪聲消融的公平對照。
    """
    tr = train_lr.reshape(len(train_lr), -1).astype(np.float32)
    te = test_lr.reshape(len(test_lr), -1).astype(np.float32)
    tr_norm = np.sum(tr * tr, axis=1)   # [n_train]
    nn_mse, nn_psnr, nn_ssim = [], [], []
    chunk = 128
    for i0 in range(0, len(test_lr), chunk):
        i1 = min(i0 + chunk, len(test_lr))
        d2 = (np.sum(te[i0:i1] * te[i0:i1], axis=1, keepdims=True)
              + tr_norm[None, :]
              - 2.0 * te[i0:i1] @ tr.T)   # [chunk, n_train]
        nn_idx = np.argmin(d2, axis=1)
        for k in range(i1 - i0):
            j = int(nn_idx[k])
            gt = test_hr[i0 + k].transpose(2, 0, 1)
            pred = train_hr[j].transpose(2, 0, 1)
            m = compute_metrics_multichannel(gt, pred, data_range)
            nn_mse.append(m["MSE"]); nn_psnr.append(m["PSNR"]); nn_ssim.append(m["SSIM"])
    return {"LR_1NN": {
        "MSE": float(np.nanmean(nn_mse)),
        "PSNR": float(np.nanmean(nn_psnr)),
        "SSIM": float(np.nanmean(nn_ssim)),
    }}


# --------------------------------------------------------------------------- #
# 5. 視覺化：GT / LR / Bicubic / 模型預測 / 誤差圖
# --------------------------------------------------------------------------- #
def save_comparison(out_path, hr, lr, bicubic, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = float(hr.min()), float(hr.max())
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
    panels = [
        ("Ground truth", hr, "bilinear"),
        ("LR (32x32)", lr, "nearest"),
        ("Bicubic", bicubic, "nearest"),
        ("model", pred, "nearest"),
    ]
    for ax, (title, data, interp) in zip(axs[:4], panels):
        im = ax.imshow(data, cmap="viridis", interpolation=interp, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axs[:4], location="bottom", fraction=0.06, shrink=0.7)

    err = np.abs(hr - pred)
    err_im = axs[4].imshow(err, cmap="magma", interpolation="nearest")
    axs[4].set_title("|error| (U-Net)", fontsize=9)
    axs[4].axis("off")
    fig.colorbar(err_im, ax=axs[4], location="bottom", fraction=0.06, shrink=0.85)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. 評估 + 輸出（訓練後與 --eval_only 共用）
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    """把 NaN / inf 換成 None（嚴格 JSON 不允許 NaN/inf）。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def evaluate_and_save(
    model, test_loader, test_raw, mean, std, data_range, factor, args, n_val, device,
    train_min, train_max, baselines=None,
):
    model.eval()
    ds_fn = get_downsample_fn(args.downsample)
    mse_list, psnr_list, ssim_list, mae_list, maxerr_list = [], [], [], [], []
    psnr_per_ch_list = []
    pred_raw_first = hr_raw_first = lr_raw_first = bicubic_first = None

    mean_t = torch.as_tensor(np.asarray(mean).reshape(-1), dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t = torch.as_tensor(np.asarray(std).reshape(-1), dtype=torch.float32, device=device).view(1, -1, 1, 1)

    with torch.no_grad():
        for lr_img, hr_img in test_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            pred_n = upsample_and_predict(model, lr_img, hr_img.shape[-2:])
            pred_raw = pred_n * std_t + mean_t          # [N, C, H, W]
            hr_raw = hr_img * std_t + mean_t
            lr_raw = lr_img * std_t + mean_t

            for j in range(pred_raw.size(0)):
                m = compute_metrics_multichannel(hr_raw[j].cpu().numpy(), pred_raw[j].cpu().numpy(), data_range)
                mse_list.append(m["MSE"]); psnr_list.append(m["PSNR"]); ssim_list.append(m["SSIM"])
                mae_list.append(m["MAE"]); maxerr_list.append(m["MaxError"])
                psnr_per_ch_list.append(m["PSNR_per_channel"])
            if pred_raw_first is None:
                pred_raw_first = pred_raw[0, 0].cpu().numpy()   # 只取 h 通道做視覺化
                hr_raw_first = hr_raw[0, 0].cpu().numpy()
                lr_raw_first = lr_raw[0, 0].cpu().numpy()

    pooled_r = float(train_max - train_min)   # pooled signal range（全資料的 max-min，不是各通道 range 的平均）
    n_groups_test = int(getattr(args, "n_test_groups", 0) or 0)
    T_per = (len(mse_list) // n_groups_test) if (n_groups_test > 0 and len(mse_list) % n_groups_test == 0) else 0
    model_summary = {
        "method": f"bicubic + {args.model} residual ({args.loss} loss, grad_weight={args.grad_loss_weight})",
        "model": args.model,
        "downsample": args.downsample,
        "n_test": len(mse_list),
        "factor": factor,
        "data_range": np.asarray(data_range).tolist(),
        "MSE": float(np.mean(mse_list)),   # pooled MSE（所有像素匯總，含 t=0 誤差）
        "PSNR": float(np.mean(psnr_list)),   # 逐樣本 PSNR 平均（t=0 零變異數跳過）
        "pooled_PSNR": float(10.0 * np.log10(pooled_r ** 2 / float(np.mean(mse_list)))) if float(np.mean(mse_list)) > 0 else float("inf"),   # pooled PSNR（誠實，含 t=0）
        "PSNR_per_channel": [float(x) for x in np.nanmean(psnr_per_ch_list, axis=0)] if psnr_per_ch_list else [],
        "MSE_by_time": [float(x) for x in np.array(mse_list).reshape(n_groups_test, T_per).mean(axis=0)] if T_per > 1 else [],   # 逐幀 MSE（暴露 t=0 失敗）
        "SSIM": float(np.mean(ssim_list)),
        "MAE": float(np.mean(mae_list)),
        "MaxError": float(np.mean(maxerr_list)),
    }

    # 同測試集上的 bicubic baseline（用「模型看到的同一批含雜訊 LR」，保證公平）
    noisy_lr_all = test_loader.dataset.lr   # [n_test, H_lr, W_lr, C]，已含雜訊
    bic_mse, bic_psnr, bic_ssim, bic_mae, bic_maxerr = [], [], [], [], []
    bic_psnr_per_ch_list = []
    for hr, lr in zip(test_raw, noisy_lr_all):
        bic = bicubic_upsample_raw(lr, hr.shape[:2])   # [H, W, C]
        m = compute_metrics_multichannel(hr.transpose(2, 0, 1), bic.transpose(2, 0, 1), data_range)
        bic_mse.append(m["MSE"]); bic_psnr.append(m["PSNR"]); bic_ssim.append(m["SSIM"])
        bic_mae.append(m["MAE"]); bic_maxerr.append(m["MaxError"])
        bic_psnr_per_ch_list.append(m["PSNR_per_channel"])
    bicubic_summary = {
        "method": f"{args.downsample} down + bicubic (align_corners=False)",
        "downsample": args.downsample,
        "n_test": len(test_raw),
        "factor": factor,
        "data_range": np.asarray(data_range).tolist(),
        "MSE": float(np.mean(bic_mse)),
        "PSNR": float(np.mean(bic_psnr)),
        "pooled_PSNR": float(10.0 * np.log10(pooled_r ** 2 / float(np.mean(bic_mse)))) if float(np.mean(bic_mse)) > 0 else float("inf"),
        "PSNR_per_channel": [float(x) for x in np.nanmean(bic_psnr_per_ch_list, axis=0)] if bic_psnr_per_ch_list else [],
        "SSIM": float(np.mean(bic_ssim)),
        "MAE": float(np.mean(bic_mae)),
        "MaxError": float(np.mean(bic_maxerr)),
        "MSE_by_time": [float(x) for x in np.array(bic_mse).reshape(n_groups_test, T_per).mean(axis=0)] if T_per > 1 else [],
    }
    if bicubic_first is None and hr_raw_first is not None:
        bicubic_first = bicubic_upsample_raw(noisy_lr_all[0][..., 0], hr_raw_first.shape)

    # ---- group-level 統計（每場模擬平均，避免把相關幀當獨立樣本） ----
    n_groups = int(getattr(args, "n_test_groups", 0) or 0)
    T_sel = (len(mse_list) // n_groups) if (n_groups > 0 and len(mse_list) % n_groups == 0) else 0
    group_summary = {}
    if T_sel > 1:
        psnr_arr = np.array(psnr_list, dtype=np.float64).reshape(n_groups, T_sel)
        per_sim_psnr = np.nanmean(psnr_arr, axis=1)
        group_summary = {
            "n_test_groups": n_groups,
            "n_frames": len(mse_list),
            "PSNR_mean": float(np.mean(per_sim_psnr)),
            "PSNR_std": float(np.std(per_sim_psnr)),
        }

    # ---- 輸出檔 ----
    train_stats = {
        "mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist(), "data_range": np.asarray(data_range).tolist(),
        "train_min": float(train_min), "train_max": float(train_max),
        "test_samples": int(len(test_raw)), "val_size": int(n_val),
        "t_start": int(args.t_start), "t_end": int(args.t_end), "t_step": int(args.t_step),
        "noise_std": float(args.noise_std), "factor": int(factor), "seed": int(args.seed),
        "downsample": args.downsample, "loss": args.loss, "grad_loss_weight": args.grad_loss_weight,
        "split_mode": args.split_mode,
        "n_unique": int(getattr(args, "n_unique", 0)),
        "n_train_groups": int(getattr(args, "n_train_groups", 0)),
        "n_test_groups": int(getattr(args, "n_test_groups", 0)),
        "n_channels": int(getattr(args, "n_channels", 1)),
    }
    with open(args.out_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(train_stats), f, indent=2, ensure_ascii=False)
    with open(args.out_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(model_summary), f, indent=2)
    with open(args.out_dir / "bicubic_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(bicubic_summary), f, indent=2)
    if baselines:
        for _name, _v in baselines.items():
            if _v.get("MSE") is not None and _v["MSE"] > 0:
                _v["pooled_PSNR"] = float(10.0 * np.log10(pooled_r ** 2 / _v["MSE"]))
    with open(args.out_dir / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe({"baselines": baselines, "group_summary": group_summary}), f, indent=2)
    with open(args.out_dir / "model_per_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "MSE", "PSNR", "SSIM", "MAE", "MaxError"])
        for i, (a, b, c, d, e) in enumerate(zip(mse_list, psnr_list, ssim_list, mae_list, maxerr_list)):
            w.writerow([i, a, b, c, d, e])
    if pred_raw_first is not None and bicubic_first is not None:
        save_comparison(
            args.out_dir / "comparison.png",
            hr_raw_first, lr_raw_first, bicubic_first, pred_raw_first,
        )

    # ---- 總結（同時印逐樣本 PSNR 與 pooled PSNR） ----
    n_frames_label = f"{n_groups} 場模擬 / {len(test_raw)} 幀" if n_groups else f"{len(test_raw)} 筆"
    print(f"\n===== 結果對比（{n_frames_label}） =====")
    print(f"{'方法':<20}{'MSE':>12}{'PSNR':>8}{'pooled':>8}{'SSIM':>9}")
    print(f"{'Bicubic':<20}{bicubic_summary['MSE']:>12.6f}{bicubic_summary['PSNR']:>8.2f}{bicubic_summary.get('pooled_PSNR', float('nan')):>8.2f}{bicubic_summary['SSIM']:>9.4f}")
    if baselines:
        for _n in ("1NN", "inv_dist", "global_lin_extrap", "local_lin_extrap", "LR_1NN"):
            if _n in baselines:
                _v = baselines[_n]
                print(f"{_n:<20}{_v['MSE']:>12.6f}{_v['PSNR']:>8.2f}{_v.get('pooled_PSNR', float('nan')):>8.2f}{_v['SSIM']:>9.4f}")
    print(f"{args.model:<20}{model_summary['MSE']:>12.6f}{model_summary['PSNR']:>8.2f}{model_summary.get('pooled_PSNR', float('nan')):>8.2f}{model_summary['SSIM']:>9.4f}")
    if group_summary:
        print(f"\nGroup-level：{group_summary['n_test_groups']} 場模擬的 PSNR = {group_summary['PSNR_mean']:.2f} ± {group_summary['PSNR_std']:.2f} dB")
    print(f"輸出目錄：{args.out_dir.resolve()}")


# --------------------------------------------------------------------------- #
# 7. 只評估（不訓練）：讀 checkpoint 直接出結果
# --------------------------------------------------------------------------- #
def _reconstruct_train_groups(n_unique, ts, ckpt, group_params, test_mask):
    """eval_only 重建訓練群組：優先用 checkpoint 存的 train_group_ids；
    舊 checkpoint 沒存 → 按 split_mode 重建（避免把 extrapolate/middle 排除的 band 當成訓練集）。"""
    train_group_ids = [int(g) for g in ts.get("train_group_ids", [])]
    if train_group_ids:
        return np.array(sorted(train_group_ids), dtype=np.int64)
    split_mode = ckpt.get("args", {}).get("split_mode", "random")
    tr_max = float(ckpt.get("args", {}).get("train_radius_max") or 0)
    te_min = float(ckpt.get("args", {}).get("test_radius_min") or 0)
    rep_radius = group_params[:, 0]
    if split_mode == "extrapolate" and tr_max > 0:
        return np.where(rep_radius <= tr_max)[0]
    if split_mode == "middle" and tr_max > 0 and te_min > 0:
        return np.where((rep_radius <= tr_max) | (rep_radius >= te_min))[0]
    return np.where(~test_mask)[0]


def run_eval_only(args, device):
    ckpt_path = Path(args.ckpt) if args.ckpt else args.out_dir / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    ts = ckpt.get("train_stats", {})
    mean = np.asarray(ts.get("mean"), dtype=np.float64)
    std = np.asarray(ts.get("std"), dtype=np.float64)
    data_range = np.asarray(ts.get("data_range"), dtype=np.float64)
    n_channels = int(ts.get("n_channels", 1))
    args.n_channels = n_channels
    args.t_start = int(ts.get("t_start", args.t_start))
    args.t_end = int(ts.get("t_end", args.t_end))
    args.t_step = int(ts.get("t_step", args.t_step))
    args.noise_std = float(ts.get("noise_std", args.noise_std))
    args.seed = int(ckpt.get("args", {}).get("seed", args.seed))   # 還原訓練 seed（存在 ckpt["args"]，不在 train_stats）
    factor = int(ts.get("factor", args.factor))
    args.downsample = ts.get("downsample", args.downsample)
    args.loss = ts.get("loss", args.loss)
    args.grad_loss_weight = float(ts.get("grad_loss_weight", args.grad_loss_weight))
    args.split_mode = ts.get("split_mode", args.split_mode)
    args.n_unique = int(ts.get("n_unique", 0))
    args.n_train_groups = int(ts.get("n_train_groups", 0))
    args.n_test_groups = int(ts.get("n_test_groups", args.n_unique - args.n_train_groups if args.n_unique else 0))
    print(f"從 checkpoint 讀取：t=[{args.t_start},{args.t_end}] step={args.t_step}, noise_std={args.noise_std}, factor={factor}, downsample={args.downsample}, split_mode={args.split_mode}, channels={n_channels}")
    print(f"mean={np.round(mean, 6)}, std={np.round(std, 6)}, data_range={np.round(data_range, 6)}")

    rep_fields, keys = load_swe_representative(args.data_path, t_index=args.t_start)
    gids, n_unique = unique_group_ids(rep_fields[..., 0])
    rep_sim_idx = np.array([np.where(gids == g)[0][0] for g in range(n_unique)], dtype=np.int64)
    full = load_swe_timeseries(args.data_path, rep_sim_idx.tolist(), args.t_start, args.t_end, args.t_step)
    sim_params = load_sim_params(args.data_path)
    group_params = sim_params[rep_sim_idx]
    temporal_split = ckpt.get("args", {}).get("temporal_split", "none")
    temporal_train_frac = float(ckpt.get("args", {}).get("temporal_train_frac", 0.25))
    if args.data_range is not None:
        data_range = np.full(n_channels, args.data_range, dtype=np.float64)
    if temporal_split != "none":
        T_sel = full.shape[1]
        K = max(1, int(round(T_sel * temporal_train_frac)))
        split_mode = ckpt.get("args", {}).get("split_mode", "random")
        if split_mode in ("extrapolate", "middle"):
            # 混合：參數外推 + 時間外推（用 checkpoint 存的 test_group_ids 重建參數切分）
            test_group_ids = set(int(g) for g in ts.get("test_group_ids", []))
            test_group_arr = np.array(sorted(test_group_ids), dtype=np.int64)
            test_mask = np.isin(np.arange(n_unique), test_group_arr)
            train_groups = _reconstruct_train_groups(n_unique, ts, ckpt, group_params, test_mask)
            test_groups = np.where(test_mask)[0]
            if temporal_split == "head":
                train_raw = full[train_groups][:, :K].reshape(-1, *full.shape[2:])
                test_raw = full[test_groups][:, K:].reshape(-1, *full.shape[2:])
            else:
                train_raw = full[train_groups][:, -K:].reshape(-1, *full.shape[2:])
                test_raw = full[test_groups][:, :-K].reshape(-1, *full.shape[2:])
            args.n_train_groups = len(train_groups)
            args.n_test_groups = len(test_groups)
        else:
            if temporal_split == "head":
                train_raw = full[:, :K].reshape(-1, *full.shape[2:])
                test_raw = full[:, K:].reshape(-1, *full.shape[2:])
            else:
                train_raw = full[:, -K:].reshape(-1, *full.shape[2:])
                test_raw = full[:, :-K].reshape(-1, *full.shape[2:])
            args.n_train_groups = int(n_unique)
            args.n_test_groups = int(n_unique)
        baselines = None
    else:
        test_group_ids = set(int(g) for g in ts.get("test_group_ids", []))
        test_group_arr = np.array(sorted(test_group_ids), dtype=np.int64)
        test_mask = np.isin(np.arange(n_unique), test_group_arr)
        train_groups = _reconstruct_train_groups(n_unique, ts, ckpt, group_params, test_mask)
        train_mask = np.isin(np.arange(n_unique), train_groups)
        train_raw = full[train_mask].reshape(-1, *full.shape[2:])
        test_raw = full[test_mask].reshape(-1, *full.shape[2:])
        test_groups = np.where(test_mask)[0]
        args.n_test_groups = len(test_groups)
        if args.skip_baselines:
            baselines = None
        else:
            baselines = compute_parameter_baselines(
                full[train_groups], group_params[train_groups],
                full[test_groups], group_params[test_groups],
                data_range,
            )
            baselines.update(compute_linear_extrapolation_baseline(
                full[train_groups], group_params[train_groups],
                full[test_groups], group_params[test_groups],
                data_range,
            ))
            baselines.update(compute_local_linear_extrapolation_baseline(
                full[train_groups], group_params[train_groups],
                full[test_groups], group_params[test_groups],
                data_range, k=32,
            ))
    n_val = len(train_raw) - int((1 - args.val_frac) * len(train_raw))
    print(f"eval_only：{temporal_split if temporal_split != 'none' else 'random'} 切分，train {len(train_raw)} / test {len(test_raw)} 個樣本")

    test_set = SWEDataset(test_raw, factor=factor, mean=mean, std=std, downsample=args.downsample, noise_std=args.noise_std, seed=args.seed + 1)
    if baselines is not None:
        train_seed = int(ts.get("seed", args.seed))
        train_full = SWEDataset(train_raw, factor=factor, mean=mean, std=std,
                                downsample=args.downsample, noise_std=args.noise_std, seed=train_seed)
        baselines.update(compute_lr_nn_baseline(
            train_full.lr, train_full.hr, test_set.lr, test_set.hr, data_range
        ))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    base = int(ckpt.get("args", {}).get("base", 32))
    model_name = ckpt.get("args", {}).get("model", "miniunet")
    model = make_model(model_name, in_ch=n_channels, out_ch=n_channels, base=base).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']+1}, model={model_name})")

    evaluate_and_save(
        model, test_loader, test_raw, mean, std, data_range, factor, args, n_val, device,
        train_min=float(train_raw.min()), train_max=float(train_raw.max()),
        baselines=baselines,
    )


# --------------------------------------------------------------------------- #
# 8. 主程式
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="工程化 U-Net 超解析度訓練（方向 A + B）")
    p.add_argument("--data_path", type=str, default="data/2D_rdb_NA_NA.h5")
    p.add_argument("--t_start", type=int, default=0, help="使用的起始時間步（方向 D：多時間步）")
    p.add_argument("--t_end", type=int, default=20, help="使用的結束時間步（含）。21 幀資料用 20；101 幀資料用 100")
    p.add_argument("--t_step", type=int, default=1, help="時間步間隔（>1 可降低相鄰幀相關性與成本）")
    p.add_argument("--noise_std", type=float, default=0.0, help="加到 LR 的量測雜訊標準差（方向 D，0=不加）")
    p.add_argument("--data_range", type=float, default=None, help="固定 PSNR 的 data_range R（預設從訓練集算）。固定它才能跨實驗比 PSNR")
    p.add_argument("--factor", type=int, default=4, help="下採樣倍率（2/4/8）")
    p.add_argument("--downsample", type=str, default="block", choices=["block", "pointwise"],
                   help="下採樣方式：block=反鋸齒平均、pointwise=隔行取樣（無反鋸齒）")
    p.add_argument("--train_frac", type=float, default=0.9, help="訓練用多少比例的唯一模擬（其餘當測試）")
    p.add_argument("--split_mode", type=str, default="random", choices=["random", "extrapolate", "middle"],
                   help="切分方式：random=隨機；extrapolate=小的訓練、大的測試；middle=兩端訓練、中間測試（缺口）")
    p.add_argument("--train_radius_max", type=float, default=None, help="半徑門檻 1：extrapolate=訓練≤此值；middle=低訓練帶上界")
    p.add_argument("--test_radius_min", type=float, default=None, help="半徑門檻 2：extrapolate=測試≥此值；middle=高訓練帶下界（中間留測試）")
    p.add_argument("--temporal_split", type=str, default="none", choices=["none", "head", "tail"],
                   help="時間切分（時間外推）：head=前幾幀訓練推後面；tail=後幾幀推前面；none=按模擬切分")
    p.add_argument("--temporal_train_frac", type=float, default=0.25, help="時間切分時訓練用的幀比例（0.25=前 25% 幀訓練）")
    p.add_argument("--val_frac", type=float, default=0.1, help="訓練集內再切出的驗證比例")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "charbonnier"],
                   help="主損失：mse(L2) / l1 / charbonnier")
    p.add_argument("--grad_loss_weight", type=float, default=0.0, help="梯度 loss 權重 λ（0 表示不用）")
    p.add_argument("--base", type=int, default=32, help="網路首層通道數（寬度）")
    p.add_argument("--model", type=str, default="miniunet", choices=["miniunet", "unet2d", "srcnn"],
                   help="模型架構：miniunet / unet2d（深層 U-Net）/ srcnn（3 層 CNN）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=15, help="早停：連續多少 epoch 無改善即停止")
    p.add_argument("--lr_patience", type=int, default=5, help="ReduceLROnPlateau 的 patience")
    p.add_argument("--out_dir", type=Path, default=Path("outputs_unet_v2"))
    p.add_argument("--resume", type=str, default=None, help="從 checkpoint 繼續訓練")
    p.add_argument("--eval_only", action="store_true", help="不訓練，只讀 checkpoint 做評估")
    p.add_argument("--skip_baselines", action="store_true", help="不計算 baselines（多模型共用同一組 baseline 時，只對第一個模型算）")
    p.add_argument("--ckpt", type=str, default=None, help="評估用的 checkpoint（預設 out_dir/model_best.pt）")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        run_eval_only(args, device)
        return

    # ---- 載入資料（方向 D：多時間步，兩階段以省記憶體） ----
    rep_fields, keys = load_swe_representative(args.data_path, t_index=args.t_start)
    n_sims, H, W, C = rep_fields.shape
    args.n_channels = int(C)
    radii = load_dam_radius(args.data_path)   # 每筆「模擬」的 dam_radius（與 sorted keys 對齊）
    sim_params = load_sim_params(args.data_path)   # [N, 2]（dam_radius, inner_height）

    # ---- 在「模擬」層級去重（相同 dam_radius → 整段時間都一樣） ----
    gids, n_unique = unique_group_ids(rep_fields[..., 0])   # gids: [n_sims]，用 h 通道去重
    rep_radius = np.array([radii[np.where(gids == g)[0][0]] for g in range(n_unique)])

    if args.split_mode == "middle" and args.train_radius_max is not None and args.test_radius_min is not None:
        # 兩端訓練、中間測試（缺口）：train = radius ≤ max 或 ≥ min，test = 中間
        train_groups = np.where((rep_radius <= args.train_radius_max) | (rep_radius >= args.test_radius_min))[0]
        test_groups = np.where((rep_radius > args.train_radius_max) & (rep_radius < args.test_radius_min))[0]
    elif args.split_mode == "extrapolate" and args.train_radius_max is not None and args.test_radius_min is not None:
        # 真缺口外推：小的訓練、大的測試（中間缺口不用）
        train_groups = np.where(rep_radius <= args.train_radius_max)[0]
        test_groups = np.where(rep_radius >= args.test_radius_min)[0]
    elif args.split_mode == "extrapolate":
        order = np.argsort(rep_radius)   # 壩半徑由小到大
        n_g = max(1, int(round(args.train_frac * n_unique)))
        train_groups = order[:n_g]
        test_groups = order[n_g:]
    else:
        order = np.random.default_rng(args.seed).permutation(n_unique)
        n_g = max(1, int(round(args.train_frac * n_unique)))
        train_groups = order[:n_g]
        test_groups = order[n_g:]

    n_train_groups = len(train_groups)
    n_test_groups = len(test_groups)
    test_group_ids = sorted(int(g) for g in test_groups)

    # ---- 只載入「唯一組」的完整時間序列（1000 場唯一，每場 21 幀，全載 ~4.1GB） ----
    rep_sim_idx = np.array([np.where(gids == g)[0][0] for g in range(n_unique)], dtype=np.int64)
    full = load_swe_timeseries(args.data_path, rep_sim_idx.tolist(), args.t_start, args.t_end, args.t_step)  # [n_unique, T_sel, H, W, C]
    T_sel = full.shape[1]
    group_params = sim_params[rep_sim_idx]   # [n_unique, 2]（每組代表模擬的參數）

    if args.temporal_split != "none":
        K = max(1, int(round(T_sel * args.temporal_train_frac)))
        if args.split_mode in ("extrapolate", "middle"):
            # 混合：參數外推 + 時間外推（雙外推）
            if args.temporal_split == "head":
                train_raw = full[train_groups][:, :K].reshape(-1, H, W, C)
                test_raw = full[test_groups][:, K:].reshape(-1, H, W, C)
            else:  # tail
                train_raw = full[train_groups][:, -K:].reshape(-1, H, W, C)
                test_raw = full[test_groups][:, :-K].reshape(-1, H, W, C)
            n_train_groups = len(train_groups)
            n_test_groups = len(test_groups)
        else:
            # 純時間外推：所有場共享，切時間
            if args.temporal_split == "head":
                train_raw = full[:, :K].reshape(-1, H, W, C)
                test_raw = full[:, K:].reshape(-1, H, W, C)
            else:  # tail
                train_raw = full[:, -K:].reshape(-1, H, W, C)
                test_raw = full[:, :-K].reshape(-1, H, W, C)
            n_train_groups = int(n_unique)
            n_test_groups = int(n_unique)
        print(f"時間切分（{args.temporal_split}）：train {K} 幀 / test {T_sel - K} 幀")
    else:
        train_raw = full[train_groups].reshape(-1, H, W, C)
        test_raw = full[test_groups].reshape(-1, H, W, C)

    overlap = len(set(train_groups.tolist()) & set(test_groups.tolist()))
    print(f"總模擬 {n_sims} 筆，唯一模擬場數 = {n_unique}，每場 {T_sel} 個時間步（共 {n_unique * T_sel} 個樣本）")
    print(f"{args.split_mode} split：train {n_train_groups} 場 / test {n_test_groups} 場，重疊 = {overlap}（應為 0）")
    if args.split_mode == "extrapolate":
        print(f"  外推：train 半徑 [{rep_radius[train_groups].min():.4f}, {rep_radius[train_groups].max():.4f}]，"
              f"test 半徑 [{rep_radius[test_groups].min():.4f}, {rep_radius[test_groups].max():.4f}]")
    elif args.split_mode == "middle":
        print(f"  中間測試帶 radius ∈ ({args.train_radius_max}, {args.test_radius_min})，"
              f"train 為兩端 ≤{args.train_radius_max} 或 ≥{args.test_radius_min}")

    args.n_unique = int(n_unique)
    args.n_train_groups = int(n_train_groups)
    args.n_test_groups = int(n_test_groups)

    # 逐通道歸一化（C=1 為單通道，C=3 為 h/u/v）
    mean = train_raw.reshape(-1, C).mean(axis=0).astype(np.float64)          # [C]
    std = train_raw.reshape(-1, C).std(axis=0).astype(np.float64)            # [C]
    data_range = train_raw.reshape(-1, C).max(axis=0) - train_raw.reshape(-1, C).min(axis=0)  # [C]
    if args.data_range is not None:
        data_range = np.full(C, args.data_range, dtype=np.float64)   # 固定 R，跨實驗可比
    print(f"train mean={np.round(mean, 6)}, std={np.round(std, 6)}, data_range={np.round(data_range, 6)}")

    # ---- baseline（oracle 1-NN/線性內插 + 線性外推；時間切分時不適用，因為需要「同時刻」訓練場） ----
    if args.skip_baselines:
        baselines = None
    elif args.temporal_split == "none":
        baselines = compute_parameter_baselines(
            full[train_groups], group_params[train_groups],
            full[test_groups], group_params[test_groups],
            data_range,
        )
        baselines.update(compute_linear_extrapolation_baseline(
            full[train_groups], group_params[train_groups],
            full[test_groups], group_params[test_groups],
            data_range,
        ))
        baselines.update(compute_local_linear_extrapolation_baseline(
            full[train_groups], group_params[train_groups],
            full[test_groups], group_params[test_groups],
            data_range, k=32,
        ))
    else:
        baselines = None
    print(f"downsample={args.downsample}, loss={args.loss}, grad_loss_weight={args.grad_loss_weight}, channels={C}")

    # ---- 訓練集內再切 train / val（按「模擬」切，避免同場相鄰幀洩漏到 val） ----
    trainval = SWEDataset(train_raw, factor=args.factor, mean=mean, std=std, downsample=args.downsample, noise_std=args.noise_std, seed=args.seed)
    n_sims_train = int(getattr(args, "n_train_groups", 0) or 0)
    T_per_sim = (len(trainval) // n_sims_train) if n_sims_train > 0 else 0
    if T_per_sim > 1:
        n_val_sims = max(1, int(round(args.val_frac * n_sims_train)))
        sim_perm = np.random.default_rng(args.seed).permutation(n_sims_train)
        val_sims = set(int(s) for s in sim_perm[:n_val_sims])
        val_idx = [i for i in range(len(trainval)) if (i // T_per_sim) in val_sims]
        train_idx = [i for i in range(len(trainval)) if (i // T_per_sim) not in val_sims]
        train_set = Subset(trainval, train_idx)
        val_set = Subset(trainval, val_idx)
        n_val = len(val_idx)
    else:
        print(f"[WARN] 每場只有 {T_per_sim} 幀 → frame-level split（此時與 sim-level 等價，無洩漏）")
        n_train = int((1 - args.val_frac) * len(trainval))
        n_val = len(trainval) - n_train
        gen = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(trainval, [n_train, n_val], generator=gen)

    test_set = SWEDataset(test_raw, factor=args.factor, mean=mean, std=std, downsample=args.downsample, noise_std=args.noise_std, seed=args.seed + 1)

    # ---- LR 最近鄰 baseline（看噪聲 LR，受噪聲影響） ----
    if baselines is not None:
        baselines.update(compute_lr_nn_baseline(
            trainval.lr, trainval.hr, test_set.lr, test_set.hr, data_range
        ))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ---- 模型 / 優化器 / 調度器 / 損失 ----
    model = make_model(args.model, in_ch=C, out_ch=C, base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    criterion = make_criterion(args.loss)

    best_path = args.out_dir / "model_best.pt"
    start_epoch = 0
    best_val_loss = float("inf")

    # ---- resume：還原模型 / 優化器 / 調度器 ----
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from {args.resume}, epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    # ---- 訓練 + 早停 + 存最佳 ----
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            pred = upsample_and_predict(model, lr_img, hr_img.shape[-2:])
            loss = compute_total_loss(pred, hr_img, criterion, args.grad_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lr_img.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                pred = upsample_and_predict(model, lr_img, hr_img.shape[-2:])
                val_loss += compute_total_loss(pred, hr_img, criterion, args.grad_loss_weight).item() * lr_img.size(0)
        val_loss /= len(val_set)

        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-7
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "train_stats": {
                        "mean": mean, "std": std, "data_range": data_range,
                        "t_start": args.t_start, "t_end": args.t_end, "t_step": args.t_step,
                        "noise_std": args.noise_std, "factor": args.factor,
                        "downsample": args.downsample, "loss": args.loss, "grad_loss_weight": args.grad_loss_weight,
                        "split_mode": args.split_mode, "test_group_ids": test_group_ids,
                        "train_group_ids": sorted(int(g) for g in train_groups),
                        "n_unique": n_unique, "n_train_groups": n_train_groups, "n_test_groups": n_test_groups, "n_channels": C,
                    },
                },
                best_path,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e} "
            f"| best={best_val_loss:.6f} | patience={patience_counter}/{args.patience}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ---- 載入最佳模型做測試 ----
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']+1} (val_loss={best_val_loss:.6f})")

    evaluate_and_save(
        model, test_loader, test_raw, mean, std, data_range, args.factor, args, n_val, device,
        train_min=float(train_raw.min()), train_max=float(train_raw.max()),
        baselines=baselines,
    )


if __name__ == "__main__":
    main()
