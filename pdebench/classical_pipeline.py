"""Classical pipeline baseline：參數反演 + forward model（Clawpack）。

流程：
  test LR（t=0 幀，h 通道）→ 反演 (dam_radius, inner_height)
  → Clawpack forward 模擬 → 21 幀 HR → 對比 test HR（逐幀 PSNR）。

用法（在 Colab 跑，需 Clawpack + PYTHONPATH=/content/PDEBench）：
  python classical_pipeline.py /content/data.h5 0.6 0.0
  # argv[2] = test 的 dam_radius 下限（外推 split 用 0.6；random split 用 0.0 = 全部）
  # argv[3] = 加到 LR 的噪聲 σ（0 = 乾淨；模擬反演對噪聲的敏感度）

輸出：classical_pipeline.json（每個 test 模擬的 (r_true, h_true, r_est, h_est) + 整體 PSNR）。
"""
import sys
import re
import json
import numpy as np
import h5py


def load_test_sims(path, r_min=0.0):
    """讀出 test 模擬（dam_radius >= r_min）。回傳 [(key, r_true, h_true, data)]。"""
    sims = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            cfg = f[key].attrs.get("config", "")
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8", errors="ignore")
            m_r = re.search(r"dam_radius:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cfg)
            m_h = re.search(r"inner_height:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cfg)
            r_true = float(m_r.group(1)) if m_r else np.nan
            h_true = float(m_h.group(1)) if m_h else np.nan
            if not np.isnan(r_true) and r_true >= r_min:
                sims.append((key, r_true, h_true, f[key]["data"][:]))   # data: [21,128,128,3]
    return sims


def invert_params(lr_h_t0, domain_size=5.0):
    """從 t=0 的 LR 水位場反演 (dam_radius, inner_height)。
    t=0 的場是「圓盤」：盤內 = inner_height（均勻），盤外 = 1.0（背景）。
    用面積估計半徑（對邊界量化較穩健）、用最大值估計盤內高度。
    domain_size = 5.0：sim_radial_dam_break 的 domain 是 [-2.5, 2.5]²（5 單位寬）。"""
    max_h = float(lr_h_t0.max())
    min_h = float(lr_h_t0.min())
    thr = (max_h + min_h) / 2.0
    mask = lr_h_t0 > thr
    if mask.sum() == 0:
        return 0.3, min_h
    H, W = lr_h_t0.shape
    r_est = float(np.sqrt(mask.sum() / np.pi)) / min(H, W) * domain_size
    h_est = float(lr_h_t0[mask].max())
    return r_est, h_est


def forward_simulate(r, h_inner):
    """用反演的 (r, h_inner) 跑 Clawpack，回傳 [21, 128, 128, 3]。"""
    from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D
    scenario = RadialDamBreak2D(grav=1.0, dam_radius=r, inner_height=h_inner, xdim=128, ydim=128)
    scenario.run(T=1.0, tsteps=100)
    H = np.asarray(scenario.save_state["h"])   # [101,128,128]
    U = np.asarray(scenario.save_state["u"])
    V = np.asarray(scenario.save_state["v"])
    return np.stack([H, U, V], axis=-1)[::5]   # [21,128,128,3]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/2D_rdb_NA_NA_.h5"
    r_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    noise_std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    sims = load_test_sims(path, r_min)
    print(f"test sims (r >= {r_min}): {len(sims)}")
    rng = np.random.default_rng(0)

    results = []
    mse_all, psnr_all = [], []
    for key, r_true, h_true, data in sims:
        # 反演：用 t=0 的 LR（pointwise ×4 + 可選噪聲）
        hr_t0 = data[0]                       # [128,128,3]
        lr_h_t0 = hr_t0[::4, ::4, 0].astype(np.float64)   # [32,32] h 通道
        if noise_std > 0:
            lr_h_t0 = lr_h_t0 + rng.normal(0.0, noise_std, lr_h_t0.shape)
        r_est, h_est = invert_params(lr_h_t0)
        err_r = r_est - r_true
        print(f"{key}: r_true={r_true:.4f} h_true={h_true:.4f} | r_est={r_est:.4f} h_est={h_est:.4f} | dr={err_r:+.4f}")

        # forward model
        pred = forward_simulate(r_est, h_est)   # [21,128,128,3]

        # 評估（逐幀、分通道、用各自 data_range；常數場與完美吻合的幀跳過）
        psnr_sim = []
        for t in range(data.shape[0]):
            for c in range(3):
                dr = float(data[t, :, :, c].max() - data[t, :, :, c].min())
                if dr < 1e-12:
                    continue   # 常數場（t=0 的 u/v=0）→ PSNR 無定義，跳過
                mse = float(((data[t, :, :, c] - pred[t, :, :, c]) ** 2).mean())
                if mse < 1e-18:
                    continue   # 完美吻合（初始條件在網格上量化一致）→ PSNR 無限，跳過
                mse_all.append(mse)
                psnr = 10.0 * np.log10(dr ** 2 / mse)
                psnr_sim.append(psnr)
                psnr_all.append(psnr)

        results.append({
            "key": key, "r_true": r_true, "h_true": h_true,
            "r_est": r_est, "h_est": h_est, "r_err": err_r,
            "PSNR_sim_mean": float(np.mean(psnr_sim)) if psnr_sim else float("nan"),
        })

    summary = {
        "r_min": r_min, "noise_std": noise_std, "n_test": len(sims),
        "MSE_pooled": float(np.mean(mse_all)),
        "PSNR_mean": float(np.mean(psnr_all)),
        "per_sim": results,
    }
    out = f"classical_pipeline_r{r_min}_n{noise_std}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n===== classical pipeline：PSNR = {summary['PSNR_mean']:.2f} dB（{len(sims)} 場） =====")
    print(f"已存 {out}")


if __name__ == "__main__":
    main()
