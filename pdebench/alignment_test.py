"""pointwise 1.5px 對齊測試。

問題：pointwise 下採樣 LR[i] = HR[4i]，但 bicubic(align_corners=False) 把 LR pixel i
放在 HR 座標 4i+1.5 → bicubic 輸出和真值差 1.5 px 的全域位移。

測試：
1. 對 bicubic 輸出 roll −2..+2 px（沿兩軸同方向），看哪個 shift 的 PSNR 最高。
2. 試「hr[2::4]」變體（LR[i] = HR[4i+2] → 殘留 0.5 px）。

判斷：若 shift≠0 的 Δ > +0.5 dB → confound 存在，重訓前要決定（改 LR 取樣或報告明說）。

用法（本地，需 torch）：python alignment_test.py data/2D_rdb_NA_NA_.h5
"""
import sys
import numpy as np
import h5py
import torch
import torch.nn.functional as F


def bicubic_up(lr_h, target=(128, 128)):
    t = torch.from_numpy(np.ascontiguousarray(lr_h, dtype=np.float32))[None, None]
    up = F.interpolate(t, size=target, mode="bicubic", align_corners=False)
    return up[0, 0].numpy()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/2D_rdb_NA_NA_.h5"
    R = 2.5   # h 通道的大致 dynamic range（只看相對 Δ，絕對值不重要）

    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())[:10]
        print(f"用 {len(keys)} 場 × 5 幀 = {len(keys)*5} 個樣本測試\n")

        mse_shift = {s: [] for s in (-2, -1, 0, 1, 2)}
        mse_alt = []

        for key in keys:
            data = f[key]["data"][:]   # [21,128,128,3]
            for t in (0, 5, 10, 15, 20):
                hr = data[t, :, :, 0].astype(np.float32)   # h 通道

                # 標準 pointwise：LR[i] = HR[4i]
                lr = hr[::4, ::4]
                bic = bicubic_up(lr)
                for s in mse_shift:
                    pred = np.roll(bic, (s, s), axis=(0, 1))
                    mse_shift[s].append(float(((hr - pred) ** 2).mean()))

                # 變體：LR[i] = HR[4i+2]（殘留 0.5 px）
                bic_alt = bicubic_up(hr[2::4, 2::4])
                mse_alt.append(float(((hr - bic_alt) ** 2).mean()))

    print("===== ① bicubic roll 測試（shift = 沿兩軸平移 px） =====")
    base_mse = float(np.mean(mse_shift[0]))
    best_s, best_p = 0, 10.0 * np.log10(R ** 2 / base_mse)
    for s in sorted(mse_shift):
        m = float(np.mean(mse_shift[s]))
        p = 10.0 * np.log10(R ** 2 / m)
        d = 10.0 * np.log10(base_mse / m)
        tag = " ← 最佳" if p > best_p else ""
        if p > best_p:
            best_s, best_p = s, p
        print(f"  shift={s:+d}: PSNR={p:.2f} dB（Δ vs shift=0: {d:+.2f} dB）{tag}")

    m_alt = float(np.mean(mse_alt))
    p_alt = 10.0 * np.log10(R ** 2 / m_alt)
    print(f"\n===== ② hr[2::4] 變體 =====")
    print(f"  PSNR={p_alt:.2f} dB（Δ vs 標準 shift=0: {10.0*np.log10(base_mse/m_alt):+.2f} dB）")

    print("\n===== 判斷 =====")
    gain = 10.0 * np.log10(base_mse / float(np.mean(mse_shift[best_s])))
    if gain > 0.5 or 10.0 * np.log10(base_mse / m_alt) > 0.5:
        print(f"⚠️ 最佳 shift={best_s:+d}（Δ=+{gain:.2f} dB）或 hr[2::4] 變體提升 > 0.5 dB")
        print("   → confound 存在！重訓前要決定：改 LR 取樣（hr[2::4]）或報告明說 bicubic 被位移懲罰。")
    else:
        print(f"✅ 所有 shift 的 Δ ≤ 0.5 dB（最佳 shift={best_s:+d}，Δ=+{gain:.2f} dB）→ confound 可忽略。")


if __name__ == "__main__":
    main()
