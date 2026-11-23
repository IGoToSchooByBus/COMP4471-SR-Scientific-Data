"""本地補算（含 pooled_PSNR / MSE_by_time / baselines），不用重新訓練。
baselines 與模型無關 → 每個實驗只對第一個模型算一次，其餘用 --skip_baselines + 複製。
用法（在任何目錄都能跑）：python pdebench/run_eval.py"""
import subprocess
import sys
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent   # pdebench/（腳本所在目錄）

MODELS = ["srcnn", "miniunet", "unet2d", "unet2d_w"]

BASE_MODELS = {
    "srcnn": "outputs_srcnn",
    "miniunet": "outputs_miniunet",
    "unet2d": "outputs_unet2d",
    "unet2d_w": "outputs_unet2d_wide",
}

CKPTS = []
for suffix in BASE_MODELS.values():
    CKPTS.append(f"ab_base/{suffix}")
for exp in ["noise005", "noise010", "noise020", "extrap", "temphead", "temptail", "mixed"]:
    for m in MODELS:
        CKPTS.append(f"ab_{exp}/ab_{exp}_{m}")

# 按實驗分組（每組第一個 = 算 baselines 的模型）
EXPERIMENTS = []
for exp in ["ab_base"] + [f"ab_{e}" for e in ["noise005", "noise010", "noise020", "extrap", "temphead", "temptail", "mixed"]]:
    outs = [c for c in CKPTS if c.startswith(exp + "/")]
    if outs:
        EXPERIMENTS.append((exp, outs))


def main():
    ok, fail = 0, 0
    for exp, outs in EXPERIMENTS:
        first = outs[0]
        print(f"\n=== eval_only（含 baselines）: {first} ===")
        r = subprocess.run(
            [sys.executable, "train_unet_v2.py", "--eval_only",
             "--data_path", "data/2D_rdb_NA_NA_.h5", "--out_dir", first],
            cwd=str(BASE),
        )
        if r.returncode != 0:
            print(f"[WARN] failed: {first}")
            fail += 1
            continue
        ok += 1
        # 讀共享 baselines（與模型無關）
        with open(BASE / first / "baselines.json", "r", encoding="utf-8") as f:
            shared = json.load(f).get("baselines")
        for out in outs[1:]:
            print(f"\n=== eval_only（skip baselines）: {out} ===")
            r = subprocess.run(
                [sys.executable, "train_unet_v2.py", "--eval_only", "--skip_baselines",
                 "--data_path", "data/2D_rdb_NA_NA_.h5", "--out_dir", out],
                cwd=str(BASE),
            )
            if r.returncode != 0:
                print(f"[WARN] failed: {out}")
                fail += 1
                continue
            ok += 1
            # 把共享 baselines 寫回（group_summary 保留各自的模型數字）
            with open(BASE / out / "baselines.json", "r", encoding="utf-8") as f:
                bj = json.load(f)
            bj["baselines"] = shared
            with open(BASE / out / "baselines.json", "w", encoding="utf-8") as f:
                json.dump(bj, f, indent=2, allow_nan=False)   # 嚴格 JSON：若有 NaN 會直接報錯，而不是寫出非法檔案
    print(f"\n===== 完成 {ok}/{len(CKPTS)}，失敗 {fail} =====")


if __name__ == "__main__":
    main()
