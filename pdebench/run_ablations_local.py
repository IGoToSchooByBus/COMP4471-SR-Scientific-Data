"""本地跑消融（7 實驗 × 4 模型 = 28 run，依序跑）。
用法：python run_ablations_local.py
AMP 預設開啟（本地 CUDA 會用 tensor cores，快 2~4 倍）。"""
import subprocess
import sys

MODELS = [
    ("srcnn",     ["--model", "srcnn"]),
    ("miniunet",  ["--model", "miniunet", "--base", "32"]),
    ("unet2d",    ["--model", "unet2d", "--base", "32"]),
    ("unet2d_w",  ["--model", "unet2d", "--base", "64"]),
]

def batch_for(m):
    return "128" if m in ("srcnn", "miniunet") else "64"

BASE = ["train_unet_v2.py", "--data_path", "data/2D_rdb_NA_NA_.h5",
        "--downsample", "pointwise", "--factor", "4",
        "--loss", "l1", "--grad_loss_weight", "0.5",
        "--t_start", "0", "--t_end", "20", "--t_step", "1",
        "--epochs", "60", "--patience", "10"]

EXPS = [
    ("noise005", ["--noise_std", "0.05"]),
    ("noise010", ["--noise_std", "0.1"]),

]

total = len(EXPS) * len(MODELS)
n = 0
for exp_name, exp_flags in EXPS:
    for model_name, model_flags in MODELS:
        n += 1
        out = f"new_result/ab_{exp_name}_{model_name}"
        cmd = ([sys.executable, "-u"] + BASE + model_flags
               + ["--batch_size", batch_for(model_name)] + exp_flags
               + ["--out_dir", out])
        print(f"\n[{n}/{total}] {exp_name} / {model_name}")
        print("  " + " ".join(cmd))
        subprocess.run(cmd)
print("\n===== All done =====")
