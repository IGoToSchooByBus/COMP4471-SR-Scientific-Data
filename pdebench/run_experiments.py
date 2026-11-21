"""一次依序跑完 4 個模型對照實驗。用法：python run_experiments.py"""
import subprocess
import sys

BASE = [
    sys.executable, "train_unet_v2.py",
    "--data_path", "data/2D_rdb_NA_NA_.h5",
    "--downsample", "pointwise", "--factor", "4",
    "--loss", "l1", "--grad_loss_weight", "0.5",
    "--t_start", "0", "--t_end", "20", "--t_step", "1",
]

EXPERIMENTS = [
    ("SRCNN",       ["--model", "srcnn"],                     "outputs_srcnn"),
    ("MiniUNet",    ["--model", "miniunet", "--base", "32"],  "outputs_miniunet"),
    ("UNet2d",      ["--model", "unet2d", "--base", "32"],    "outputs_unet2d"),
    ("UNet2d wide", ["--model", "unet2d", "--base", "64"],    "outputs_unet2d_wide"),
]


def main():
    for i, (name, extra, out_dir) in enumerate(EXPERIMENTS, 1):
        cmd = BASE + extra + ["--out_dir", out_dir]
        print(f"\n[{i}/{len(EXPERIMENTS)}] {name} ...")
        print("  " + " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[WARN] {name} failed (returncode={r.returncode}), continue next ...")
    print("\n===== All done =====")


if __name__ == "__main__":
    main()
