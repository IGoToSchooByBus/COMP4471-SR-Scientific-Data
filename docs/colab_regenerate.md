# Colab 資料重新生成（重置版）

目標：生成 **1000 樣本、3 通道（h/u/v）、每 5 步存 21 幀** 的資料。

> ⚠️ **Cell 1 只能跑一次**（它有 `rm -rf`，重跑會造成巢狀 clone）。之後要改東西直接改檔案，不要再 clone。

---

## Cell 1：重置 + 安裝（只跑一次）

```python
%cd /content
!rm -rf PDEBench
!apt-get update -qq && apt-get install -y -qq gfortran
!git clone https://github.com/pdebench/PDEBench.git
%cd PDEBench
!pip install -q clawpack hydra-core python-dotenv h5py numpy scipy matplotlib
import os
os.environ["PYTHONPATH"] = "/content/PDEBench" + os.pathsep + os.environ.get("PYTHONPATH", "")
print("重置 + 安裝 + PYTHONPATH ✅")
```

## Cell 2：套所有修改（含 3 通道，強制整個方法替換）

```python
from pathlib import Path
import re

# ===== sim_radial_dam_break.py =====
p = Path("pdebench/data_gen/src/sim_radial_dam_break.py")
s = p.read_text()

new_method = '''    def save_state_to_disk(self, data_f, seed_str) -> None:
        T = np.asarray(self.save_state["t"])[::5]
        X = np.asarray(self.save_state["x"])
        Y = np.asarray(self.save_state["y"])
        H = np.expand_dims(np.asarray(self.save_state["h"])[::5], -1)
        U = np.expand_dims(np.asarray(self.save_state["u"])[::5], -1)
        V = np.expand_dims(np.asarray(self.save_state["v"])[::5], -1)
        data = np.concatenate([H, U, V], axis=-1)   # [T/5, H, W, 3]

        data_f.create_dataset(f"{seed_str}/data", data=data, dtype="f")
        data_f.create_dataset(f"{seed_str}/grid/x", data=X, dtype="f")
        data_f.create_dataset(f"{seed_str}/grid/y", data=Y, dtype="f")
        data_f.create_dataset(f"{seed_str}/grid/t", data=T, dtype="f")'''

s = re.sub(r'    def save_state_to_disk\(self, data_f, seed_str\) -> None:.*?(?=\n    def )',
           new_method, s, flags=re.DOTALL)
assert "np.concatenate([H, U, V]" in s, "save_state_to_disk 沒替換成功"
s = s.replace("        h_in = 2.0", "        h_in = self.inner_height")
p.write_text(s)

# ===== gen_radial_dam_break.py =====
p = Path("pdebench/data_gen/gen_radial_dam_break.py")
s = p.read_text()
s = s.replace("    # config.sim.inner_height = np.random.uniform(1.5, 2.5)",
              "    config.sim.inner_height = rng.uniform(1.5, 2.5)")
s = s.replace("        dam_radius=config.sim.dam_radius,\n        xdim=config.sim.xdim,",
              "        dam_radius=config.sim.dam_radius,\n        inner_height=config.sim.inner_height,\n        xdim=config.sim.xdim,")
s = s.replace('config.output_path = output_path / config.output_path.with_suffix(".h5")',
              'config.output_path = str(output_path.with_suffix(".h5"))')
s = s.replace("scenario.run(T=config.sim.T_end, tsteps=config.sim.n_time_steps, plot=False)",
              "scenario.run(T=config.sim.T_end, tsteps=config.sim.n_time_steps)")
s = re.sub(r"num_samples_final = \d+", "num_samples_final = 1000", s)
p.write_text(s)

# 驗證（4 個都要對）
sim = Path("pdebench/data_gen/src/sim_radial_dam_break.py").read_text()
gen = Path("pdebench/data_gen/gen_radial_dam_break.py").read_text()
print("3 通道:", "np.concatenate([H, U, V]" in sim)
print("inner_height 用參數:", "h_in = self.inner_height" in sim)
print("output_path 已修:", 'str(output_path.with_suffix(".h5"))' in gen)
print("plot=False 已移除:", "plot=False" not in gen)
print("num_samples:", re.search(r"num_samples_final = (\d+)", gen).group(1))
```

**全部印 `True`（num_samples 印 1000）後**才跑下一步。

## Cell 3：生成（~2 小時）

```python
!find /content/PDEBench -name "*.h5" -delete
!python pdebench/data_gen/gen_radial_dam_break.py
```

## Cell 3.5：驗證通道 = 3（關鍵！）

```python
import h5py, glob
for f in glob.glob('/content/PDEBench/**/*.h5', recursive=True):
    with h5py.File(f, 'r') as h:
        ks = sorted(h.keys())
        shape = h[ks[0]]['data'].shape
        print(f"樣本數: {len(ks)}, data 形狀: {shape}, 通道: {shape[-1]}")
        print("✅ 通道 = 3" if shape[-1] == 3 else f"❌ 通道 = {shape[-1]}（應為 3，回去檢查 Cell 2）")
```

**確認 ✅ 才存 Drive。**

## Cell 4：存 Google Drive

```python
from google.colab import drive
import shutil, glob
drive.mount('/content/drive')
for f in glob.glob('/content/PDEBench/**/*.h5', recursive=True):
    shutil.copy(f, '/content/drive/MyDrive/')
    print("已存:", f)
```

---

## 回 Windows 驗證

```bash
python -c "import h5py; f=h5py.File('data/<檔名>.h5','r'); ks=sorted(f.keys()); g=f[ks[0]]; print('樣本數:', len(ks)); print('data 形狀:', g['data'].shape)"
```

預期：`樣本數: 1000`、`data 形狀: (21, 128, 128, 3)`。
