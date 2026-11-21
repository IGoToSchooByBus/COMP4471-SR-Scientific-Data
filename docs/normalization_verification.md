# 归一化统计验证（仅 CPU，不训练）

脚本：`tools/verify_normalization_stats.py`。只依赖 NumPy 和 h5py；不会导入 PyTorch、修改训练脚本、加载 checkpoint 或下载数据。

## 放置数据

将已有 HDF5 放入仓库的 `pdebench/data/`。该目录整体被 `.gitignore` 忽略。

默认文件名为 `2D_rdb_NA_NA_1000.h5`；可以保留原文件名并通过 `--data-path` 指定。

预期格式为：顶层模拟组（例如 `0000`），各组包含 `[T,H,W,3]` 的 `data`，通道顺序为 h/u/v。普通实验应使用 21 帧数据。

## 从仓库根目录运行（PowerShell）

```powershell
python -B tools/verify_normalization_stats.py --self-test

python -B tools/verify_normalization_stats.py `
  --data-path "pdebench/data/2D_rdb_NA_NA_1000.h5" `
  --seed 0 --train-frac 0.9 `
  --t-start 0 --t-end 20 --t-step 1 `
  --output "pdebench/data/normalization_check.json"
```

如果缺少依赖，在你的项目 Python 环境运行：`python -m pip install numpy h5py`。
报告使用独占写入，不覆盖已有文件；再次运行请选择新报告名或省略 `--output`。

有旧结果时，将输出目录放回仓库，脚本会自动查找所有 `train_stats.json`。也可以显式传入外部文件（可重复指定）：

```powershell
python -B tools/verify_normalization_stats.py `
  --data-path "pdebench/data/你的文件名.h5" `
  --stats-json "E:/旧结果/ab_noise005_miniunet/train_stats.json" `
  --output "pdebench/data/normalization_check_with_saved_stats.json"
```

## 与原训练保持一致的范围

当前脚本只验证普通的 `split_mode=random`、`temporal_split=none` 实验。

1. 按 HDF5 顶层 key 排序。
2. 将 `t_start` 的 h 通道转换为 float32，按其完整字节去重；每组采用排序后首次出现的模拟。
3. 使用 `np.random.default_rng(seed).permutation(n_unique)`，保留原 permutation 顺序。
4. 使用 `max(1, int(round(train_frac * n_unique)))` 个训练模拟。
5. 按训练模拟顺序、选定时间帧顺序及像素 C-order 处理数据。
6. 统计对象是切出 validation **之前**的整个 `train_raw`。不额外剔除 validation，不加入 test，不加入噪声。

默认参数与 `docs/colab_ablation.md` 的普通噪声实验一致：seed=0、train_frac=0.9、帧 0..20。
训练脚本的通用 `t_end` 默认值是 100，文档中对 21 帧数据显式传 20；验证脚本默认取 20。
如果已完成实验用过其他 seed、训练比例或时间区间，必须通过参数传入原值。

`--self-test` 会提取实际训练脚本的加载函数和 `main()` 中训练前的划分/统计语句，在临时小 HDF5 上验证：去重组、代表模拟、train/test ID 顺序、时间帧、每个原始像素、float32 mean/std 完全匹配。覆盖重复模拟和三个参数组合，不运行训练代码。

## 低内存计算原理

脚本读取两遍选中的训练数据。通过 HDF5 行块读取，默认 `--chunk-rows 65536` 个像素一块。
除去重所需的初始 h 场字节外，仅保留当前数据块和少量工作数组；不会同时加载整个 HDF5、全部时间序列或全尺寸方差临时数组。

原始 float32 结果使用连续的 float32 逐行累加复现，包括全局 float32 mean、float32 偏差平方和总体标准差（ddof=0）。**不能用每块 float32 均值再平均**，因为那会改变舍入顺序，掩盖需要验证的问题。
每次运行会先比较当前 NumPy 版本上的分块复现与原始整体 `mean/std(axis=0)`；若不完全一致则停止，避免把近似结果当成原算法结果。

float64 参考值基于与训练一致的 float32 输入数据，分两遍以 float64 计算均值、中心化平方偏差和标准差。原始数据存储类型若为 float64，也先转 float32，以符合训练加载逻辑。

## 如何理解输出

每个通道报告 mean/std 的原 float32 值、float64 参考值、绝对误差与相对误差（比例，乘 100 为百分数）。
均值接近零时，相对误差可能很大甚至无定义；应优先看：

- `mean_shift_in_reference_std`：均值误差相当于多少个真实标准差。
- `original_std_over_reference_std`：旧标准差与参考标准差的比例。

JSON 同时记录环境版本、训练脚本 SHA256、准确的有序模拟 key/组 ID 和帧列表，方便复核。
旧 JSON 的参数不匹配会明确列出；即使参数一致，也不能仅靠元数据证明使用过完全相同的 HDF5 或代码。

发现数值差异不等于必须重训。应先比较真实数据与旧统计，再判断通道损失权重、模型排序或课程报告结论是否受影响。对已有 checkpoint，继续使用训练时保存的统计；**不要直接替换为新统计再评估旧权重**。

## 当前状态

创建脚本时，本地尚未提供真实 HDF5 或 `train_stats.json`。因此目前只能验证脚本和分块算法，不能报告真实 h/u/v 误差，也不能据此决定重训。

退出码：0=成功；2=未找到数据；1=数据/计算/一致性验证失败。
