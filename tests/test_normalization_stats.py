"""CPU regression tests without importing torch or running the training entry point."""

import ast
from pathlib import Path
import unittest

import numpy as np


source = Path(__file__).resolve().parents[1] / "pdebench/train_unet_v2.py"
tree = ast.parse(source.read_text(encoding="utf-8-sig"))
helper = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "compute_normalization_stats")
namespace = {"np": np}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
compute_stats = namespace["compute_normalization_stats"]


class NormalizationStatsTests(unittest.TestCase):
    def test_matches_float64_for_multiple_channels_and_chunk_boundaries(self):
        rng = np.random.default_rng(4471)
        for channels in (1, 3):
            data = rng.normal(size=(3, 13, 17, channels)).astype(np.float32)
            flat = data.reshape(-1, channels)
            for chunk_rows in (1, 37, 65536):
                with self.subTest(channels=channels, chunk_rows=chunk_rows):
                    mean, std = compute_stats(data, chunk_rows)
                    np.testing.assert_allclose(mean, flat.mean(0, dtype=np.float64), rtol=1e-12, atol=1e-13)
                    np.testing.assert_allclose(std, flat.std(0, dtype=np.float64), rtol=1e-12, atol=1e-13)
                    self.assertEqual(mean.dtype, np.dtype(np.float64))
                    self.assertEqual(std.dtype, np.dtype(np.float64))

    def test_large_float32_reduction_regression(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(131073, 3)).astype(np.float32)
        data[:, 0] += 2
        mean, std = compute_stats(data, 4096)
        np.testing.assert_allclose(mean, data.mean(0, dtype=np.float64), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(std, data.std(0, dtype=np.float64), rtol=1e-12, atol=1e-13)
        self.assertGreater(np.max(np.abs(data.std(0) - std)), 1e-7)

    def test_large_offset_small_variance_and_input_unchanged(self):
        rng = np.random.default_rng(1)
        data = (1e8 + rng.normal(size=(4099, 3))).astype(np.float64)
        before = data.copy()
        mean, std = compute_stats(data, 113)
        np.testing.assert_allclose(mean, data.mean(0), rtol=1e-14)
        np.testing.assert_allclose(std, data.std(0), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(data, before)

    def test_constant_fields_remain_zero_variance(self):
        mean, std = compute_stats(np.full((19, 3), 2, dtype=np.float32), 7)
        np.testing.assert_array_equal(mean, [2, 2, 2])
        np.testing.assert_array_equal(std, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
