# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Portable reduction-contract tests; no Torch/Triton import or GPU required.

The real kernel's constexpr admission expression is executed. Independent CPU
integer/FP32 arithmetic models pin its bounds and rounding contract; they do
not emulate a Triton launch or replace device numerical/Graph qualification.
"""
import ast
import functools
import math
import random
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INT32_MAX = (1 << 31) - 1
SAFE_K_MAX = INT32_MAX // 1024


@functools.lru_cache(None)
def _guard_expression():
    source = ROOT / "boltops/fused_moe/triton/moe_op.py"
    function = next(node for node in ast.parse(source.read_bytes()).body
                    if isinstance(node, ast.FunctionDef) and node.name ==
                    "fused_moe_kernel_gptq_awq_w4a8_channelwise")
    guard = next((node for node in function.body
                  if isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Name)
                  and node.target.id == "safe_int32_acc"), None)
    if guard is None:
        raise AssertionError("Kernel has no guarded whole-K integer reduction")
    return compile(ast.Expression(guard.value), str(source), "eval")


def _integer_admitted(k):
    return eval(_guard_expression(), {"__builtins__": {}, "K": k})


def _dot_int64(a, b):
    if len(a) != len(b):
        raise ValueError("Dot inputs must have equal logical K")
    if any(not -128 <= value <= 127 for value in a):
        raise ValueError("Activation outside signed INT8")
    if any(not -8 <= value <= 7 for value in b):
        raise ValueError("Weight outside signed INT4")
    # Python sums exactly; explicit range check makes the oracle INT64-bounded.
    value = sum(int(x) * int(y) for x, y in zip(a, b))
    if not -(1 << 63) <= value < (1 << 63):
        raise OverflowError("INT64 oracle overflow")
    return value


def _tiled_int32(a, b, bk):
    accumulator = 0
    for start in range(0, len(a), bk):
        accumulator += _dot_int64(a[start:start + bk], b[start:start + bk])
        if not -(1 << 31) <= accumulator <= INT32_MAX:
            raise OverflowError("INT32 partial sum overflow")
    return accumulator


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _scale(value, token_scale, channel_scale):
    return _f32(_f32(_f32(value) * _f32(token_scale)) * _f32(channel_scale))


def _old_block_scaled(a, b, bk, token_scale, channel_scale):
    accumulator = _f32(0.0)
    for start in range(0, len(a), bk):
        partial = _dot_int64(a[start:start + bk], b[start:start + bk])
        accumulator = _f32(accumulator + _scale(partial, token_scale,
                                               channel_scale))
    return accumulator


class W4A8ChannelwiseAccumulatorTest(unittest.TestCase):
    def test_actual_guard_preserves_empty_and_unsafe_fallback(self):
        for k in (0, -1, SAFE_K_MAX + 1, 1 << 32):
            self.assertFalse(_integer_admitted(k), k)
        for k in (1, 130, 384, 1536, SAFE_K_MAX):
            self.assertTrue(_integer_admitted(k), k)

    def test_full_k_bound_covers_every_prefix(self):
        # Every signed INT8 x signed INT4 product has absolute value<=1024.
        # The same bound applies to every prefix, even with cancellation.
        self.assertEqual(SAFE_K_MAX, 2097151)
        self.assertTrue(_integer_admitted(SAFE_K_MAX))
        self.assertLessEqual(1024 * SAFE_K_MAX, INT32_MAX)
        self.assertGreaterEqual(-1024 * SAFE_K_MAX, -(1 << 31))
        self.assertFalse(_integer_admitted(SAFE_K_MAX + 1))
        self.assertGreater(1024 * (SAFE_K_MAX + 1), INT32_MAX)

    def test_empty_k_does_not_apply_unused_nonfinite_scales(self):
        # Empty K is a low-level fallback boundary, not a legal Flash shape.
        # The old empty reduction is +0; unused scales must remain unused.
        for scale in (float("nan"), float("inf"), -float("inf")):
            value = (_scale(0, scale, 1.0) if _integer_admitted(0)
                     else _old_block_scaled([], [], 32, scale, 1.0))
            self.assertFalse(math.isnan(value))
            self.assertEqual(struct.pack("f", value), struct.pack("f", 0.0))

    def test_extrema_negative_values_and_exact_cancellation(self):
        cases = [([-128] * 130, [-8] * 130),
                 ([127] * 130, [-8] * 130),
                 ([-128] * 127 + [127] * 128, [-8] * 255)]
        self.assertEqual(_dot_int64(*cases[-1]), 0)
        for a, b in cases:
            for bk in (32, 64, 128):
                self.assertEqual(_tiled_int32(a, b, bk), _dot_int64(a, b))

    def test_random_reductions_match_independent_int64_oracle(self):
        rng = random.Random(0)
        for k in (1, 31, 32, 33, 65, 130, 384, 1536):
            a = [rng.randrange(-128, 128) for _ in range(k)]
            b = [rng.randrange(-8, 8) for _ in range(k)]
            for bk in (32, 64, 128):
                self.assertEqual(_tiled_int32(a, b, bk), _dot_int64(a, b))

    def test_packed_tail_and_unused_nibble_are_masked_by_zero_activation(self):
        for k in (33, 65, 130, 131):
            a = [-128 if i % 2 else 127 for i in range(k)]
            b = [(i % 16) - 8 for i in range(k)]
            packed = [((b[i] & 15) << 4)
                      | ((b[i + 1] & 15) if i + 1 < k else 7)
                      for i in range(0, k, 2)]  # even K in high nibble
            decoded = [(byte >> shift) & 15 for byte in packed
                       for shift in (4, 0)]
            decoded = [value - 16 if value >= 8 else value
                       for value in decoded]
            padded_k = ((k + 31) // 32) * 32
            masked_a = a + [0] * (padded_k - k)
            padded_b = decoded + [0] * (padded_k - len(decoded))
            self.assertEqual(_dot_int64(masked_a, padded_b), _dot_int64(a, b))

    def test_finite_rounding_does_not_assume_old_block_order_is_exact(self):
        rng = random.Random(42)
        differences = 0
        for _ in range(8):
            a = [rng.randrange(-128, 128) for _ in range(1536)]
            b = [rng.randrange(-8, 8) for _ in range(1536)]
            new = _scale(_tiled_int32(a, b, 64), 0.1, 0.07)
            oracle = _scale(_dot_int64(a, b), 0.1, 0.07)
            self.assertEqual(new, oracle)
            old = _old_block_scaled(a, b, 64, 0.1, 0.07)
            differences += struct.pack("f", new) != struct.pack("f", old)
        self.assertGreater(differences, 0)

    def test_zero_activation_has_exact_zero_dot_for_each_blocking(self):
        # Masked-load arithmetic; actual device store masking is GPU coverage.
        for bk in (32, 64, 128):
            dot = _tiled_int32([0] * 130, [-8] * 130, bk)
            self.assertEqual(dot, _dot_int64([0] * 130, [-8] * 130))
            self.assertEqual(_scale(dot, 0.0, 0.25), 0.0)

    def test_oracle_rejects_values_outside_quantized_contract(self):
        with self.assertRaises(ValueError):
            _dot_int64([128], [0])
        with self.assertRaises(ValueError):
            _dot_int64([0], [8])


if __name__ == "__main__":
    unittest.main()
