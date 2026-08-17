"""Torch dtype aliases used by migrated kernels."""

import torch

fp16 = torch.float16
bf16 = torch.bfloat16
fp32 = torch.float32
i32 = torch.int32
i16 = torch.int16
i8 = torch.int8
u8 = torch.uint8
i64 = torch.int64
u64 = torch.uint64
fp8 = getattr(torch, "float8_e4m3fn", torch.uint8)

__all__ = ["fp16", "bf16", "fp32", "i32", "i16", "i8", "u8", "i64", "u64", "fp8"]

