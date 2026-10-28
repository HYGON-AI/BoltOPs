#!/bin/bash -x
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
cd "$(dirname "$0")" || exit 1
boltops-fused-moe-tune-cli nmz bf16 --ffn-hidden-size 4096 --hidden-size-default 352 --num-experts 128 --tp-list 1 --benchmark
