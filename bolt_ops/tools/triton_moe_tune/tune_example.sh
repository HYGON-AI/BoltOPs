#!/bin/bash -x
cd "$(dirname "$0")" || exit 1
bolt-moe-tune-cli nmz bf16 --ffn-hidden-size 4096 --hidden-size-default 352 --num-experts 128 --tp-list 1 --benchmark
