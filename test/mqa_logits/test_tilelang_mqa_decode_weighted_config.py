# SPDX-License-Identifier: MIT
"""Independent weighted policy starts with the full gfx938 QSA table."""

import pytest

from boltops.mqa_logits.tilelang.configs.mqa_decode import gfx938, weighted


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_weighted_decode_policy_matches_qsa(fp8):
    original = gfx938.get_config_fp8 if fp8 else gfx938.get_config
    getter = weighted.get_config_fp8 if fp8 else weighted.get_config
    assert getter is not original
    for batch in (1, 2, 3, 8, 9, 16, 17, 32, 33, 64, 65, 128):
        for capacity in (1, 95, 8192, 8193, 16384, 16385, 32768, 32769, 65536, 131072):
            for heads in (4, 8, 16, 32, 64):
                assert getter(batch, capacity, 72, heads) == original(batch, capacity, 72, heads)


@pytest.mark.parametrize("args", [(0, 8192, 72, 4), (16, 0, 72, 4),
                                  (16, 8192, 0, 4), (16, 8192, 72, 0)])
def test_weighted_decode_policy_rejects_invalid_arguments(args):
    for getter in (weighted.get_config, weighted.get_config_fp8):
        with pytest.raises(ValueError, match="weighted MQA decode requires"):
            getter(*args)
