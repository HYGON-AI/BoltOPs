import pytest
import torch

from boltops.fla.gdn.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from ._gdn_reference import naive_chunk_h


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA required")


def test_invalid_state_update_layout_is_rejected():
    keys = torch.zeros(1, 65, 1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="STATE_UPDATE_VK"):
        chunk_gated_delta_rule_fwd_h(
            keys, keys, keys,
            kernel_cfg=dict(BV=32, num_warps=4, num_stages=1, STATE_UPDATE_VK="invalid"))


@pytest.mark.parametrize("key_dim,value_dim,tokens,transpose,exp2,varlen,tile,gate_mode", [
    (128, 128, 193, True, False, False, 64, "scalar"),
    (128, 128, 193, True, True, True, 128, "scalar"),
    (128, 128, 65, False, False, False, 64, "scalar"),
    (64, 64, 65, True, False, False, 64, "scalar"),
    (256, 64, 65, True, False, False, 64, "scalar"),
    (128, 70, 65, True, False, False, 64, "none"),
    (128, 70, 65, True, False, False, 128, "key"),
    (128, 70, 65, False, True, False, 128, "scalar"),
])
def test_direct_state_update_matches_recurrence(key_dim, value_dim, tokens, transpose, exp2, varlen, tile, gate_mode):
    torch.manual_seed(20261106)
    keys = (torch.randn(1, tokens, 1, key_dim) * 0.05).bfloat16()
    weights = (torch.randn(1, tokens, 2, key_dim) * 0.05).bfloat16()
    values = torch.randn(1, tokens, 2, value_dim).bfloat16()
    gate = -torch.rand(1, tokens, 2) * 0.04
    boundaries = [0, 65, tokens] if varlen else [0, tokens]
    for begin, end in zip(boundaries, boundaries[1:]):
        for start in range(begin, end, 64):
            gate[:, start:min(start + 64, end)] = gate[:, start:min(start + 64, end)].cumsum(1)
    if exp2:
        gate *= 1.4426950408889634
    state_shape = (value_dim, key_dim) if transpose else (key_dim, value_dim)
    initial = torch.randn(len(boundaries) - 1, 2, *state_shape) * 0.01
    lengths = torch.tensor(boundaries, dtype=torch.int32) if varlen else None
    gate = gate if gate_mode == "scalar" else None
    key_gate = -torch.rand(1, tokens, 2, key_dim) * 0.04 if gate_mode == "key" else None
    options = dict(g=gate, gk=key_gate, initial_state=initial, transpose_state_layout=transpose,
                   use_exp2=exp2, cu_seqlens=lengths)
    reference = naive_chunk_h(keys, weights, values, **options)
    device_inputs = [tensor.cuda() for tensor in (keys, weights, values)]
    device_options = dict(options, g=gate.cuda() if gate is not None else None,
                          gk=key_gate.cuda() if key_gate is not None else None, initial_state=initial.cuda(),
                          cu_seqlens=lengths.cuda() if varlen else None)
    common_config = dict(BV=32, BK=tile, num_warps=4, num_stages=1)
    original = chunk_gated_delta_rule_fwd_h(*device_inputs, kernel_cfg=common_config, **device_options)
    actual = chunk_gated_delta_rule_fwd_h(
        *device_inputs, kernel_cfg=dict(common_config, STATE_UPDATE_VK=True), **device_options)
    for result, baseline, expected in zip(actual, original, reference):
        torch.testing.assert_close(result.float(), baseline.float(), atol=0.08, rtol=0.08)
        torch.testing.assert_close(result.cpu().float(), expected.float(), atol=0.08, rtol=0.08)
