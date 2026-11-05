# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Stream-ordering guarantees of the shared graph-timing helper."""


def test_graph_timing_syncs_input_stream_first(monkeypatch):
    """The graph bench tool runs on its own side stream, so the caller's
    stream must be synchronized before any timed kernel starts."""
    import torch
    import triton

    from boltops.utility import bench as bench_module

    calls = []

    class _Stream:
        def synchronize(self):
            calls.append("sync")

    def fake_cudagraph_bench(run, rep=None, return_mode=None):
        calls.append("bench")
        return 1.25

    monkeypatch.setattr(triton.testing, "do_bench_cudagraph", fake_cudagraph_bench)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())

    ms, method = bench_module.time_ms(lambda: None, use_graph=True)

    assert (ms, method) == (1.25, "cudagraph")
    assert calls == ["sync", "bench"]
