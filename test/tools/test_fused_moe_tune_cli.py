from __future__ import annotations

import pytest

from boltops.tools.fused_moe_triton_tune.tune_moe_cli import (
    is_moe_process_command,
)


@pytest.mark.parametrize(
    "command",
    [
        "python3 -m boltops.tools.fused_moe_triton_tune.tune_moe_cli nmz bf16",
        "/usr/bin/python3 -u -m boltops.tools.fused_moe_triton_tune.tune_moe_cli bmz int8",
        "python3 /zhenxin/BoltOPs/boltops/tools/fused_moe_triton_tune/tune_moe_cli.py nmz fp8",
        "boltops-fused-moe-tune-cli nmz fp8_channel",
        "/usr/bin/python3 /usr/local/bin/boltops-fused-moe-tune-cli bmz bf16",
    ],
)
def test_matches_cli_entry_points(command: str) -> None:
    assert is_moe_process_command(command)


def test_does_not_match_parent_shell_containing_module_command() -> None:
    command = (
        "bash -lc python3 -m "
        "boltops.tools.fused_moe_triton_tune.tune_moe_cli nmz bf16"
    )
    assert not is_moe_process_command(command)


def test_does_not_match_unrelated_python_process() -> None:
    assert not is_moe_process_command("python3 -m pytest test/test_package_smoke.py")
