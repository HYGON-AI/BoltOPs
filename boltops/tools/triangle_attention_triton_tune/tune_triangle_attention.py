"""Offline tuner and benchmark for the TriangleAttention forward kernel."""
import argparse
from contextlib import contextmanager
import importlib
import itertools
import json
from pathlib import Path
import statistics

import torch
import triton.testing
from triton.runtime.errors import HSACOError, OutOfResources
from triton.compiler.errors import CompilationError
import yaml

from boltops.fa import triangle_attention_fwd
from boltops.fa.triangle_attention.triton import config
from boltops.utility.arch_info import get_device_info
from boltops.utility.triton_capability import get_triton_capabilities, get_triton_config_dir

CASE_FILE = Path(__file__).with_name('benchmark_cases.yaml')


def torch_production_attention(q, k, v, mask_bias, triangle_bias):
    """Match Protenix's BF16 attention expression for correctness checks."""
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores += mask_bias
    scores += triangle_bias
    with torch.amp.autocast("cuda", enabled=False):
        probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def make_inputs(batch, heads, n, d, density, mask_layout="contiguous", *,
                bias_dtype=torch.bfloat16, mask_dtype=torch.bfloat16):
    """Build the production Q/K/V view and additive bias layouts."""
    q = torch.randn((batch, n, heads, d), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn((batch, n, heads, d), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn((batch, n, heads, d), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    q = q / (d ** 0.5)
    if mask_layout == "contiguous":
        mask = torch.rand((batch, 1, 1, n), device="cuda") < density
        mask_bias = torch.zeros(mask.shape, device="cuda", dtype=mask_dtype).masked_fill_(~mask, -1.0e9)
    elif mask_layout == "strided":
        base = torch.rand((n, batch), device="cuda") < density
        mask = torch.as_strided(
            base.t(), (batch, 1, 1, n), (1, batch * n, batch * n, batch)
        )
        base_bias = torch.zeros(base.shape, device="cuda", dtype=mask_dtype).masked_fill_(~base, -1.0e9)
        mask_bias = torch.as_strided(
            base_bias.t(), (batch, 1, 1, n), (1, batch * n, batch * n, batch)
        )
    else:
        raise ValueError(f"unknown mask_layout: {mask_layout}")
    mask[..., 0] = True
    mask_bias[..., 0] = 0
    triangle_bias = torch.randn((n, n, heads), device="cuda", dtype=bias_dtype).permute(2, 0, 1).unsqueeze(0)
    return q, k, v, mask, mask_bias, triangle_bias


def load_cases(path=CASE_FILE):
    return [dict(c, mask_layout=c.get('mask_layout', 'contiguous'))
            for c in yaml.safe_load(Path(path).read_text())['cases']
            if c['enabled'] and c.get('performance', False)]


def candidates(quick=False):
    if quick:
        return [dict(config.DEFAULT_CONFIG)]
    return [dict(BLOCK_M=m, BLOCK_N=n, num_warps=w, num_stages=s)
            for m, n, w, s in itertools.product(
                (16, 32, 64, 128), (16, 32, 64, 128, 256),
                (1, 2, 4, 8), (1, 2, 3, 4)
            )]


@contextmanager
def override(candidate):
    backend = importlib.import_module('boltops.fa.triangle_attention.triton.triangle_attention')
    original = backend._get_triangle_config
    def selected(*args, **kwargs):
        # Preserve device-specific launch options such as gfx936 kpack=2.
        return dict(original(*args, **kwargs), **candidate)
    backend._get_triangle_config = selected
    try:
        yield
    finally:
        backend._get_triangle_config = original


def check(actual, expected):
    assert torch.isfinite(actual).all(), 'non-finite output'
    torch.testing.assert_close(actual.float(), expected.float(), atol=.03, rtol=.02)


def measure(fn, args):
    return float(triton.testing.do_bench(fn, warmup=args.warmup_ms,
                                       rep=args.rep_ms, return_mode='median'))


def evaluate_candidate(candidate, run, expected, args):
    record = {'config': candidate}
    try:
        with override(candidate):
            if not args.skip_correctness:
                check(run(), expected)
            record['ms'] = measure(run, args)
        record['status'] = 'timed' if args.skip_correctness else 'pass'
    except (AssertionError, CompilationError, HSACOError, OutOfResources) as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    except RuntimeError as exc:
        if 'PassManager::run failed' not in str(exc):
            raise
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    return record


def write_config(case, candidate, root):
    info = get_device_info()
    filename = config.make_config_filename(device_label=f'{info.arch}_cu{info.cu_count}')
    path = get_triton_config_dir(root, get_triton_capabilities()) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'key': list(config.CONFIG_KEY_FIELDS), 'config': {}}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload['key'] != list(config.CONFIG_KEY_FIELDS):
            raise ValueError(f'Incompatible config: {path}')
    key = config.make_config_key(**{k: case[k] for k in config.CONFIG_KEY_FIELDS})
    payload['config'][str(key)] = candidate
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--case-file', type=Path, default=CASE_FILE)
    p.add_argument('--case', action='append')
    p.add_argument('--list-cases', action='store_true')
    p.add_argument('--benchmark-only', action='store_true')
    p.add_argument('--quick', action='store_true', help='Smoke test default config only')
    p.add_argument(
        '--skip-correctness', action='store_true',
        help='Skip candidate/output correctness checks; timing only',
    )
    p.add_argument('--output-root', type=Path, default=config.CONFIG_ROOT)
    p.add_argument('--report', type=Path, default=Path('/tmp/triangle_attention_report.json'))
    p.add_argument('--warmup-ms', type=int, default=25)
    p.add_argument('--rep-ms', type=int, default=100)
    p.add_argument('--rounds', type=int, default=5)
    a = p.parse_args(argv)
    cases = load_cases(a.case_file)
    if a.list_cases:
        print('\n'.join(c['name'] for c in cases))
        return 0
    if a.rounds < 1 or a.rep_ms <= 0 or a.warmup_ms < 0:
        p.error('rounds/rep-ms must be positive; warmup-ms must be nonnegative')
    if a.case:
        if set(a.case) - {c['name'] for c in cases}:
            p.error('Unknown or non-performance case')
        cases = [c for c in cases if c['name'] in a.case]
    report = {'torch': torch.__version__, 'triton': triton.__version__,
              'device': torch.cuda.get_device_name(), 'atol': .03, 'rtol': .02,
              'timing': 'GPU event median; full interface including direct additive mask loads and output allocation',
              'cases': []}
    with torch.inference_mode():
        for case_index, c in enumerate(cases, 1):
            torch.manual_seed(0)
            q, k, v, mask, mb, bias = make_inputs(
                c['batch'], c['heads'], c['seq_len'], c['head_dim'],
                c['mask_density'], c.get('mask_layout', 'contiguous')
            )
            ref_fn = lambda: torch_production_attention(q, k, v, mb, bias)
            run = lambda: triangle_attention_fwd(q, k, v, [mb, bias])
            expected = ref_fn()
            records = []
            if not a.benchmark_only:
                search_space = candidates(a.quick)
                for candidate_index, candidate in enumerate(search_space, 1):
                    record = evaluate_candidate(candidate, run, expected, a)
                    records.append(record)
                    print(
                        f"[{c['name']} {candidate_index}/{len(search_space)}] "
                        f"M/N={candidate['BLOCK_M']}/{candidate['BLOCK_N']} "
                        f"warps={candidate['num_warps']} stages={candidate['num_stages']} "
                        f"status={record['status']}"
                        + (f" ms={record['ms']:.3f}" if 'ms' in record else "")
                        + (f" error={record['error']}" if 'error' in record else ""),
                        flush=True,
                    )
                valid_status = {'timed'} if a.skip_correctness else {'pass'}
                valid = [r for r in records if r['status'] in valid_status]
                if not valid:
                    raise RuntimeError('No correct candidate; no config written')
                best = min(valid, key=lambda r: r['ms'])['config']
                path = write_config(c, best, a.output_root)
                config.CONFIG_ROOT = a.output_root
                config.clear_config_cache()
                loaded = config.get_kernel_config(
                    **{key: c[key] for key in config.CONFIG_KEY_FIELDS}, device=q.device,
                )
                assert loaded == best, (loaded, best)
                print('runtime config verified:', path, flush=True)
            if not a.skip_correctness:
                check(run(), expected)
            times = {'torch_ms': [], 'triton_ms': []}
            for i in range(a.rounds):
                order = [('torch_ms', ref_fn), ('triton_ms', run)]
                for label, fn in (order if i % 2 == 0 else order[::-1]):
                    times[label].append(measure(fn, a))
            torch_median = statistics.median(times['torch_ms'])
            triton_median = statistics.median(times['triton_ms'])
            result = dict(case=c, correctness_skipped=a.skip_correctness,
                          candidates=records,
                          torch_median_ms=torch_median,
                          triton_median_ms=triton_median,
                          active_config=config.get_kernel_config(
                              **{key: c[key] for key in config.CONFIG_KEY_FIELDS}, device=q.device,
                          ),
                          strides={'q': q.stride(), 'bias': bias.stride(), 'mask_bias': mb.stride()})
            result['speedup'] = torch_median / triton_median
            report['cases'].append(result)
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(report, indent=2) + '\n')
            print(
                f"[{case_index}/{len(cases)}] {c['name']} "
                f"torch_median_ms={torch_median:.3f} "
                f"triton_median_ms={triton_median:.3f} "
                f"speedup={result['speedup']:.2f}x",
                flush=True,
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
