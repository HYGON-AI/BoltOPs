"""Original shapes from shape_09_kimi26_128k_s2000.xlsx.

49 cases cover 24 network shapes; 13 cases form the default representative set.
Keep complete scalar dictionaries so pytest parameters also describe CSV rows.
"""


def _prefill_case(index, path, **overrides):
    sq, sk = (6656, 6656 * (index + 1)) if index < 19 else (4608, 131072)
    case = dict(
        case_id=f"{path.upper()}-S{index:02d}", path=path,
        B=4, H=64, Sq=sq, Sk=sk, Dqk=192, Dv=128,
        source_op=f"mha_slice{index}", sheet="prefill_shape",
        source_rows=f"{16 + index} / {53 + index}",
        q_dtype="BF16", k_dtype="BF16", v_dtype="BF16",
        sm_scale=0.14467962580268923, k_scale=1.0, v_scale=1.0,
        page_size=0, rank=-1, layout="contiguous",
        representative=index in (0, 9, 19), seed=sk,
    )
    case.update(overrides)
    return case


CASES = []
for index in range(20):
    CASES.append(_prefill_case(index, "dense"))
    if index == 9:
        CASES.append(_prefill_case(index, "dense", case_id="DENSE-S09-STRIDE", layout="strided"))

for page in (0, 131072, 64, 16):
    CASES.append(dict(
        case_id=f"DECODE-P{page}" if page else "DECODE-LOGICAL", path="decode",
        B=4, H=64, Sq=1, Sk=131072, Dqk=128, Dv=128,
        source_op="eagle3_paged_attn", sheet="decode_shape", source_rows="66",
        q_dtype="BF16", k_dtype="FP8E4M3FN", v_dtype="FP8E4M3FN",
        sm_scale=0.08838834764831845, k_scale=0.37, v_scale=1.7, page_size=page, rank=-1,
        **({"page_mapping": "reverse_requests"} if page == 131072 else {}),
        layout="shuffled_pages" if page else "logical",
        representative=page != 131072, seed=20261021,
    ))

for index, (sq, sk) in enumerate(((6656, 53248), (6656, 106496), (3072, 131072))):
    for rank in range(8):
        CASES.append(_prefill_case(
            index, "segmented", case_id=f"CP8-S{index}-R{rank}",
            B=32, Sq=sq, Sk=sk, sheet="prefill_shape_cp8",
            source_rows=f"{16 + index} / {36 + index}", rank=rank,
            layout="independent_B32", representative=rank in (0, 7),
            seed=20261023 + 100 * index + rank,
        ))
