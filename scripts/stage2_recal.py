"""Stage 2 (recalibrated): headline condition matrix for both vector families
on the held-out TruthfulQA test slice.

Writes results/recal/{family}/{condition}.jsonl. Resume-safe by row count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from liar.eval import evaluate_truthfulqa_mc  # noqa: E402
from liar.model import load_model  # noqa: E402
from liar.progress import count_jsonl, write_progress  # noqa: E402

DATA = ROOT / "data"
RC = ROOT / "artifacts" / "recal"
OUT = ROOT / "results" / "recal"

CONDS = [
    "v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256", "v_perp_al1024",
    "v_perp_cur", "v_perp_stat", "v_par_al64", "v_perp_al64_nm",
    "v_rand_s0", "v_rand_s1", "v_rand_s2",
]


def main() -> None:
    cfg = json.loads((RC / "config.json").read_text())
    capture = json.loads((RC / "capture_ids.json").read_text())
    vectors = torch.load(RC / "vectors.pt", weights_only=True)
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    test_idx = tqa["splits"]["test"]
    rows = [tqa["rows"][i] for i in test_idx]

    families = list(cfg["families"].keys())
    total = 1 + len(families) * len(CONDS)  # +1 shared baseline
    lm = load_model(cfg["model_id"])
    done = 0
    write_progress(OUT, "running", done, total, {})

    # shared baseline (no intervention) -- same for both families
    base_path = OUT / "baseline.jsonl"
    if count_jsonl(base_path) < len(rows):
        base_path.parent.mkdir(parents=True, exist_ok=True)
        res = evaluate_truthfulqa_mc(lm, rows, capture_ids=capture)
        with open(base_path, "w") as f:
            for qi, r in zip(test_idx, res):
                f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"],
                                    "eta_logits": r.get("eta_logits")}) + "\n")
        mc2 = sum(r["mc2"] for r in res) / len(res)
        mc1 = sum(r["mc1"] for r in res) / len(res)
        print(f"[stage2r] baseline: mc1={mc1:.4f} mc2={mc2:.4f}", flush=True)
    done += 1
    write_progress(OUT, "running", done, total, {"condition": "baseline"})

    for fam in families:
        op = cfg["families"][fam]
        fdir = OUT / fam
        fdir.mkdir(parents=True, exist_ok=True)
        for cond in CONDS:
            path = fdir / f"{cond}.jsonl"
            if count_jsonl(path) >= len(rows):
                print(f"[stage2r] {fam}/{cond}: complete, skipping", flush=True)
                done += 1
                write_progress(OUT, "running", done, total, {"condition": f"{fam}/{cond}"})
                continue
            vec = vectors[f"{fam}/{cond}"]
            res = evaluate_truthfulqa_mc(
                lm, rows, layer=op["layer"], vector=vec, coefficient=op["alpha"],
                capture_ids=capture,
            )
            with open(path, "w") as f:
                for qi, r in zip(test_idx, res):
                    f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"],
                                        "eta_logits": r.get("eta_logits")}) + "\n")
            mc2 = sum(r["mc2"] for r in res) / len(res)
            mc1 = sum(r["mc1"] for r in res) / len(res)
            print(f"[stage2r] {fam}/{cond}: mc1={mc1:.4f} mc2={mc2:.4f}", flush=True)
            done += 1
            write_progress(OUT, "running", done, total, {"condition": f"{fam}/{cond}"})

    write_progress(OUT, "done", total, total, {})
    (OUT / "DONE").touch()
    print("[stage2r] complete", flush=True)


if __name__ == "__main__":
    main()
