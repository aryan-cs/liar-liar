"""Stage 2: headline TruthfulQA condition matrix on the held-out test slice.

Each condition writes results/stage2/{condition}.jsonl with one row per
question: {idx, mc1, mc2, eta_logits}. Resume: a condition whose row count
matches the test split is skipped.
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
S1 = ROOT / "artifacts" / "stage1"
OUT = ROOT / "results" / "stage2"


def conditions(vectors: dict, alpha: float) -> list[tuple[str, torch.Tensor | None, float]]:
    c: list[tuple[str, torch.Tensor | None, float]] = [("baseline", None, 0.0)]
    order = [
        "v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256", "v_perp_al1024",
        "v_perp_cur", "v_perp_stat", "v_par_al64", "v_perp_al64_nm",
        "v_rand_s0", "v_rand_s1", "v_rand_s2", "v_mm",
    ]
    for name in order:
        c.append((name, vectors[name], alpha))
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((S1 / "config.json").read_text())
    capture = json.loads((S1 / "capture_ids.json").read_text())
    vectors = torch.load(S1 / "vectors.pt", weights_only=True)
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    test_idx = tqa["splits"]["test"]
    rows = [tqa["rows"][i] for i in test_idx]

    conds = conditions(vectors, cfg["alpha_star"])
    write_progress(OUT, "running", 0, len(conds), {})
    lm = load_model(cfg["model_id"])

    for ci, (name, vec, alpha) in enumerate(conds):
        path = OUT / f"{name}.jsonl"
        if count_jsonl(path) >= len(rows):
            print(f"[stage2] {name}: complete, skipping", flush=True)
            write_progress(OUT, "running", ci + 1, len(conds), {"condition": name})
            continue
        print(f"[stage2] scoring condition {name} (alpha={alpha})", flush=True)
        res = evaluate_truthfulqa_mc(
            lm, rows,
            layer=cfg["layer_star"], vector=vec, coefficient=alpha,
            capture_ids=capture,
        )
        with open(path, "w") as f:
            for qi, r in zip(test_idx, res):
                f.write(json.dumps({
                    "idx": qi, "mc1": r["mc1"], "mc2": r["mc2"],
                    "eta_logits": r.get("eta_logits"),
                }) + "\n")
        mc1 = sum(r["mc1"] for r in res) / len(res)
        mc2 = sum(r["mc2"] for r in res) / len(res)
        print(f"[stage2] {name}: mc1={mc1:.4f} mc2={mc2:.4f}", flush=True)
        write_progress(OUT, "running", ci + 1, len(conds), {"condition": name})

    write_progress(OUT, "done", len(conds), len(conds), {})
    (OUT / "DONE").touch()
    print("[stage2] complete", flush=True)


if __name__ == "__main__":
    main()
