"""Stage 0: download model weights and datasets; write data manifests.

Idempotent: skips anything already present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "stage0"

MODEL_CANDIDATES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "NousResearch/Meta-Llama-3-8B-Instruct",
]

SEED = 0
N_VAL_SWEEP = 120
N_MM = 200


def main() -> None:
    import random

    from datasets import load_dataset

    from liar.progress import write_progress

    DATA.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_progress(OUT, "running", 0, 3, {"step": "model"})

    # --- model weights ---
    chosen = None
    marker = OUT / "model.json"
    if marker.exists():
        chosen = json.loads(marker.read_text())["model_id"]
    else:
        from huggingface_hub import snapshot_download

        for cand in MODEL_CANDIDATES:
            try:
                snapshot_download(cand, allow_patterns=["*.safetensors", "*.json", "*.model"])
                chosen = cand
                break
            except Exception as e:
                print(f"[stage0] {cand} unavailable: {e}", flush=True)
        if chosen is None:
            raise RuntimeError("no model candidate downloadable")
        marker.write_text(json.dumps({"model_id": chosen}))
    print(f"[stage0] model: {chosen}", flush=True)
    write_progress(OUT, "running", 1, 3, {"step": "truthful_qa"})

    # --- TruthfulQA MC + splits ---
    tqa_path = DATA / "truthfulqa_mc.json"
    if not tqa_path.exists():
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
        rows = [dict(r) for r in ds]
        rng = random.Random(SEED)
        order = list(range(len(rows)))
        rng.shuffle(order)
        splits = {
            "val_sweep": order[:N_VAL_SWEEP],
            "mm_source": order[N_VAL_SWEEP : N_VAL_SWEEP + N_MM],
            "test": order[N_VAL_SWEEP + N_MM :],
        }
        tqa_path.write_text(json.dumps({"rows": rows, "splits": splits}))
        print(f"[stage0] truthfulqa: {len(rows)} rows, test={len(splits['test'])}", flush=True)
    write_progress(OUT, "running", 2, 3, {"step": "alpaca"})

    # --- Alpaca instructions for contrast pairs / calibration ---
    alp_path = DATA / "alpaca_questions.json"
    if not alp_path.exists():
        ds = load_dataset("tatsu-lab/alpaca")["train"]
        qs = []
        for r in ds:
            if r["input"].strip() == "" and len(r["instruction"]) < 200:
                qs.append(r["instruction"])
            if len(qs) >= 400:
                break
        alp_path.write_text(json.dumps(qs))
        print(f"[stage0] alpaca: {len(qs)} instructions", flush=True)

    write_progress(OUT, "done", 3, 3, {})
    (OUT / "DONE").touch()


if __name__ == "__main__":
    main()
