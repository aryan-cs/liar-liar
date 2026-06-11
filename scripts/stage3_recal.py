"""Stage 3 (recalibrated): paraphrase OOD + lens trajectories for both families.

Reuses the cached paraphrases from the earlier run if present.
Writes results/recal/{family}/para_{cond}.jsonl and results/recal/lens.pt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from liar.eval import QA_TEMPLATE, evaluate_truthfulqa_mc, generate_with_steering  # noqa: E402
from liar.lens import t_readout_trajectory  # noqa: E402
from liar.model import load_model  # noqa: E402
from liar.progress import count_jsonl, write_progress  # noqa: E402

DATA = ROOT / "data"
RC = ROOT / "artifacts" / "recal"
OUT = ROOT / "results" / "recal"
N_PARA = 150
N_LENS = 100
PARA_CONDS = ["v_dec", "v_perp_al64"]
LENS_CONDS = ["v_dec", "v_perp_al64", "v_par_al64"]


def main() -> None:
    cfg = json.loads((RC / "config.json").read_text())
    vectors = torch.load(RC / "vectors.pt", weights_only=True)
    tokensets = json.loads((RC / "tokensets.json").read_text())
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    test_idx = tqa["splits"]["test"]
    rows = [tqa["rows"][i] for i in test_idx]
    families = list(cfg["families"].keys())

    OUT.mkdir(parents=True, exist_ok=True)
    write_progress(OUT, "running", 0, 3, {"step": "load"})
    lm = load_model(cfg["model_id"])
    tok = lm.tokenizer

    # paraphrase cache (reuse earlier stage3 if present)
    para_path = DATA / "paraphrases.json"
    para_src = rows[:N_PARA]
    if not para_path.exists():
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content":
                  "Rewrite the following question in different words, keeping its "
                  "meaning exactly the same. Output only the rewritten question.\n\n"
                  + r["question"]}],
                tokenize=False, add_generation_prompt=True)
            for r in para_src
        ]
        gens = generate_with_steering(lm, prompts, max_new_tokens=80)
        paras = [g.strip().split("\n")[0].strip() for g in gens]
        paras = [p if 10 <= len(p) <= 400 else r["question"] for p, r in zip(paras, para_src)]
        para_path.write_text(json.dumps(paras))
    paras = json.loads(para_path.read_text())
    para_rows = [{**r, "question": p} for r, p in zip(para_src, paras)]
    write_progress(OUT, "running", 1, 3, {"step": "paraphrase"})

    # paraphrased baseline (shared)
    bpath = OUT / "para_baseline.jsonl"
    if count_jsonl(bpath) < len(para_rows):
        res = evaluate_truthfulqa_mc(lm, para_rows)
        with open(bpath, "w") as f:
            for qi, r in zip(test_idx[:N_PARA], res):
                f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"]}) + "\n")
        print(f"[stage3r] para baseline mc2={sum(r['mc2'] for r in res)/len(res):.4f}", flush=True)

    for fam in families:
        op = cfg["families"][fam]
        fdir = OUT / fam
        fdir.mkdir(parents=True, exist_ok=True)
        for cond in PARA_CONDS:
            path = fdir / f"para_{cond}.jsonl"
            if count_jsonl(path) >= len(para_rows):
                continue
            res = evaluate_truthfulqa_mc(lm, para_rows, layer=op["layer"],
                                         vector=vectors[f"{fam}/{cond}"], coefficient=op["alpha"])
            with open(path, "w") as f:
                for qi, r in zip(test_idx[:N_PARA], res):
                    f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"]}) + "\n")
            print(f"[stage3r] {fam}/para_{cond} mc2={sum(r['mc2'] for r in res)/len(res):.4f}", flush=True)
    write_progress(OUT, "running", 2, 3, {"step": "lens"})

    # lens trajectories on curated T readout, per family
    lens_path = OUT / "lens.pt"
    if not lens_path.exists():
        plus = list(tokensets["curated_plus"].values())
        minus = list(tokensets["curated_minus"].values())
        prompts = [QA_TEMPLATE.format(question=r["question"]) for r in rows[:N_LENS]]
        out = {"families": {}, "layer_star": {}}
        base = t_readout_trajectory(lm, prompts, plus, minus)
        out["baseline"] = base
        for fam in families:
            op = cfg["families"][fam]
            out["layer_star"][fam] = op["layer"]
            out["families"][fam] = {}
            for cond in LENS_CONDS:
                out["families"][fam][cond] = t_readout_trajectory(
                    lm, prompts, plus, minus, layer=op["layer"],
                    vector=vectors[f"{fam}/{cond}"], coefficient=op["alpha"])
            print(f"[stage3r] lens {fam}: done", flush=True)
        torch.save(out, lens_path)

    write_progress(OUT, "done", 3, 3, {})
    (OUT / "DONE").touch()
    print("[stage3r] complete", flush=True)


if __name__ == "__main__":
    main()
