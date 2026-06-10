"""Stage 3: OOD paraphrase probe and layer-wise lens trajectories.

Artifacts:
  data/paraphrases.json          cached model-generated paraphrases
  results/stage3/{cond}.jsonl    MC results on paraphrased questions
  results/stage3/lens.pt         (condition, layer, prompt) honest-shift tensor
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
S1 = ROOT / "artifacts" / "stage1"
OUT = ROOT / "results" / "stage3"

N_PARA = 150
N_LENS = 100
LENS_CONDS = ["baseline", "v_dec", "v_perp_al64", "v_par_al64"]
PARA_CONDS = ["baseline", "v_dec", "v_perp_al64"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((S1 / "config.json").read_text())
    vectors = torch.load(S1 / "vectors.pt", weights_only=True)
    tokensets = json.loads((S1 / "tokensets.json").read_text())
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    test_idx = tqa["splits"]["test"]
    rows = [tqa["rows"][i] for i in test_idx]

    total = len(PARA_CONDS) + 1 + 1
    write_progress(OUT, "running", 0, total, {"step": "load"})
    lm = load_model(cfg["model_id"])

    # --- paraphrase cache ---
    para_path = DATA / "paraphrases.json"
    para_rows_src = rows[:N_PARA]
    if not para_path.exists():
        print("[stage3] generating paraphrases", flush=True)
        tok = lm.tokenizer
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content":
                  "Rewrite the following question in different words, keeping its "
                  "meaning exactly the same. Output only the rewritten question.\n\n"
                  + r["question"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in para_rows_src
        ]
        gens = generate_with_steering(lm, prompts, max_new_tokens=80)
        paras = [g.strip().split("\n")[0].strip() for g in gens]
        # Fall back to the original question if the rewrite is degenerate.
        paras = [
            p if 10 <= len(p) <= 400 else r["question"]
            for p, r in zip(paras, para_rows_src)
        ]
        para_path.write_text(json.dumps(paras))
    paras = json.loads(para_path.read_text())
    para_rows = []
    for r, p in zip(para_rows_src, paras):
        nr = dict(r)
        nr = {**nr, "question": p}
        para_rows.append(nr)
    write_progress(OUT, "running", 1, total, {"step": "paraphrase_mc"})

    # --- paraphrased MC under three conditions ---
    for ci, name in enumerate(PARA_CONDS):
        path = OUT / f"para_{name}.jsonl"
        if count_jsonl(path) >= len(para_rows):
            print(f"[stage3] para_{name}: complete, skipping", flush=True)
            continue
        vec = None if name == "baseline" else vectors[name]
        alpha = 0.0 if name == "baseline" else cfg["alpha_star"]
        res = evaluate_truthfulqa_mc(
            lm, para_rows, layer=cfg["layer_star"], vector=vec, coefficient=alpha
        )
        with open(path, "w") as f:
            for qi, r in zip(test_idx[:N_PARA], res):
                f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"]}) + "\n")
        mc2 = sum(r["mc2"] for r in res) / len(res)
        print(f"[stage3] para_{name}: mc2={mc2:.4f}", flush=True)
        write_progress(OUT, "running", 1 + ci + 1, total, {"step": f"para_{name}"})

    # --- lens trajectories on curated T readout ---
    lens_path = OUT / "lens.pt"
    if not lens_path.exists():
        plus = list(tokensets["curated_plus"].values())
        minus = list(tokensets["curated_minus"].values())
        prompts = [QA_TEMPLATE.format(question=r["question"]) for r in rows[:N_LENS]]
        trajs = {}
        for name in LENS_CONDS:
            vec = None if name == "baseline" else vectors[name]
            alpha = 0.0 if name == "baseline" else cfg["alpha_star"]
            trajs[name] = t_readout_trajectory(
                lm, prompts, plus, minus,
                layer=cfg["layer_star"], vector=vec, coefficient=alpha,
            )
            print(f"[stage3] lens {name}: done", flush=True)
        torch.save({"conditions": LENS_CONDS, "trajectories": trajs,
                    "layer_star": cfg["layer_star"]}, lens_path)

    write_progress(OUT, "done", total, total, {})
    (OUT / "DONE").touch()
    print("[stage3] complete", flush=True)


if __name__ == "__main__":
    main()
