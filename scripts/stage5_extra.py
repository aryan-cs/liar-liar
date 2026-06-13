"""Stage 5: two claim-licensing experiments.

1. CAA at every coherent grid setting, evaluated on the full held-out test
   set (the paper's universal 'no gain at any coherent strength' claim was
   previously supported only at the validation-selected operating point).

2. Lens trajectory of the aligned-64 readout itself, the set whose direct
   path the projection certifiably removes, split into T+ / T- by the sign
   of the direct effect (W-tilde v). This tests the re-synthesis claim on
   the readout that was actually suppressed.

Writes results/recal/dec_grid/L{layer}_a{alpha}.jsonl and
results/recal/lens_al64.pt. Resume-safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from liar.eval import QA_TEMPLATE, evaluate_truthfulqa_mc  # noqa: E402
from liar.lens import t_readout_trajectory  # noqa: E402
from liar.model import get_final_norm, get_lm_head, load_model  # noqa: E402
from liar.progress import count_jsonl  # noqa: E402
from liar.steering import _last_token_residuals, caa_vector  # noqa: E402
from liar.unembedding import mean_jacobian  # noqa: E402

DATA = ROOT / "data"
RC = ROOT / "artifacts" / "recal"
OUT = ROOT / "results" / "recal"
N_CALIB = 256
N_LENS = 100
LENS_CONDS = ["v_dec", "v_perp_al64", "v_par_al64"]


def main() -> None:
    cfg = json.loads((RC / "config.json").read_text())
    calib = json.loads((RC / "calibration.json").read_text())
    vectors = torch.load(RC / "vectors.pt", weights_only=True)
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    test_idx = tqa["splits"]["test"]
    rows = [tqa["rows"][i] for i in test_idx]
    alpaca = json.loads((DATA / "alpaca_questions.json").read_text())

    lm = load_model(cfg["model_id"])
    tok = lm.tokenizer

    # ---- 1. CAA coherent grid on the test set ----
    op = cfg["families"]["dec"]
    coherent = [r for r in calib["grid"]
                if r["family"] == "dec" and r["coherent"]
                and not (r["layer"] == op["layer"] and r["alpha"] == op["alpha"])]
    print(f"[stage5] {len(coherent)} coherent CAA settings beyond the operating point", flush=True)
    caa = caa_vector(lm, alpaca[:N_CALIB], sorted({r["layer"] for r in coherent}))
    gdir = OUT / "dec_grid"
    gdir.mkdir(parents=True, exist_ok=True)
    for r in coherent:
        path = gdir / f"L{r['layer']}_a{r['alpha']:g}.jsonl"
        if count_jsonl(path) >= len(rows):
            print(f"[stage5] {path.name}: complete, skipping", flush=True)
            continue
        res = evaluate_truthfulqa_mc(lm, rows, layer=r["layer"],
                                     vector=caa[r["layer"]], coefficient=r["alpha"])
        with open(path, "w") as f:
            for qi, rr in zip(test_idx, res):
                f.write(json.dumps({"idx": qi, "mc1": rr["mc1"], "mc2": rr["mc2"]}) + "\n")
        mc2 = sum(x["mc2"] for x in res) / len(res)
        print(f"[stage5] dec L{r['layer']} a={r['alpha']}: test mc2={mc2:.4f}", flush=True)

    # ---- 2. aligned-64 readout lens ----
    lens_path = OUT / "lens_al64.pt"
    if not lens_path.exists():
        calib_texts = [
            tok.apply_chat_template([{"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True)
            for q in alpaca[:N_CALIB]
        ]
        final_layer = lm.n_layers - 1
        z = _last_token_residuals(lm, calib_texts, [final_layer])[final_layer].float().cuda()
        norm = get_final_norm(lm.model)
        gamma = norm.weight.detach().float().cuda()
        eps = float(getattr(norm, "variance_epsilon", 1e-5))
        W_U = get_lm_head(lm.model).weight.detach().float()
        Jbar = mean_jacobian(z, gamma, eps)
        W_tilde = (W_U.cuda() @ Jbar).contiguous()

        prompts = [QA_TEMPLATE.format(question=r["question"]) for r in rows[:N_LENS]]
        out = {"families": {}, "layer_star": {}, "sets": {}}
        for fam in cfg["families"]:
            v = vectors[f"{fam}/v_dec"].float().cuda()
            direct = W_tilde @ v
            top = direct.abs().topk(64).indices
            plus = [int(t) for t in top if float(direct[t]) > 0]
            minus = [int(t) for t in top if float(direct[t]) <= 0]
            out["sets"][fam] = {"plus": plus, "minus": minus}
            out.setdefault("decoded", {})[fam] = {
                "plus": [{"id": i, "tok": tok.decode([i])} for i in plus],
                "minus": [{"id": i, "tok": tok.decode([i])} for i in minus],
            }
            fop = cfg["families"][fam]
            out["layer_star"][fam] = fop["layer"]
            out["families"][fam] = {}
            base = t_readout_trajectory(lm, prompts, plus, minus)
            out["families"][fam]["baseline"] = base
            for cond in LENS_CONDS:
                out["families"][fam][cond] = t_readout_trajectory(
                    lm, prompts, plus, minus, layer=fop["layer"],
                    vector=vectors[f"{fam}/{cond}"], coefficient=fop["alpha"])
            print(f"[stage5] aligned-64 lens {fam}: |T+|={len(plus)} |T-|={len(minus)} done", flush=True)
        torch.save(out, lens_path)
        (OUT / "aligned64_decoded.json").write_text(
            json.dumps(out["decoded"], indent=2, ensure_ascii=False))
        print("[stage5] wrote aligned64_decoded.json", flush=True)

    print("[stage5] complete", flush=True)


if __name__ == "__main__":
    main()
