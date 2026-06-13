"""Two queued verification jobs.

1. Per-point certificate transfer: the zero-direct-effect certificate is exact
   for the calibration-averaged effective unembedding. Here we measure the
   residual direct effect |W_U J(z_i) v_perp| restricted to the aligned-64 set
   at each individual calibration readout point z_i, using the closed form
   J(z) u = (1/sigma) * gamma .* (u - z (z.u) / (d sigma^2)).

2. BPE boundary check: MC scoring concatenates token ids of prompt and
   " "+choice; verify this equals tokenizing the whole string, for every
   prompt-choice pair in the benchmark.

Writes results/recal/verify.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from liar.eval import QA_TEMPLATE  # noqa: E402
from liar.model import get_final_norm, get_lm_head, load_model  # noqa: E402
from liar.steering import _last_token_residuals  # noqa: E402
from liar.tokenset import aligned_token_set  # noqa: E402
from liar.unembedding import mean_jacobian  # noqa: E402

DATA = ROOT / "data"
RC = ROOT / "artifacts" / "recal"
OUT = ROOT / "results" / "recal"
N_CALIB = 256


def main() -> None:
    cfg = json.loads((RC / "config.json").read_text())
    vectors = torch.load(RC / "vectors.pt", weights_only=True)
    alpaca = json.loads((DATA / "alpaca_questions.json").read_text())
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())

    lm = load_model(cfg["model_id"])
    tok = lm.tokenizer
    out: dict = {}

    # ---- 2. BPE boundary check (cheap, do first) ----
    total = mismatch = 0
    for r in tqa["rows"]:
        q = QA_TEMPLATE.format(question=r["question"])
        a = tok.encode(q, add_special_tokens=True)
        whole_cache = {}
        for key in ("mc1_targets", "mc2_targets"):
            for c in r[key]["choices"]:
                if c in whole_cache:
                    continue
                whole_cache[c] = True
                b = tok.encode(" " + c, add_special_tokens=False)
                w = tok.encode(q + " " + c, add_special_tokens=True)
                total += 1
                if a + b != w:
                    mismatch += 1
    out["bpe_check"] = {"pairs": total, "mismatches": mismatch}
    print(f"[verify] bpe: {mismatch}/{total} mismatched pairs", flush=True)

    # ---- 1. per-point leakage ----
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
    d = z.shape[1]

    Jbar = mean_jacobian(z, gamma, eps)
    W_tilde = (W_U.cuda() @ Jbar).contiguous()

    def point_apply(zi, u):
        # J(z) u in closed form
        sigma2 = float((zi @ zi) / d + eps)
        sigma = sigma2 ** 0.5
        return (gamma * (u - zi * float(zi @ u) / (d * sigma2))) / sigma

    out["leakage"] = {}
    for fam in cfg["families"]:
        v_dec = vectors[f"{fam}/v_dec"].float().cuda()
        v_perp = vectors[f"{fam}/v_perp_al64"].float().cuda()
        ids, _ = aligned_token_set(W_tilde, v_dec, 64)
        rows = W_U[torch.tensor(ids)].cuda()  # (64, d)
        leak, scale = [], []
        for i in range(z.shape[0]):
            ju_p = point_apply(z[i], v_perp)
            ju_d = point_apply(z[i], v_dec)
            leak.append(float((rows @ ju_p).abs().max()))
            scale.append(float((rows @ ju_d).abs().max()))
        leak_t = torch.tensor(leak)
        scale_t = torch.tensor(scale)
        out["leakage"][fam] = {
            "n_points": len(leak),
            "leak_max": float(leak_t.max()),
            "leak_median": float(leak_t.median()),
            "leak_p95": float(leak_t.quantile(0.95)),
            "scale_median_vdec": float(scale_t.median()),
            "ratio_median": float((leak_t / scale_t).median()),
        }
        print(f"[verify] {fam}: per-point leak max={leak_t.max():.4f} "
              f"median={leak_t.median():.4f} vs |Av| median={scale_t.median():.3f}", flush=True)

    (OUT / "verify.json").write_text(json.dumps(out, indent=2))
    print("[verify] wrote results/recal/verify.json", flush=True)


if __name__ == "__main__":
    main()
