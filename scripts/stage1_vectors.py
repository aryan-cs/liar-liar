"""Stage 1: calibration, token sets, steering vectors, layer/alpha sweep,
projections, and zero-direct-effect certificates.

Artifacts (artifacts/stage1/):
  vectors.pt        all steering vectors and projected variants (float32 cpu)
  tokensets.json    token ids + surfaces for every T construction
  capture_ids.json  union readout list captured during stage-2 scoring
  sweep.json        layer/alpha sweep table on the val slice
  config.json       chosen layer, alpha, model id, seed
  certificates.json max |A v_perp| per token set (the zero-direct-effect check)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from liar.eval import evaluate_truthfulqa_mc  # noqa: E402
from liar.model import get_final_norm, get_lm_head, load_model  # noqa: E402
from liar.progress import write_progress  # noqa: E402
from liar.steering import caa_vector, contrast_chat_texts, mass_mean_vector  # noqa: E402
from liar.tokenset import aligned_token_set, curated_honesty_tokens, statistical_token_set  # noqa: E402
from liar.unembedding import mean_jacobian, projector_perp, token_subspace  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "stage1"

LAYER_GRID = [10, 12, 14, 16]
ALPHA_GRID = [4.0, 8.0, 12.0]
ALIGNED_KS = [16, 64, 256, 1024]
RAND_SEEDS = [0, 1, 2]
RAND_K = 64
N_CALIB = 256
SEED = 0


def qa_statements(rows: list[dict]) -> tuple[list[str], list[bool]]:
    """True/false statements derived from TruthfulQA mc1 targets."""
    stmts, labels = [], []
    for r in rows:
        choices = r["mc1_targets"]["choices"]
        labs = r["mc1_targets"]["labels"]
        true_c = [c for c, l in zip(choices, labs) if l == 1]
        false_c = [c for c, l in zip(choices, labs) if l == 0]
        if true_c and false_c:
            stmts.append(f"Q: {r['question']}\nA: {true_c[0]}")
            labels.append(True)
            stmts.append(f"Q: {r['question']}\nA: {false_c[0]}")
            labels.append(False)
    return stmts, labels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "DONE").exists():
        print("[stage1] already complete", flush=True)
        return

    model_id = json.loads((ROOT / "artifacts" / "stage0" / "model.json").read_text())["model_id"]
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    rows = tqa["rows"]
    val_rows = [rows[i] for i in tqa["splits"]["val_sweep"]]
    mm_rows = [rows[i] for i in tqa["splits"]["mm_source"]]
    alpaca = json.loads((DATA / "alpaca_questions.json").read_text())

    write_progress(OUT, "running", 0, 8, {"step": "load_model"})
    lm = load_model(model_id)
    torch.manual_seed(SEED)

    # --- 1. CAA vectors on the layer grid ---
    write_progress(OUT, "running", 1, 8, {"step": "caa_vectors"})
    caa = caa_vector(lm, alpaca[:N_CALIB], LAYER_GRID)

    # --- 2. layer/alpha sweep on the val slice (MC2 delta) ---
    write_progress(OUT, "running", 2, 8, {"step": "sweep"})
    base = evaluate_truthfulqa_mc(lm, val_rows)
    base_mc2 = sum(r["mc2"] for r in base) / len(base)
    sweep = []
    for li in LAYER_GRID:
        for a in ALPHA_GRID:
            res = evaluate_truthfulqa_mc(lm, val_rows, layer=li, vector=caa[li], coefficient=a)
            mc2 = sum(r["mc2"] for r in res) / len(res)
            sweep.append({"layer": li, "alpha": a, "mc2": mc2, "delta": mc2 - base_mc2})
            print(f"[stage1] sweep layer={li} alpha={a} mc2={mc2:.4f} d={mc2-base_mc2:+.4f}", flush=True)
    best = max(sweep, key=lambda s: s["delta"])
    layer_star, alpha_star = best["layer"], best["alpha"]
    (OUT / "sweep.json").write_text(json.dumps({"baseline_mc2": base_mc2, "grid": sweep, "best": best}, indent=2))

    v_dec = caa[layer_star]

    # --- 3. mass-mean vector at layer* ---
    write_progress(OUT, "running", 3, 8, {"step": "mass_mean"})
    stmts, labels = qa_statements(mm_rows)
    v_mm = mass_mean_vector(lm, stmts, labels, [layer_star])[layer_star]

    # --- 4. calibration residuals -> mean Jacobian -> effective unembedding ---
    write_progress(OUT, "running", 4, 8, {"step": "effective_unembedding"})
    from liar.steering import _last_token_residuals

    tok = lm.tokenizer
    calib_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for q in alpaca[:N_CALIB]
    ]
    final_layer = lm.n_layers - 1
    z = _last_token_residuals(lm, calib_texts, [final_layer])[final_layer]  # (N, d) cpu
    norm = get_final_norm(lm.model)
    gamma = norm.weight.detach().float().cpu()
    eps = float(getattr(norm, "variance_epsilon", 1e-5))
    Jbar = mean_jacobian(z.cuda(), gamma.cuda(), eps)  # (d, d) float32 gpu
    W_U = get_lm_head(lm.model).weight.detach().float()  # (V, d) gpu
    W_tilde = W_U @ Jbar  # (V, d) float32 gpu

    # --- 5. token sets ---
    write_progress(OUT, "running", 5, 8, {"step": "token_sets"})
    tp, tm, sp, sm = curated_honesty_tokens(tok)
    stat_p, stat_m, _shift = statistical_token_set(lm, alpaca[N_CALIB : N_CALIB + 128], k=32)
    aligned: dict[int, list[int]] = {}
    v_dec_gpu = v_dec.cuda()
    for k in ALIGNED_KS:
        ids, _scores = aligned_token_set(W_tilde, v_dec_gpu, k)
        aligned[k] = ids
    tokensets = {
        "curated_plus": tp,
        "curated_minus": tm,
        "spill_plus": sp,
        "spill_minus": sm,
        "statistical_plus": stat_p,
        "statistical_minus": stat_m,
        "aligned": {str(k): v for k, v in aligned.items()},
    }
    (OUT / "tokensets.json").write_text(json.dumps(tokensets, indent=2))

    capture = sorted(
        set(tp.values()) | set(tm.values()) | set(sp.values()) | set(sm.values())
        | set(stat_p) | set(stat_m) | set(aligned[64])
    )
    (OUT / "capture_ids.json").write_text(json.dumps(capture))

    # --- 6. projections + certificates ---
    write_progress(OUT, "running", 6, 8, {"step": "projections"})
    sets_for_projection: dict[str, list[int]] = {
        f"al{k}": aligned[k] for k in ALIGNED_KS
    }
    sets_for_projection["cur"] = sorted(set(tp.values()) | set(tm.values()))
    sets_for_projection["stat"] = sorted(set(stat_p) | set(stat_m))

    vectors: dict[str, torch.Tensor] = {"v_dec": v_dec, "v_mm": v_mm}
    certs: dict[str, dict] = {}
    for name, ids in sets_for_projection.items():
        A = token_subspace(W_tilde, ids).cpu().double()  # (k, d) float64
        P = projector_perp(A)
        vp64 = P @ v_dec.double()
        vp = vp64.float().contiguous()
        vectors[f"v_perp_{name}"] = vp
        direct_before = float((A @ v_dec.double()).abs().max())
        direct_after = float((A @ vp64).abs().max())
        certs[name] = {
            "k": len(ids),
            "max_direct_before": direct_before,
            "max_direct_after": direct_after,
            "norm_ratio": float(vp.norm() / v_dec.norm()),
        }
        print(f"[stage1] {name}: |Av| {direct_before:.4f} -> {direct_after:.2e}, "
              f"norm ratio {certs[name]['norm_ratio']:.4f}", flush=True)

    vectors["v_par_al64"] = (v_dec.float() - vectors["v_perp_al64"]).contiguous()
    nm = vectors["v_perp_al64"] * (v_dec.norm() / vectors["v_perp_al64"].norm())
    vectors["v_perp_al64_nm"] = nm.contiguous()

    d = v_dec.shape[0]
    for s in RAND_SEEDS:
        g = torch.Generator().manual_seed(s)
        Q, _ = torch.linalg.qr(torch.randn(d, RAND_K, generator=g))
        vr = v_dec.float() - Q @ (Q.T @ v_dec.float())
        vectors[f"v_rand_s{s}"] = vr.contiguous()
        certs[f"rand_s{s}"] = {"k": RAND_K, "norm_ratio": float(vr.norm() / v_dec.norm())}

    (OUT / "certificates.json").write_text(json.dumps(certs, indent=2))
    torch.save({k: v.cpu() for k, v in vectors.items()}, OUT / "vectors.pt")

    # --- 7. config ---
    write_progress(OUT, "running", 7, 8, {"step": "config"})
    (OUT / "config.json").write_text(json.dumps({
        "model_id": model_id,
        "layer_star": layer_star,
        "alpha_star": alpha_star,
        "seed": SEED,
        "n_calib": N_CALIB,
        "aligned_ks": ALIGNED_KS,
        "rand_k": RAND_K,
        "rand_seeds": RAND_SEEDS,
    }, indent=2))

    write_progress(OUT, "done", 8, 8, {})
    (OUT / "DONE").touch()
    print(f"[stage1] complete: layer*={layer_star} alpha*={alpha_star}", flush=True)


if __name__ == "__main__":
    main()
