"""Recalibration stage: coherence-gated operating points for both steering
vector families, then build all projection variants at those points.

The alpha=8 / layer=10 run was a broken-instrument artifact: ||alpha*v_dec||
was 2.8x the median residual norm and free generation was gibberish, while
teacher-forced MC2 was nearly blind to the collapse. Here we select the
operating point (layer, alpha) per vector by a criterion independent of the
depth statistic rho: the largest validation MC2 improvement among settings
that keep held-out fluency within a coherence gate.

Outputs (artifacts/recal/):
  calibration.json   full (family, layer, alpha) -> {nll_ratio, val_mc2_delta,
                     val_eta, alpha_norm_ratio} table  (auditable curve)
  config.json        chosen (layer, alpha) per family + gate + model id
  tokensets.json     shared token-set constructions
  capture_ids.json   union readout captured during scoring
  certificates.json  zero-direct-effect certificates per family/set
  vectors.pt         namespaced: {family}/{variant} -> (d,) float32 cpu
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from liar.eval import evaluate_truthfulqa_mc, held_out_nll, logit_shift_eta  # noqa: E402
from liar.model import get_final_norm, get_lm_head, load_model  # noqa: E402
from liar.progress import write_progress  # noqa: E402
from liar.steering import _last_token_residuals, caa_vector, mass_mean_vector  # noqa: E402
from liar.tokenset import aligned_token_set, curated_honesty_tokens, statistical_token_set  # noqa: E402
from liar.unembedding import mean_jacobian, projector_perp, token_subspace  # noqa: E402

DATA = ROOT / "data"
S1 = ROOT / "artifacts" / "stage1"
OUT = ROOT / "artifacts" / "recal"

LAYER_GRID = [12, 14]
ALPHA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
COHERENCE_GATE = 1.5          # max held-out PPL ratio (exp of NLL diff)
ALIGNED_KS = [16, 64, 256, 1024]
RAND_SEEDS = [0, 1, 2]
RAND_K = 64
N_CALIB = 256
N_HELDOUT = 96
SEED = 0


def qa_statements(rows: list[dict]) -> tuple[list[str], list[bool]]:
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


def build_variants(W_tilde, v, tp, tm, stat_p, stat_m, certs_out, family):
    """All projection variants for one steering vector at its operating point."""
    aligned = {k: aligned_token_set(W_tilde, v.cuda(), k)[0] for k in ALIGNED_KS}
    sets = {f"al{k}": aligned[k] for k in ALIGNED_KS}
    sets["cur"] = sorted(set(tp.values()) | set(tm.values()))
    sets["stat"] = sorted(set(stat_p) | set(stat_m))

    out = {"v_dec": v.contiguous()}
    for name, ids in sets.items():
        A = token_subspace(W_tilde, ids).cpu().double()
        P = projector_perp(A)
        vp64 = P @ v.double()
        vp = vp64.float().contiguous()
        out[f"v_perp_{name}"] = vp
        certs_out[f"{family}/{name}"] = {
            "k": len(ids),
            "max_direct_before": float((A @ v.double()).abs().max()),
            "max_direct_after": float((A @ vp64).abs().max()),
            "norm_ratio": float(vp.norm() / v.norm()),
        }
    out["v_par_al64"] = (v.double() - out["v_perp_al64"].double()).float().contiguous()
    out["v_perp_al64_nm"] = (out["v_perp_al64"] * (v.norm() / out["v_perp_al64"].norm())).contiguous()
    d = v.shape[0]
    for s in RAND_SEEDS:
        g = torch.Generator().manual_seed(s)
        Q, _ = torch.linalg.qr(torch.randn(d, RAND_K, generator=g))
        vr = (v.double() - Q.double() @ (Q.double().T @ v.double())).float()
        out[f"v_rand_s{s}"] = vr.contiguous()
        certs_out[f"{family}/rand_s{s}"] = {"k": RAND_K, "norm_ratio": float(vr.norm() / v.norm())}
    return out, aligned[64]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "DONE").exists():
        print("[recal] already complete", flush=True)
        return

    model_id = json.loads((S1 / "config.json").read_text())["model_id"]
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    rows = tqa["rows"]
    val_rows = [rows[i] for i in tqa["splits"]["val_sweep"]]
    mm_rows = [rows[i] for i in tqa["splits"]["mm_source"]]
    alpaca = json.loads((DATA / "alpaca_questions.json").read_text())
    heldout_text = alpaca[N_CALIB : N_CALIB + N_HELDOUT]  # fluent English, disjoint from vectors

    write_progress(OUT, "running", 0, 6, {"step": "load"})
    lm = load_model(model_id)
    torch.manual_seed(SEED)
    tok = lm.tokenizer

    # --- vectors at candidate layers ---
    write_progress(OUT, "running", 1, 6, {"step": "vectors"})
    caa = caa_vector(lm, alpaca[:N_CALIB], LAYER_GRID)
    stmts, labels = qa_statements(mm_rows)
    mm = mass_mean_vector(lm, stmts, labels, LAYER_GRID)
    families = {"dec": caa, "mm": mm}

    # --- effective unembedding (shared; readout-point Jacobian) ---
    write_progress(OUT, "running", 2, 6, {"step": "effective_unembedding"})
    calib_texts = [
        tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
        for q in alpaca[:N_CALIB]
    ]
    final_layer = lm.n_layers - 1
    z = _last_token_residuals(lm, calib_texts, [final_layer])[final_layer]
    norm = get_final_norm(lm.model)
    gamma = norm.weight.detach().float().cpu()
    eps = float(getattr(norm, "variance_epsilon", 1e-5))
    Jbar = mean_jacobian(z.cuda(), gamma.cuda(), eps)
    W_U = get_lm_head(lm.model).weight.detach().float()
    W_tilde = (W_U @ Jbar).contiguous()

    # --- token sets (shared) ---
    tp, tm, sp, sm = curated_honesty_tokens(tok)
    # The 400-instruction Alpaca pool leaves exactly 48 instructions after the
    # 256 CAA prompts and the 96 held-out fluency texts. Slice the remainder
    # explicitly and fail loudly if the pool ever shrinks.
    stat_block = alpaca[N_CALIB + N_HELDOUT:]
    assert len(stat_block) == 48, f"statistical-token-set pool changed: {len(stat_block)} != 48"
    stat_p, stat_m, _ = statistical_token_set(lm, stat_block, k=32)

    # --- coherence + behavioral calibration ---
    write_progress(OUT, "running", 3, 6, {"step": "calibrate"})
    base_nll = held_out_nll(lm, heldout_text)
    base_eval = evaluate_truthfulqa_mc(lm, val_rows)
    base_mc2 = sum(r["mc2"] for r in base_eval) / len(base_eval)

    cur_plus = list(tp.values())
    cur_minus = list(tm.values())
    cap_for_eta = sorted(set(cur_plus) | set(cur_minus))

    import math
    calibration = []
    median_h = {}
    for fam, vecs in families.items():
        for layer in LAYER_GRID:
            v = vecs[layer]
            for a in ALPHA_GRID:
                nll = held_out_nll(lm, heldout_text, layer=layer, vector=v, coefficient=a)
                ratio = math.exp(nll - base_nll)
                ev = evaluate_truthfulqa_mc(lm, val_rows, layer=layer, vector=v,
                                            coefficient=a, capture_ids=cap_for_eta)
                mc2 = sum(r["mc2"] for r in ev) / len(ev)
                etas = [logit_shift_eta(torch.tensor(r["eta_logits"]), cap_for_eta,
                                        cur_plus, cur_minus)
                        for r in ev if r.get("eta_logits") is not None]
                rec = {
                    "family": fam, "layer": layer, "alpha": a,
                    "nll_ratio": ratio, "val_mc2": mc2, "val_mc2_delta": mc2 - base_mc2,
                    "val_eta": sum(etas) / len(etas) if etas else None,
                    "alpha_vnorm": float(a * v.norm()),
                    "coherent": ratio <= COHERENCE_GATE,
                }
                calibration.append(rec)
                print(f"[recal] {fam} L{layer} a={a}: ppl_ratio={ratio:.3f} "
                      f"mc2={mc2:.4f} d={mc2-base_mc2:+.4f} eta={rec['val_eta']:+.3f} "
                      f"{'OK' if rec['coherent'] else 'INCOHERENT'}", flush=True)

    (OUT / "calibration.json").write_text(json.dumps(
        {"base_nll": base_nll, "base_mc2": base_mc2, "gate": COHERENCE_GATE,
         "grid": calibration}, indent=2))

    # --- select operating point per family: max coherent val_mc2_delta ---
    config = {"model_id": model_id, "gate": COHERENCE_GATE, "seed": SEED,
              "aligned_ks": ALIGNED_KS, "rand_k": RAND_K, "families": {}}
    for fam in families:
        cand = [r for r in calibration if r["family"] == fam and r["coherent"]]
        if not cand:
            cand = [r for r in calibration if r["family"] == fam and r["alpha"] == min(ALPHA_GRID)]
        best = max(cand, key=lambda r: r["val_mc2_delta"])
        config["families"][fam] = {"layer": best["layer"], "alpha": best["alpha"],
                                   "val_mc2_delta": best["val_mc2_delta"],
                                   "ppl_ratio": best["nll_ratio"], "val_eta": best["val_eta"]}
        print(f"[recal] OP {fam}: layer={best['layer']} alpha={best['alpha']} "
              f"(ppl_ratio={best['nll_ratio']:.3f}, val d_mc2={best['val_mc2_delta']:+.4f})", flush=True)

    # --- build variants at operating points ---
    write_progress(OUT, "running", 4, 6, {"step": "variants"})
    all_vectors: dict[str, torch.Tensor] = {}
    certs: dict[str, dict] = {}
    capture = set(cap_for_eta) | set(sp.values()) | set(sm.values()) | set(stat_p) | set(stat_m)
    for fam in families:
        op = config["families"][fam]
        v = families[fam][op["layer"]]
        variants, al64 = build_variants(W_tilde, v, tp, tm, stat_p, stat_m, certs, fam)
        for name, vec in variants.items():
            all_vectors[f"{fam}/{name}"] = vec
        capture |= set(al64)

    tokensets = {
        "curated_plus": tp, "curated_minus": tm,
        "spill_plus": sp, "spill_minus": sm,
        "statistical_plus": stat_p, "statistical_minus": stat_m,
    }
    (OUT / "tokensets.json").write_text(json.dumps(tokensets, indent=2))
    (OUT / "capture_ids.json").write_text(json.dumps(sorted(capture)))
    (OUT / "certificates.json").write_text(json.dumps(certs, indent=2))
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    torch.save({k: v.cpu() for k, v in all_vectors.items()}, OUT / "vectors.pt")

    write_progress(OUT, "done", 6, 6, {})
    (OUT / "DONE").touch()
    print("[recal] complete", flush=True)


if __name__ == "__main__":
    main()
