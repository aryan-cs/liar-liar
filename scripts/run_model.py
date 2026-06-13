"""Multi-model replication: run the full coherence-gated, two-family depth
pipeline for an arbitrary HF causal LM, namespaced by a short model name.

Usage:  python scripts/run_model.py <name> <hf_model_id>
Example: python scripts/run_model.py mistral mistralai/Mistral-7B-Instruct-v0.3

Writes artifacts/recal_<name>/ and results/recal_<name>/ in exactly the same
layout as the Llama-3 run (config, calibration, certificates, tokensets,
capture_ids, vectors; baseline.jsonl, {fam}/{cond}.jsonl, para_*). Resumable
by row count. The RMSNorm correction and standard architecture accessors apply
to any Llama-style pre-norm transformer (Llama, Mistral, Qwen2.5); models with
a (1+w) norm convention or logit softcapping (e.g. Gemma-2) are not handled.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

from liar.eval import evaluate_truthfulqa_mc, held_out_nll, logit_shift_eta  # noqa: E402
from liar.model import get_final_norm, get_lm_head, load_model  # noqa: E402
from liar.progress import count_jsonl, write_progress  # noqa: E402
from liar.steering import _last_token_residuals, caa_vector, mass_mean_vector  # noqa: E402
from liar.tokenset import aligned_token_set, curated_honesty_tokens, statistical_token_set  # noqa: E402
from liar.unembedding import mean_jacobian, projector_perp, token_subspace  # noqa: E402

DATA = ROOT / "data"
ALPHA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
COHERENCE_GATE = 1.5
ALIGNED_KS = [16, 64, 256, 1024]
RAND_SEEDS = [0, 1, 2]
RAND_K = 64
N_CALIB = 256
N_HELDOUT = 96
N_STAT = 48
N_PARA = 150
SEED = 0
CONDS = ["v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256", "v_perp_al1024",
         "v_perp_cur", "v_perp_stat", "v_par_al64", "v_perp_al64_nm",
         "v_rand_s0", "v_rand_s1", "v_rand_s2"]


def qa_statements(rows):
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
    aligned = {k: aligned_token_set(W_tilde, v.to(DEVICE), k)[0] for k in ALIGNED_KS}
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


def main():
    name, model_id = sys.argv[1], sys.argv[2]
    OUT_A = ROOT / "artifacts" / f"recal_{name}"
    OUT_R = ROOT / "results" / f"recal_{name}"
    OUT_A.mkdir(parents=True, exist_ok=True)
    OUT_R.mkdir(parents=True, exist_ok=True)
    if (OUT_R / "DONE").exists():
        print(f"[{name}] already complete", flush=True)
        return

    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    rows = tqa["rows"]
    val_rows = [rows[i] for i in tqa["splits"]["val_sweep"]]
    mm_rows = [rows[i] for i in tqa["splits"]["mm_source"]]
    test_idx = tqa["splits"]["test"]
    test_rows = [rows[i] for i in test_idx]
    alpaca = json.loads((DATA / "alpaca_questions.json").read_text())
    heldout_text = alpaca[N_CALIB:N_CALIB + N_HELDOUT]

    lm = load_model(model_id)
    torch.manual_seed(SEED)
    tok = lm.tokenizer
    L = lm.n_layers
    layer_grid = sorted({max(1, int(round(0.375 * L))), max(2, int(round(0.4375 * L)))})
    print(f"[{name}] L={L} d={lm.d_model} V={lm.vocab_size} layer_grid={layer_grid}", flush=True)

    if (OUT_A / "DONE").exists():
        cfg = json.loads((OUT_A / "config.json").read_text())
        vectors = torch.load(OUT_A / "vectors.pt", weights_only=True)
        capture = json.loads((OUT_A / "capture_ids.json").read_text())
    else:
        caa = caa_vector(lm, alpaca[:N_CALIB], layer_grid)
        stmts, labels = qa_statements(mm_rows)
        mm = mass_mean_vector(lm, stmts, labels, layer_grid)
        families = {"dec": caa, "mm": mm}

        calib_texts = [tok.apply_chat_template([{"role": "user", "content": q}],
                       tokenize=False, add_generation_prompt=True) for q in alpaca[:N_CALIB]]
        z = _last_token_residuals(lm, calib_texts, [L - 1])[L - 1]
        norm = get_final_norm(lm.model)
        gamma = norm.weight.detach().float().cpu()
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
        Jbar = mean_jacobian(z.to(DEVICE), gamma.to(DEVICE), eps)
        W_U = get_lm_head(lm.model).weight.detach().float()
        W_tilde = (W_U @ Jbar).contiguous()

        tp, tm, sp, sm = curated_honesty_tokens(tok)
        stat_p, stat_m, _ = statistical_token_set(
            lm, alpaca[N_CALIB + N_HELDOUT:N_CALIB + N_HELDOUT + N_STAT], k=32)

        base_nll = held_out_nll(lm, heldout_text)
        base_eval = evaluate_truthfulqa_mc(lm, val_rows)
        base_mc2 = sum(r["mc2"] for r in base_eval) / len(base_eval)
        cur_plus, cur_minus = list(tp.values()), list(tm.values())
        cap_eta = sorted(set(cur_plus) | set(cur_minus))

        calibration = []
        for fam, vecs in families.items():
            for layer in layer_grid:
                v = vecs[layer]
                for a in ALPHA_GRID:
                    nll = held_out_nll(lm, heldout_text, layer=layer, vector=v, coefficient=a)
                    ratio = math.exp(nll - base_nll)
                    ev = evaluate_truthfulqa_mc(lm, val_rows, layer=layer, vector=v,
                                                coefficient=a, capture_ids=cap_eta)
                    mc2 = sum(r["mc2"] for r in ev) / len(ev)
                    etas = [logit_shift_eta(torch.tensor(r["eta_logits"]), cap_eta, cur_plus, cur_minus)
                            for r in ev if r.get("eta_logits") is not None]
                    calibration.append({"family": fam, "layer": layer, "alpha": a,
                        "nll_ratio": ratio, "val_mc2": mc2, "val_mc2_delta": mc2 - base_mc2,
                        "val_eta": sum(etas) / len(etas) if etas else None,
                        "coherent": ratio <= COHERENCE_GATE})
                    print(f"[{name}] {fam} L{layer} a={a}: ppl={ratio:.3f} d={mc2-base_mc2:+.4f} "
                          f"{'OK' if ratio <= COHERENCE_GATE else 'INCOHERENT'}", flush=True)
        (OUT_A / "calibration.json").write_text(json.dumps(
            {"base_nll": base_nll, "base_mc2": base_mc2, "gate": COHERENCE_GATE, "grid": calibration}, indent=2))

        cfg = {"model_id": model_id, "name": name, "gate": COHERENCE_GATE, "seed": SEED,
               "n_layers": L, "layer_grid": layer_grid, "aligned_ks": ALIGNED_KS,
               "rand_k": RAND_K, "families": {}}
        for fam in families:
            cand = [r for r in calibration if r["family"] == fam and r["coherent"]]
            if not cand:
                cand = [r for r in calibration if r["family"] == fam and r["alpha"] == min(ALPHA_GRID)]
            best = max(cand, key=lambda r: r["val_mc2_delta"])
            cfg["families"][fam] = {"layer": best["layer"], "alpha": best["alpha"],
                "val_mc2_delta": best["val_mc2_delta"], "ppl_ratio": best["nll_ratio"]}
            print(f"[{name}] OP {fam}: L{best['layer']} a={best['alpha']} "
                  f"(ppl={best['nll_ratio']:.3f}, d={best['val_mc2_delta']:+.4f})", flush=True)

        all_vectors, certs = {}, {}
        capture = set(cap_eta) | set(sp.values()) | set(sm.values()) | set(stat_p) | set(stat_m)
        for fam in families:
            op = cfg["families"][fam]
            v = families[fam][op["layer"]]
            variants, al64 = build_variants(W_tilde, v, tp, tm, stat_p, stat_m, certs, fam)
            for vn, vec in variants.items():
                all_vectors[f"{fam}/{vn}"] = vec
            capture |= set(al64)
        (OUT_A / "tokensets.json").write_text(json.dumps({"curated_plus": tp, "curated_minus": tm,
            "spill_plus": sp, "spill_minus": sm, "statistical_plus": stat_p, "statistical_minus": stat_m}, indent=2))
        capture = sorted(capture)
        (OUT_A / "capture_ids.json").write_text(json.dumps(capture))
        (OUT_A / "certificates.json").write_text(json.dumps(certs, indent=2))
        (OUT_A / "config.json").write_text(json.dumps(cfg, indent=2))
        torch.save({k: v.cpu() for k, v in all_vectors.items()}, OUT_A / "vectors.pt")
        (OUT_A / "DONE").touch()
        vectors = all_vectors

    # ---- headline matrix ----
    base_path = OUT_R / "baseline.jsonl"
    if count_jsonl(base_path) < len(test_rows):
        res = evaluate_truthfulqa_mc(lm, test_rows, capture_ids=capture)
        with open(base_path, "w") as f:
            for qi, r in zip(test_idx, res):
                f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"],
                                    "eta_logits": r.get("eta_logits")}) + "\n")
        print(f"[{name}] baseline mc2={sum(r['mc2'] for r in res)/len(res):.4f}", flush=True)
    for fam in cfg["families"]:
        op = cfg["families"][fam]
        fdir = OUT_R / fam
        fdir.mkdir(parents=True, exist_ok=True)
        for cond in CONDS:
            path = fdir / f"{cond}.jsonl"
            if count_jsonl(path) >= len(test_rows):
                continue
            res = evaluate_truthfulqa_mc(lm, test_rows, layer=op["layer"],
                vector=vectors[f"{fam}/{cond}"], coefficient=op["alpha"], capture_ids=capture)
            with open(path, "w") as f:
                for qi, r in zip(test_idx, res):
                    f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"],
                                        "eta_logits": r.get("eta_logits")}) + "\n")
            print(f"[{name}] {fam}/{cond} mc2={sum(r['mc2'] for r in res)/len(res):.4f}", flush=True)

    # ---- paraphrase OOD (reuse cached paraphrases) ----
    para_path = DATA / "paraphrases.json"
    if para_path.exists():
        paras = json.loads(para_path.read_text())
        para_rows = [{**r, "question": p} for r, p in zip(test_rows[:N_PARA], paras)]
        bpath = OUT_R / "para_baseline.jsonl"
        if count_jsonl(bpath) < len(para_rows):
            res = evaluate_truthfulqa_mc(lm, para_rows)
            with open(bpath, "w") as f:
                for qi, r in zip(test_idx[:N_PARA], res):
                    f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"]}) + "\n")
        for fam in cfg["families"]:
            op = cfg["families"][fam]
            for cond in ("v_dec", "v_perp_al64"):
                path = OUT_R / fam / f"para_{cond}.jsonl"
                if count_jsonl(path) >= len(para_rows):
                    continue
                res = evaluate_truthfulqa_mc(lm, para_rows, layer=op["layer"],
                    vector=vectors[f"{fam}/{cond}"], coefficient=op["alpha"])
                with open(path, "w") as f:
                    for qi, r in zip(test_idx[:N_PARA], res):
                        f.write(json.dumps({"idx": qi, "mc1": r["mc1"], "mc2": r["mc2"]}) + "\n")

    write_progress(OUT_R, "done", 1, 1, {})
    (OUT_R / "DONE").touch()
    print(f"[{name}] complete", flush=True)


if __name__ == "__main__":
    main()
