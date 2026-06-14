"""Cross-model aggregation (Mac-side). For every model run present under
results/recal[_<name>]/, recompute the headline depth statistics and emit a
cross-model comparison table, figure, and macros. Establishes whether the
double dissociation replicates beyond Llama-3-8B-Instruct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
ART = ROOT / "artifacts"
FIG = ROOT / "figures"
TAB = ROOT / "docs" / "tables"
N_BOOT = 10_000

# (dirname, artifact dirname, display label)
MODELS = [
    ("recal", "recal", "Llama-3-8B-Instruct"),
    ("recal_mistral", "recal_mistral", "Mistral-7B-Instruct-v0.3"),
    ("recal_qwen", "recal_qwen", "Qwen2.5-7B-Instruct"),
]
GREEN_D, BLUE_D, GRAY = "#2E9E57", "#2F8FCC", "#8C8C8C"
FAM_LABEL = {"dec": "CAA", "mm": "mass-mean"}


def load_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def align(base, cond, key="mc2"):
    pos = {r["idx"]: r[key] for r in cond}
    b, c = [], []
    for r in base:
        if r["idx"] in pos:
            b.append(r[key]); c.append(pos[r["idx"]])
    return np.array(b, float), np.array(c, float)


def boot_mean(x, rng, n=N_BOOT):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(num, den, rng, n=N_BOOT):
    k = len(num)
    idx = rng.integers(0, k, size=(n, k))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def model_stats(resdir, rng):
    base = load_jsonl(resdir / "baseline.jsonl")
    if not base:
        return None
    out = {}
    for fam in ("dec", "mm"):
        fdir = resdir / fam
        vd = load_jsonl(fdir / "v_dec.jsonl")
        if not vd:
            continue
        b2, v2 = align(base, vd)
        d_dec = v2 - b2
        e = boot_mean(d_dec, rng)
        ent = {"delta": e[0], "delta_ci": [e[1], e[2]]}
        perp = load_jsonl(fdir / "v_perp_al64.jsonl")
        if perp:
            _, p2 = align(base, perp)
            r = boot_ratio(p2 - b2, d_dec, rng)
            ent["rho"] = r[0]; ent["rho_ci"] = [r[1], r[2]]
        # aligned-minus-random
        rand = []
        for s in range(3):
            rr = load_jsonl(fdir / f"v_rand_s{s}.jsonl")
            if rr:
                _, rv = align(base, rr)
                rand.append(rv - b2)
        if perp and rand:
            d_al = p2 - b2
            d_rand = np.mean(rand, axis=0)
            k = len(d_al)
            idx = rng.integers(0, k, size=(N_BOOT, k))
            diff = (d_al[idx].mean(1) / d_dec[idx].mean(1)) - (d_rand[idx].mean(1) / d_dec[idx].mean(1))
            ent["amr"] = float((d_al.mean() / d_dec.mean()) - (d_rand.mean() / d_dec.mean()))
            ent["amr_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        out[fam] = ent
    return out


def main():
    rng = np.random.default_rng(0)
    present = []
    for resname, artname, label in MODELS:
        st = model_stats(RES / resname, rng)
        if st:
            present.append((label, artname, st))
            print(f"[stage6] {label}: "
                  + "; ".join(f"{FAM_LABEL[f]} d={st[f]['delta']:+.3f} rho={st[f].get('rho','-')}"
                              for f in st))
    if len(present) < 2:
        print(f"[stage6] only {len(present)} model(s) present; need >=2 for cross-model output")
        return

    # --- table ---
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"model & family & $\Delta$MC2 [95\% CI] & $\rho$ (al-64) [95\% CI] & aligned$-$random [95\% CI] \\",
             r"\midrule"]
    for label, _, st in present:
        for fi, fam in enumerate(("dec", "mm")):
            if fam not in st:
                continue
            e = st[fam]
            mcell = label if fi == 0 else ""
            d = f"${e['delta']:+.3f}$ {{\\scriptsize $[{e['delta_ci'][0]:+.3f}, {e['delta_ci'][1]:+.3f}]$}}"
            rho = (f"${e['rho']:.2f}$ {{\\scriptsize $[{e['rho_ci'][0]:.2f}, {e['rho_ci'][1]:.2f}]$}}"
                   if "rho" in e else "---")
            amr = (f"${e['amr']:+.2f}$ {{\\scriptsize $[{e['amr_ci'][0]:+.2f}, {e['amr_ci'][1]:+.2f}]$}}"
                   if "amr" in e else "---")
            lines.append(f"{mcell} & {FAM_LABEL[fam]} & {d} & {rho} & {amr} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "multimodel.tex").write_text("\n".join(lines) + "\n")

    # --- figure: rho per model per family ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    labels = [l for l, _, _ in present]
    x = np.arange(len(labels))
    for fam, col, off in (("mm", GREEN_D, -0.12), ("dec", BLUE_D, 0.12)):
        ys, los, his = [], [], []
        for _, _, st in present:
            e = st.get(fam, {})
            ys.append(e.get("rho", np.nan))
            ci = e.get("rho_ci", [np.nan, np.nan])
            los.append(ci[0]); his.append(ci[1])
        ys = np.array(ys); los = np.array(los); his = np.array(his)
        ax.errorbar(x + off, ys, yerr=[ys - los, his - ys], fmt="o", color=col, capsize=3,
                    ms=6, lw=1.4, label=f"{FAM_LABEL[fam]} $\\rho$ (al-64)")
    ax.axhline(1, ls=":", color="gray", lw=0.9)
    ax.axhline(0, ls=":", color="gray", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"depth statistic $\rho$ (MC2)")
    ax.set_title("Depth statistic across models")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(FIG / "multimodel.pdf")
    plt.close(fig)

    # --- macros ---
    L = []
    namekey = {"Mistral-7B-Instruct-v0.3": "Mistral", "Qwen2.5-7B-Instruct": "Qwen",
               "Llama-3-8B-Instruct": "Llama"}
    def sgn(s):
        return s.replace("+", "$+$").replace("-", "$-$")

    def ci(lo, hi, nd=2):
        return sgn(f"[{lo:+.{nd}f}, {hi:+.{nd}f}]")

    for label, _, st in present:
        nk = namekey.get(label, label.split("-")[0])
        for fam, ftag in (("dec", "Dec"), ("mm", "Mm")):
            e = st.get(fam)
            if not e:
                continue
            dval = sgn(f"{e['delta']:+.3f}")
            L.append(f"\\newcommand{{\\{nk}{ftag}Delta}}{{{dval}}}")
            L.append(f"\\newcommand{{\\{nk}{ftag}DeltaCI}}{{{ci(e['delta_ci'][0], e['delta_ci'][1], 3)}}}")
            sig = (e['delta_ci'][0] > 0 or e['delta_ci'][1] < 0)
            L.append(f"\\newcommand{{\\{nk}{ftag}Sig}}{{{'yes' if sig else 'no'}}}")
            if "rho" in e:
                L.append(f"\\newcommand{{\\{nk}{ftag}Rho}}{{{e['rho']:.2f}}}")
                L.append(f"\\newcommand{{\\{nk}{ftag}RhoCI}}{{{ci(*e['rho_ci'])}}}")
            if "amr" in e:
                aval = sgn(f"{e['amr']:+.2f}")
                L.append(f"\\newcommand{{\\{nk}{ftag}Amr}}{{{aval}}}")
                L.append(f"\\newcommand{{\\{nk}{ftag}AmrCI}}{{{ci(*e['amr_ci'])}}}")
    (TAB / "multimodel_numbers.tex").write_text("\n".join(L) + "\n")
    print(f"[stage6] wrote multimodel table + figure + {len(L)} macros ({len(present)} models)")


if __name__ == "__main__":
    main()
