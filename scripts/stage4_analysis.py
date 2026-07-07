"""Stage 4 (Mac-side): aggregate results, statistics, figures, and tables.

Inputs (fetched by scripts/fetch_results.sh):
  artifacts/stage1/{config,sweep,tokensets,capture_ids,certificates}.json
  results/stage2/{condition}.jsonl
  results/stage3/para_{condition}.jsonl, lens.pt

Outputs:
  results/summary.json          every number cited in the paper
  figures/*.pdf                 publication figures
  docs/paper/tables/*.tex       generated table bodies
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liar.plotting import FAMILY_COLOR, FAMILY_SHADE, PAPER_FONT_RC, TURBO  # noqa: E402

S1 = ROOT / "artifacts" / "stage1"
S2 = ROOT / "results" / "stage2"
S3 = ROOT / "results" / "stage3"
FIG = ROOT / "figures"
TAB = ROOT / "docs" / "paper" / "tables"

# Legacy pipeline: keep condition-level accents on the real Turbo map. Family
# identity in the current pipeline comes from the shared sky-purple pairing.
PALETTE = [
    TURBO["selected"], TURBO["perp"], TURBO["cool_accent"],
    TURBO["anchor"], TURBO["warm_accent"], TURBO["parallel"],
]
CAA_COLOR = FAMILY_COLOR["dec"]
MM_COLOR = FAMILY_COLOR["mm"]
N_BOOT = 10_000
SEED = 0

CONDS_MAIN = [
    "baseline", "v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256",
    "v_perp_al1024", "v_perp_cur", "v_perp_stat", "v_par_al64",
    "v_perp_al64_nm", "v_rand_s0", "v_rand_s1", "v_rand_s2", "v_mm",
]

# ``\bot``/``\Vert`` use the Computer Modern symbols that visually match the
# paper's ``\perp``/``\parallel`` notation without Matplotlib's STIX fallback.
LABELS = {
    "baseline": "baseline", "v_dec": r"$v_{\mathrm{dec}}$",
    "v_perp_al16": r"$v^{\bot}$ al-16", "v_perp_al64": r"$v^{\bot}$ al-64",
    "v_perp_al256": r"$v^{\bot}$ al-256", "v_perp_al1024": r"$v^{\bot}$ al-1024",
    "v_perp_cur": r"$v^{\bot}$ curated", "v_perp_stat": r"$v^{\bot}$ statistical",
    "v_par_al64": r"$v^{\Vert}$ al-64", "v_perp_al64_nm": r"$v^{\bot}$ al-64 (nm)",
    "v_rand_s0": "rand-64 s0", "v_rand_s1": "rand-64 s1", "v_rand_s2": "rand-64 s2",
    "v_mm": r"$v_{\mathrm{mm}}$",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return rows


def paired_arrays(results: dict[str, list[dict]], key: str) -> dict[str, np.ndarray]:
    """Align per-question metric arrays across conditions by idx."""
    base_idx = [r["idx"] for r in results["baseline"]]
    pos = {qi: i for i, qi in enumerate(base_idx)}
    out = {}
    for cond, rows in results.items():
        arr = np.full(len(base_idx), np.nan)
        for r in rows:
            if r["idx"] in pos:
                arr[pos[r["idx"]]] = r[key]
        out[cond] = arr
    return out


def boot_ratio(num: np.ndarray, den: np.ndarray, rng: np.random.Generator,
               n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Bootstrap percentile CI for mean(num)/mean(den), paired resampling."""
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    nm = num[idx].mean(axis=1)
    dm = den[idx].mean(axis=1)
    ratios = nm / dm
    point = float(num.mean() / den.mean())
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return point, float(lo), float(hi)


def boot_mean(x: np.ndarray, rng: np.random.Generator,
              n_boot: int = N_BOOT) -> tuple[float, float, float]:
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def eta_per_question(rows: list[dict], capture: list[int],
                     plus: list[int], minus: list[int]) -> np.ndarray:
    pos = {tid: i for i, tid in enumerate(capture)}
    p_idx = [pos[t] for t in plus if t in pos]
    m_idx = [pos[t] for t in minus if t in pos]
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        el = r.get("eta_logits")
        if el is None:
            continue
        el = np.asarray(el)
        out[i] = el[p_idx].mean() - el[m_idx].mean()
    return out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cfg = json.loads((S1 / "config.json").read_text())
    certs = json.loads((S1 / "certificates.json").read_text())
    tokensets = json.loads((S1 / "tokensets.json").read_text())
    capture = json.loads((S1 / "capture_ids.json").read_text())
    sweep = json.loads((S1 / "sweep.json").read_text())

    results = {c: load_jsonl(S2 / f"{c}.jsonl") for c in CONDS_MAIN if (S2 / f"{c}.jsonl").exists()}
    n_q = len(results["baseline"])
    mc2 = paired_arrays(results, "mc2")
    mc1 = paired_arrays(results, "mc1")

    summary: dict = {"config": cfg, "certificates": certs, "sweep_best": sweep["best"],
                     "n_test": n_q, "conditions": {}}

    d_dec2 = mc2["v_dec"] - mc2["baseline"]
    d_dec1 = mc1["v_dec"] - mc1["baseline"]

    for cond in results:
        e2 = boot_mean(mc2[cond], rng)
        e1 = boot_mean(mc1[cond], rng)
        entry = {"mc2": e2[0], "mc2_ci": [e2[1], e2[2]],
                 "mc1": e1[0], "mc1_ci": [e1[1], e1[2]],
                 "delta_mc2": float((mc2[cond] - mc2["baseline"]).mean()),
                 "delta_mc1": float((mc1[cond] - mc1["baseline"]).mean())}
        if cond not in ("baseline", "v_dec"):
            num2 = mc2[cond] - mc2["baseline"]
            r2 = boot_ratio(num2, d_dec2, rng)
            entry["rho_mc2"] = r2[0]
            entry["rho_mc2_ci"] = [r2[1], r2[2]]
            num1 = mc1[cond] - mc1["baseline"]
            r1 = boot_ratio(num1, d_dec1, rng)
            entry["rho_mc1"] = r1[0]
            entry["rho_mc1_ci"] = [r1[1], r1[2]]
        summary["conditions"][cond] = entry

    # sigma_T across token-set constructions at comparable k
    sig_conds = ["v_perp_al64", "v_perp_cur", "v_perp_stat"]
    rhos = [summary["conditions"][c]["rho_mc2"] for c in sig_conds if c in summary["conditions"]]
    summary["sigma_T"] = float(np.std(rhos, ddof=1)) if len(rhos) > 1 else None
    summary["rho_values_for_sigma"] = dict(zip(sig_conds, rhos))

    # eta: curated / statistical / spillover readouts
    cur_p = list(tokensets["curated_plus"].values())
    cur_m = list(tokensets["curated_minus"].values())
    sp_p = list(tokensets["spill_plus"].values())
    sp_m = list(tokensets["spill_minus"].values())

    eta = {}
    for cond in results:
        eta[cond] = {
            "curated": eta_per_question(results[cond], capture, cur_p, cur_m),
            "spill": eta_per_question(results[cond], capture, sp_p, sp_m),
        }
    summary["eta"] = {}
    for readout in ("curated", "spill"):
        base = eta["baseline"][readout]
        d_dec = eta["v_dec"][readout] - base
        ent = {"delta_v_dec": float(np.nanmean(d_dec))}
        for cond in ("v_perp_al64", "v_perp_cur", "v_par_al64"):
            if cond in eta:
                d = eta[cond][readout] - base
                ok = ~np.isnan(d) & ~np.isnan(d_dec)
                r = boot_ratio(d[ok], d_dec[ok], rng)
                ent[f"rho_eta_{cond}"] = r[0]
                ent[f"rho_eta_{cond}_ci"] = [r[1], r[2]]
                ent[f"delta_{cond}"] = float(np.nanmean(d))
        summary["eta"][readout] = ent

    # paraphrase OOD
    para = {}
    for c in ("baseline", "v_dec", "v_perp_al64"):
        p = S3 / f"para_{c}.jsonl"
        if p.exists():
            para[c] = load_jsonl(p)
    if len(para) == 3:
        pm = paired_arrays(para, "mc2")
        d_dec_p = pm["v_dec"] - pm["baseline"]
        d_perp_p = pm["v_perp_al64"] - pm["baseline"]
        r = boot_ratio(d_perp_p, d_dec_p, rng)
        summary["paraphrase"] = {
            "n": len(para["baseline"]),
            "delta_v_dec": float(d_dec_p.mean()),
            "delta_v_perp_al64": float(d_perp_p.mean()),
            "rho_ood": r[0], "rho_ood_ci": [r[1], r[2]],
        }

    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2)[:2000])

    make_figures(summary, mc2, eta, cfg, certs, rng)
    make_tables(summary, certs)
    print("[stage4] complete")


def make_figures(summary, mc2, eta, cfg, certs, rng) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        **PAPER_FONT_RC,
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })

    # --- Figure 1: rho(k) curve ---
    ks = [16, 64, 256, 1024]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    pts, los, his = [], [], []
    for k in ks:
        e = summary["conditions"][f"v_perp_al{k}"]
        pts.append(e["rho_mc2"])
        los.append(e["rho_mc2_ci"][0])
        his.append(e["rho_mc2_ci"][1])
    ax.fill_between(ks, los, his, color=CAA_COLOR, alpha=0.18, lw=0)
    ax.plot(ks, pts, "o-", color=CAA_COLOR, label=r"$v^{\bot}$ aligned-$k$")
    # random projections at k=64
    rand = [summary["conditions"][f"v_rand_s{s}"]["rho_mc2"] for s in (0, 1, 2)
            if f"v_rand_s{s}" in summary["conditions"]]
    if rand:
        ax.scatter([64] * len(rand), rand, marker="x", s=40, color=PALETTE[3],
                   label="random 64-dim", zorder=5)
    # curated / statistical
    for cond, marker, color, lab in (
        ("v_perp_cur", "s", PALETTE[2], "curated"),
        ("v_perp_stat", "D", PALETTE[4], "statistical"),
    ):
        if cond in summary["conditions"]:
            e = summary["conditions"][cond]
            ax.scatter([certs[cond.replace("v_perp_", "")]["k"]], [e["rho_mc2"]],
                       marker=marker, s=40, color=color, label=lab, zorder=5)
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.axhline(0.0, ls=":", color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"tokens projected out ($k$)")
    ax.set_ylabel(r"depth statistic $\rho$ (MC2)")
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.savefig(FIG / "rho_curve.pdf")
    plt.close(fig)

    # --- Figure 2: MC2 per condition ---
    conds = ["baseline", "v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256",
             "v_perp_al1024", "v_perp_cur", "v_perp_stat", "v_perp_al64_nm",
             "v_par_al64", "v_rand_s0", "v_mm"]
    conds = [c for c in conds if c in summary["conditions"]]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    xs = np.arange(len(conds))
    vals = [summary["conditions"][c]["mc2"] for c in conds]
    errs = np.array([
        [summary["conditions"][c]["mc2"] - summary["conditions"][c]["mc2_ci"][0] for c in conds],
        [summary["conditions"][c]["mc2_ci"][1] - summary["conditions"][c]["mc2"] for c in conds],
    ])
    colors = [PALETTE[0] if c == "baseline" else
              MM_COLOR if c == "v_mm" else
              PALETTE[5] if c.startswith("v_rand") or c == "v_par_al64" else
              CAA_COLOR for c in conds]
    ax.bar(xs, vals, yerr=errs, color=colors, width=0.7, capsize=2,
           error_kw={"lw": 0.8})
    ax.axhline(summary["conditions"]["baseline"]["mc2"], ls=":", color="gray", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("TruthfulQA MC2")
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.04)
    fig.savefig(FIG / "mc2_conditions.pdf")
    plt.close(fig)

    # --- Figure 3: lens resynthesis trajectories ---
    lens_path = S3 / "lens.pt"
    if lens_path.exists():
        import torch
        lens = torch.load(lens_path, weights_only=True)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        base = lens["trajectories"]["baseline"].numpy()
        colmap = {
            "v_dec": FAMILY_SHADE["dec"]["dark"],
            "v_perp_al64": FAMILY_SHADE["dec"]["base"],
            "v_par_al64": FAMILY_SHADE["dec"]["mid"],
        }
        for name in ("v_dec", "v_perp_al64", "v_par_al64"):
            if name in lens["trajectories"]:
                tr = lens["trajectories"][name].numpy() - base
                m = tr.mean(axis=1)
                se = tr.std(axis=1) / np.sqrt(tr.shape[1])
                ls_ = np.arange(len(m))
                ax.plot(ls_, m, color=colmap[name], label=LABELS[name])
                ax.fill_between(ls_, m - 2 * se, m + 2 * se, color=colmap[name],
                                alpha=0.15, lw=0)
        ax.axvline(lens["layer_star"], ls="--", color="gray", lw=0.8)
        ax.text(lens["layer_star"] + 0.3, ax.get_ylim()[1] * 0.92,
                r"$\ell^{*}$", fontsize=8, color="gray")
        ax.axhline(0.0, ls=":", color="gray", lw=0.8)
        ax.set_xlabel("layer")
        ax.set_ylabel(r"honest-shift at $T$ (logit-lens, vs.\ baseline)")
        ax.legend(frameon=False, fontsize=7.5)
        fig.savefig(FIG / "lens_resynthesis.pdf")
        plt.close(fig)

    # --- Figure 4: per-question eta scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9), sharex=False)
    for ax_, readout, title in zip(axes, ("curated", "spill"),
                                   ("curated $T$ readout", "spillover (synonym) readout")):
        base = eta["baseline"][readout]
        x = eta["v_dec"][readout] - base
        y = eta["v_perp_al64"][readout] - base
        ok = ~np.isnan(x) & ~np.isnan(y)
        ax_.scatter(x[ok], y[ok], s=6, alpha=0.5, color=CAA_COLOR, edgecolors="none")
        lim = max(abs(x[ok]).max(), abs(y[ok]).max()) * 1.05
        ax_.plot([-lim, lim], [-lim, lim], ls=":", color="gray", lw=0.8)
        ax_.axhline(0, ls=":", color="gray", lw=0.5)
        ax_.axvline(0, ls=":", color="gray", lw=0.5)
        ax_.set_xlabel(r"$\eta$ shift under $v_{\mathrm{dec}}$")
        ax_.set_title(title, fontsize=8.5)
    axes[0].set_ylabel(r"$\eta$ shift under $v^{\bot}$ al-64")
    fig.savefig(FIG / "eta_scatter.pdf")
    plt.close(fig)


def make_tables(summary, certs) -> None:
    rows = []
    order = ["v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256", "v_perp_al1024",
             "v_perp_cur", "v_perp_stat", "v_perp_al64_nm", "v_par_al64",
             "v_rand_s0", "v_rand_s1", "v_rand_s2", "v_mm"]
    base = summary["conditions"]["baseline"]
    rows.append(
        f"baseline & --- & --- & {base['mc1']:.3f} & {base['mc2']:.3f} & --- & --- \\\\"
    )
    for cond in order:
        if cond not in summary["conditions"]:
            continue
        e = summary["conditions"][cond]
        cert_key = cond.replace("v_perp_", "").replace("v_rand_", "rand_")
        c = certs.get(cert_key, {})
        k = str(c.get("k", "---"))
        nr = f"{c['norm_ratio']:.3f}" if "norm_ratio" in c else "---"
        if "rho_mc2" in e:
            rho = f"{e['rho_mc2']:.2f} [{e['rho_mc2_ci'][0]:.2f}, {e['rho_mc2_ci'][1]:.2f}]"
        else:
            rho = "---"
        rows.append(
            f"{LABELS[cond]} & {k} & {nr} & {e['mc1']:.3f} & {e['mc2']:.3f} & "
            f"{e['delta_mc2']:+.3f} & {rho} \\\\"
        )
    (TAB / "main_results.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
