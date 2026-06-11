"""Stage 4 (recalibrated, Mac-side): aggregate the two-family depth results,
compute the discriminating statistics, and emit figures + tables.

The decisive test is rho(aligned) vs rho(random): if removing the
unembedding-aligned subspace kills the steering effect more than removing a
random subspace of equal rank, the effect is concentrated in the readout
(shallow); if the two are indistinguishable, the effect is not concentrated in
the readout (deep, or at least not vocabulary suppression). rho is only
interpretable because the recalibrated effect Delta(v_dec) is bounded away
from zero -- we report its CI and refuse to interpret rho otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RC = ROOT / "artifacts" / "recal"
RES = ROOT / "results" / "recal"
FIG = ROOT / "figures"
TAB = ROOT / "docs" / "paper" / "tables"

PALETTE = ["#63FBC5", "#00B6EB", "#8000FF", "#FF0000", "#FFB360", "#D6DD81"]
N_BOOT = 10_000
SEED = 0
FAM_LABEL = {"dec": "CAA", "mm": "mass-mean"}
ALIGNED_KS = [16, 64, 256, 1024]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return rows


def align(base: list[dict], cond: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pos = {r["idx"]: r[key] for r in cond}
    b, c = [], []
    for r in base:
        if r["idx"] in pos:
            b.append(r[key])
            c.append(pos[r["idx"]])
    return np.array(b, float), np.array(c, float)


def boot_mean(x, rng, n=N_BOOT):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(num, den, rng, n=N_BOOT):
    """Paired bootstrap CI for mean(num)/mean(den)."""
    k = len(num)
    idx = rng.integers(0, k, size=(n, k))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_diff(a_num, a_den, b_num, b_den, rng, n=N_BOOT):
    """Bootstrap CI for rho_a - rho_b (paired resample of question indices)."""
    k = len(a_num)
    idx = rng.integers(0, k, size=(n, k))
    ra = a_num[idx].mean(1) / a_den[idx].mean(1)
    rb = b_num[idx].mean(1) / b_den[idx].mean(1)
    d = ra - rb
    return float((a_num.mean() / a_den.mean()) - (b_num.mean() / b_den.mean())), \
        float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def eta_arr(rows, capture, plus, minus):
    posmap = {t: i for i, t in enumerate(capture)}
    pi = [posmap[t] for t in plus if t in posmap]
    mi = [posmap[t] for t in minus if t in posmap]
    out = {}
    for r in rows:
        el = r.get("eta_logits")
        if el is not None:
            el = np.asarray(el)
            out[r["idx"]] = float(el[pi].mean() - el[mi].mean())
    return out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cfg = json.loads((RC / "config.json").read_text())
    certs = json.loads((RC / "certificates.json").read_text())
    calib = json.loads((RC / "calibration.json").read_text())
    toksets = json.loads((RC / "tokensets.json").read_text())
    capture = json.loads((RC / "capture_ids.json").read_text())

    base = load_jsonl(RES / "baseline.jsonl")
    summary = {"config": cfg, "n_test": len(base), "families": {}}

    cur_p = list(toksets["curated_plus"].values())
    cur_m = list(toksets["curated_minus"].values())
    sp_p = list(toksets["spill_plus"].values())
    sp_m = list(toksets["spill_minus"].values())
    base_eta_cur = eta_arr(base, capture, cur_p, cur_m)
    base_eta_sp = eta_arr(base, capture, sp_p, sp_m)

    for fam in cfg["families"]:
        fdir = RES / fam
        v_dec = load_jsonl(fdir / "v_dec.jsonl")
        if not v_dec:
            print(f"[stage4r] {fam}: no results yet")
            continue
        fent = {"operating_point": cfg["families"][fam], "conditions": {}}

        # baseline + v_dec effect (the denominator)
        for metric in ("mc2", "mc1"):
            bb, vv = align(base, v_dec, metric)
            e = boot_mean(vv - bb, rng)
            fent[f"delta_vdec_{metric}"] = {"point": e[0], "ci": [e[1], e[2]]}
        b2, vdec2 = align(base, v_dec, "mc2")
        d_dec = vdec2 - b2

        # every projected/control condition
        for cond_file in sorted(fdir.glob("*.jsonl")):
            cond = cond_file.stem
            if cond.startswith("para_"):
                continue
            rows = load_jsonl(cond_file)
            if not rows:
                continue
            bb, cc = align(base, rows, "mc2")
            e = boot_mean(cc, rng)
            d_cond = cc - bb
            entry = {"mc2": e[0], "mc2_ci": [e[1], e[2]],
                     "delta_mc2": float(d_cond.mean())}
            if cond != "v_dec":
                r = boot_ratio(d_cond, d_dec, rng)
                entry["rho"] = r[0]
                entry["rho_ci"] = [r[1], r[2]]
            fent["conditions"][cond] = entry

        # decisive comparison: aligned-64 vs each random seed
        _, al = align(base, load_jsonl(fdir / "v_perp_al64.jsonl"), "mc2")
        d_al = al - b2
        rand_rhos = []
        for s in range(3):
            rr = load_jsonl(fdir / f"v_rand_s{s}.jsonl")
            if rr:
                _, rv = align(base, rr, "mc2")
                rand_rhos.append((rv - b2))
        if rand_rhos:
            d_rand = np.mean(rand_rhos, axis=0)
            diff = boot_diff(d_al, d_dec, d_rand, d_dec, rng)
            fent["aligned_minus_random_rho"] = {"point": diff[0], "ci": [diff[1], diff[2]]}

        # sigma_T across token-set constructions
        rhos_T = [fent["conditions"][c]["rho"] for c in ("v_perp_al64", "v_perp_cur", "v_perp_stat")
                  if c in fent["conditions"] and "rho" in fent["conditions"][c]]
        fent["sigma_T"] = float(np.std(rhos_T, ddof=1)) if len(rhos_T) > 1 else None
        fent["rho_by_tokenset"] = {c: fent["conditions"][c].get("rho")
                                   for c in ("v_perp_al64", "v_perp_cur", "v_perp_stat")
                                   if c in fent["conditions"]}

        # eta depth on curated and spillover readouts
        fent["eta"] = {}
        for ro, (bmap, plus, minus) in (("curated", (base_eta_cur, cur_p, cur_m)),
                                        ("spillover", (base_eta_sp, sp_p, sp_m))):
            dec_map = eta_arr(v_dec, capture, plus, minus)
            perp_rows = load_jsonl(fdir / "v_perp_al64.jsonl")
            perp_map = eta_arr(perp_rows, capture, plus, minus)
            ids = [r["idx"] for r in base if r["idx"] in dec_map and r["idx"] in perp_map and r["idx"] in bmap]
            d_dec_e = np.array([dec_map[i] - bmap[i] for i in ids])
            d_perp_e = np.array([perp_map[i] - bmap[i] for i in ids])
            ent = {"delta_vdec": float(d_dec_e.mean()), "delta_vperp": float(d_perp_e.mean())}
            if abs(d_dec_e.mean()) > 1e-6:
                r = boot_ratio(d_perp_e, d_dec_e, rng)
                ent["rho_eta"] = r[0]
                ent["rho_eta_ci"] = [r[1], r[2]]
            fent["eta"][ro] = ent

        # paraphrase OOD
        pbase = load_jsonl(RES / "para_baseline.jsonl")
        pdec = load_jsonl(fdir / "para_v_dec.jsonl")
        pperp = load_jsonl(fdir / "para_v_perp_al64.jsonl")
        if pbase and pdec and pperp:
            bb, dd = align(pbase, pdec, "mc2")
            _, pp = align(pbase, pperp, "mc2")
            dec_d = dd - bb
            perp_d = pp - bb
            ent = {"delta_vdec": float(dec_d.mean()), "delta_vperp": float(perp_d.mean())}
            if abs(dec_d.mean()) > 1e-6:
                r = boot_ratio(perp_d, dec_d, rng)
                ent["rho_ood"] = r[0]
                ent["rho_ood_ci"] = [r[1], r[2]]
            fent["paraphrase"] = ent

        summary["families"][fam] = fent
        print(f"[stage4r] {fam}: d_vdec_mc2={fent['delta_vdec_mc2']['point']:+.4f} "
              f"rho_al64={fent['conditions'].get('v_perp_al64', {}).get('rho')}")

    (ROOT / "results" / "summary_recal.json").write_text(json.dumps(summary, indent=2, default=float))
    make_figures(summary, calib, cfg, certs)
    make_tables(summary, certs, cfg)
    print("[stage4r] complete")


def make_figures(summary, calib, cfg, certs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})

    # calibration curve: ppl ratio + MC2 delta vs alpha, gate line
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, fam in zip(axes, ["dec", "mm"]):
        grid = [r for r in calib["grid"] if r["family"] == fam]
        layers = sorted({r["layer"] for r in grid})
        for li, layer in enumerate(layers):
            g = sorted([r for r in grid if r["layer"] == layer], key=lambda r: r["alpha"])
            al = [r["alpha"] for r in g]
            ax.plot(al, [r["nll_ratio"] for r in g], "o-", color=PALETTE[li],
                    label=f"L{layer} PPL", ms=3)
        ax.axhline(cfg["gate"], ls="--", color=PALETTE[3], lw=0.9)
        ax.text(al[0], cfg["gate"] + 0.03, "coherence gate", fontsize=6.5, color=PALETTE[3])
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(FAM_LABEL[fam], fontsize=9)
        ax.set_ylabel("held-out PPL ratio")
        ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(FIG / "calibration.pdf")
    plt.close(fig)

    # rho(k) for both families with random band
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for fi, fam in enumerate(summary["families"]):
        f = summary["families"][fam]
        pts, lo, hi = [], [], []
        for k in ALIGNED_KS:
            c = f["conditions"].get(f"v_perp_al{k}")
            if c and "rho" in c:
                pts.append(c["rho"]); lo.append(c["rho_ci"][0]); hi.append(c["rho_ci"][1])
            else:
                pts.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        col = PALETTE[1] if fam == "dec" else PALETTE[2]
        ax.plot(ALIGNED_KS, pts, "o-", color=col, label=f"{FAM_LABEL[fam]} $v^\\perp$ aligned")
        ax.fill_between(ALIGNED_KS, lo, hi, color=col, alpha=0.15, lw=0)
        rr = [f["conditions"][f"v_rand_s{s}"]["rho"] for s in range(3)
              if f"v_rand_s{s}" in f["conditions"]]
        if rr:
            ax.errorbar([64], [np.mean(rr)], yerr=[[np.mean(rr) - min(rr)], [max(rr) - np.mean(rr)]],
                        fmt="x", color=col, ms=7, capsize=3, alpha=0.8)
    ax.axhline(1, ls=":", color="gray", lw=0.8); ax.axhline(0, ls=":", color="gray", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks(ALIGNED_KS); ax.set_xticklabels(map(str, ALIGNED_KS))
    ax.set_xlabel(r"tokens projected out ($k$); $\times$ = random-64 control")
    ax.set_ylabel(r"depth statistic $\rho$ (MC2)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIG / "rho_curve.pdf")
    plt.close(fig)

    # lens resynthesis (CAA family if present)
    lens_path = RES / "lens.pt"
    if lens_path.exists():
        import torch
        lens = torch.load(lens_path, weights_only=True)
        fig, axes = plt.subplots(1, len(lens["families"]), figsize=(6.4, 2.8), squeeze=False)
        b = lens["baseline"].numpy()
        for ax, fam in zip(axes[0], lens["families"]):
            cols = {"v_dec": PALETTE[3], "v_perp_al64": PALETTE[1], "v_par_al64": PALETTE[5]}
            for name, tr in lens["families"][fam].items():
                d = tr.numpy() - b
                m = d.mean(1); se = d.std(1) / np.sqrt(d.shape[1])
                xs = np.arange(len(m))
                ax.plot(xs, m, color=cols.get(name, "gray"), label=name)
                ax.fill_between(xs, m - 2 * se, m + 2 * se, color=cols.get(name, "gray"), alpha=0.15, lw=0)
            ax.axvline(lens["layer_star"][fam], ls="--", color="gray", lw=0.8)
            ax.axhline(0, ls=":", color="gray", lw=0.6)
            ax.set_title(FAM_LABEL[fam], fontsize=9); ax.set_xlabel("layer")
        axes[0][0].set_ylabel(r"honest-shift at $T$ vs.\ baseline")
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.savefig(FIG / "lens_resynthesis.pdf")
        plt.close(fig)


def make_tables(summary, certs, cfg):
    order = ["v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256", "v_perp_al1024",
             "v_perp_cur", "v_perp_stat", "v_par_al64", "v_perp_al64_nm",
             "v_rand_s0", "v_rand_s1", "v_rand_s2"]
    lab = {"v_dec": r"$v_{\mathrm{dec}}$", "v_perp_al16": r"$v^{\perp}$ al-16",
           "v_perp_al64": r"$v^{\perp}$ al-64", "v_perp_al256": r"$v^{\perp}$ al-256",
           "v_perp_al1024": r"$v^{\perp}$ al-1024", "v_perp_cur": r"$v^{\perp}$ cur",
           "v_perp_stat": r"$v^{\perp}$ stat", "v_par_al64": r"$v^{\parallel}$ al-64",
           "v_perp_al64_nm": r"$v^{\perp}$ al-64 nm", "v_rand_s0": "rand s0",
           "v_rand_s1": "rand s1", "v_rand_s2": "rand s2"}
    for fam, f in summary["families"].items():
        lines = []
        for cond in order:
            c = f["conditions"].get(cond)
            if not c:
                continue
            rho = (f"{c['rho']:.2f} [{c['rho_ci'][0]:.2f}, {c['rho_ci'][1]:.2f}]"
                   if "rho" in c else "---")
            lines.append(f"{lab[cond]} & {c['mc2']:.3f} & {c['delta_mc2']:+.3f} & {rho} \\\\")
        (TAB / f"results_{fam}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
