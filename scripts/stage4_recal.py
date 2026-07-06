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

from liar.plotting import (  # noqa: E402
    FAMILY_COLOR,
    FAMILY_MARKER,
    GRID,
    INK,
    MUTED,
    NEUTRAL,
    TURBO,
)

RC = ROOT / "artifacts" / "recal"
RES = ROOT / "results" / "recal"
FIG = ROOT / "figures"
TAB = ROOT / "docs" / "tables"

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
    assert len(num) == len(den), f"unpaired ratio inputs: {len(num)} vs {len(den)}"
    k = len(num)
    idx = rng.integers(0, k, size=(n, k))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_diff(a_num, a_den, b_num, b_den, rng, n=N_BOOT):
    """Bootstrap CI for rho_a - rho_b (paired resample of question indices)."""
    assert len(a_num) == len(a_den) == len(b_num) == len(b_den), "unpaired diff inputs"
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
            e_den = boot_mean(d_dec_e, rng)
            ent = {"delta_vdec": float(d_dec_e.mean()), "delta_vdec_ci": [e_den[1], e_den[2]],
                   "delta_vperp": float(d_perp_e.mean())}
            # interpret the ratio only when its denominator CI excludes zero
            if e_den[1] > 0 or e_den[2] < 0:
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
            e_dec = boot_mean(dec_d, rng)
            e_perp = boot_mean(perp_d, rng)
            ent = {"delta_vdec": float(dec_d.mean()), "delta_vdec_ci": [e_dec[1], e_dec[2]],
                   "delta_vperp": float(perp_d.mean()), "delta_vperp_ci": [e_perp[1], e_perp[2]]}
            # matched-subset comparison: in-distribution effect on the same questions.
            # dec_d is in pbase order but only over pbase rows present in pdec; key by
            # exactly those kept idx (not all of pbase) so a dropped row cannot silently
            # misalign the (idx, delta) pairing, and restrict the matched set to ids
            # present in baseline, v_dec, AND the paraphrase delta so id and ood pair up.
            b_map = {r["idx"]: r["mc2"] for r in base}
            vd_map = {r["idx"]: r["mc2"] for r in v_dec}
            pdec_ids = {r["idx"] for r in pdec}
            kept_idx = [r["idx"] for r in pbase if r["idx"] in pdec_ids]
            ood_map = {i: d for i, d in zip(kept_idx, dec_d)}
            common = [r["idx"] for r in pbase if r["idx"] in b_map and r["idx"] in vd_map and r["idx"] in ood_map]
            id_d = np.array([vd_map[i] - b_map[i] for i in common])
            ood_sub = np.array([ood_map[i] for i in common])
            e_id = boot_mean(id_d, rng)
            e_pair = boot_mean(ood_sub - id_d, rng)
            ent["matched_id_delta"] = float(id_d.mean())
            ent["matched_id_ci"] = [e_id[1], e_id[2]]
            ent["paired_diff"] = float((ood_sub - id_d).mean())
            ent["paired_diff_ci"] = [e_pair[1], e_pair[2]]
            if e_dec[1] > 0 or e_dec[2] < 0:
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
    make_numbers(summary, calib, cfg, certs, rng)
    make_appendix_tables(summary, certs, cfg, calib)
    make_caagrid_table(summary, cfg)
    print("[stage4r] complete")


FAM_COLOR = FAMILY_COLOR
COL_PAR = TURBO["parallel"]  # warm end of Turbo: parallel component
COL_RAND = TURBO["anchor"]   # subordinate via alpha + open-diamond marker
COL_GATE = TURBO["gate"]     # deep red end of Turbo: incoherent region
COL_VDEC = TURBO["anchor"]   # violet start of Turbo: unprojected vector
COL_PERP = TURBO["perp"]     # bright-blue Turbo sample: projected vector

CONDITION_ROWS = [
    ("v_dec", r"$v$ (unprojected)"),
    ("v_perp_al16", r"$v^{\perp}$ aligned-16"),
    ("v_perp_al64", r"$v^{\perp}$ aligned-64"),
    ("v_perp_al256", r"$v^{\perp}$ aligned-256"),
    ("v_perp_al1024", r"$v^{\perp}$ aligned-1024"),
    ("v_perp_cur", r"$v^{\perp}$ curated"),
    ("v_perp_stat", r"$v^{\perp}$ statistical"),
    ("v_perp_al64_nm", r"$v^{\perp}$ al-64, norm-matched"),
    ("v_par_al64", r"$v^{\parallel}$ aligned-64"),
    ("v_rand_s0", "random-64, seed 0"),
    ("v_rand_s1", "random-64, seed 1"),
    ("v_rand_s2", "random-64, seed 2"),
]


def _delta_ci(base, rows, rng, key="mc2"):
    bb, cc = align(base, rows, key)
    d = cc - bb
    return boot_mean(d, rng)


def make_figures(summary, calib, cfg, certs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.bbox": "tight",
        "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.72, "grid.linewidth": 0.5,
        "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    })
    rng = np.random.default_rng(SEED)
    base = load_jsonl(RES / "baseline.jsonl")
    fams = list(summary["families"].keys())

    # ---- Figure: calibration trade-off (PPL gate, top; val MC2 delta, bottom) ----
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
    for ci_, fam in enumerate(["dec", "mm"]):
        grid = [r for r in calib["grid"] if r["family"] == fam]
        layers = sorted({r["layer"] for r in grid})
        op = cfg["families"][fam]
        ax_p, ax_m = axes[0][ci_], axes[1][ci_]
        for layer in layers:
            g = sorted([r for r in grid if r["layer"] == layer], key=lambda r: r["alpha"])
            al = [r["alpha"] for r in g]
            col = FAM_COLOR[fam]
            is_selected_layer = layer == op["layer"]
            ls = "-" if is_selected_layer else "--"
            marker_face = col if is_selected_layer else "white"
            line_width = 1.6 if is_selected_layer else 1.2
            ax_p.plot(al, [r["nll_ratio"] for r in g], ls, marker="o", ms=3.5,
                      color=col, markerfacecolor=marker_face, markeredgecolor=col,
                      markeredgewidth=0.8, alpha=1.0, lw=line_width,
                      label=f"layer {layer}")
            ax_m.plot(al, [r["val_mc2_delta"] for r in g], ls, marker="o", ms=3.5,
                      color=col, markerfacecolor=marker_face, markeredgecolor=col,
                      markeredgewidth=0.8, alpha=1.0, lw=line_width)
            for r in g:
                if not r["coherent"]:
                    ax_m.plot(r["alpha"], r["val_mc2_delta"], "x", color=COL_GATE,
                              ms=7, mew=1.6, zorder=5)
        ax_p.set_yscale("log")
        ax_p.axhline(cfg["gate"], ls=":", color=COL_GATE, lw=1.2)
        ax_p.axhspan(cfg["gate"], ax_p.get_ylim()[1] * 20, color=COL_GATE, alpha=0.06, lw=0)
        ax_p.text(0.55, cfg["gate"] * 1.12, "incoherent (gate 1.5)", fontsize=6.8, color=COL_GATE)
        opr = [r for r in grid if r["layer"] == op["layer"] and r["alpha"] == op["alpha"]][0]
        ax_m.plot(op["alpha"], opr["val_mc2_delta"], "*", color=TURBO["selected"],
                  mec=COL_VDEC, mew=0.9, ms=13, zorder=6,
                  label="selected operating point")
        ax_m.axhline(0, ls=":", color=COL_RAND, alpha=0.45, lw=0.8)
        ax_p.set_title(FAM_LABEL[fam])
        ax_m.set_xlabel(r"steering strength $\alpha$")
        if ci_ == 0:
            ax_p.set_ylabel("held-out PPL ratio (log)")
            ax_m.set_ylabel(r"validation $\Delta$MC2")
        ax_p.legend(fontsize=7, loc="upper left")
        if ci_ == 0:
            ax_m.legend(fontsize=7, loc="upper left")
    fig.align_ylabels()
    fig.savefig(FIG / "calibration.pdf")
    plt.close(fig)

    # ---- Figure: forest plot of paired test-set deltas with 95% CIs ----
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5), sharey=True)
    ylocs = np.arange(len(CONDITION_ROWS))[::-1]
    for ci_, fam in enumerate(fams):
        ax = axes[ci_]
        fdir = RES / fam
        for (cond, lab), y in zip(CONDITION_ROWS, ylocs):
            rows = load_jsonl(fdir / f"{cond}.jsonl")
            if not rows:
                continue
            m, lo, hi = _delta_ci(base, rows, rng)
            if cond == "v_dec":
                col, mfc, ms_, marker, alpha = COL_VDEC, COL_VDEC, 5.5, "o", 1.0
            elif cond.startswith("v_rand"):
                col, mfc, ms_, marker, alpha = COL_RAND, "white", 4.7, "D", 0.55
            elif cond == "v_par_al64":
                col, mfc, ms_, marker, alpha = COL_PAR, COL_PAR, 5.2, "^", 1.0
            else:
                col, mfc, ms_, marker, alpha = FAM_COLOR[fam], "white", 5, "o", 1.0
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=marker, color=col,
                        mfc=mfc, mew=1.3, ms=ms_, capsize=2.4, lw=1.3, alpha=alpha)
        ax.axvline(0, ls=":", color=COL_RAND, alpha=0.45, lw=1.0)
        ax.set_title(FAM_LABEL[fam])
        ax.set_xlabel(r"$\Delta$MC2 vs. baseline (95% CI)")
    axes[0].set_yticks(ylocs)
    axes[0].set_yticklabels([lab for _, lab in CONDITION_ROWS], fontsize=8)
    axes[0].grid(axis="y", alpha=0)
    axes[1].grid(axis="y", alpha=0)
    fig.savefig(FIG / "forest.pdf")
    plt.close(fig)

    # ---- Figure: rho(k) per family with random-control band ----
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, fam in zip(axes, fams):
        f = summary["families"][fam]
        col = FAM_COLOR[fam]
        pts, lo, hi = [], [], []
        for k in ALIGNED_KS:
            c = f["conditions"].get(f"v_perp_al{k}", {})
            pts.append(c.get("rho", np.nan))
            lo.append(c.get("rho_ci", [np.nan, np.nan])[0])
            hi.append(c.get("rho_ci", [np.nan, np.nan])[1])
        ax.plot(ALIGNED_KS, pts, color=col, marker=FAMILY_MARKER[fam], ls="-",
                ms=4.5, lw=1.5, zorder=4,
                label=r"$v^{\perp}$ aligned-$k$")
        ax.fill_between(ALIGNED_KS, lo, hi, color=col, alpha=0.16, lw=0)
        rr = [f["conditions"][f"v_rand_s{s}"]["rho"] for s in range(3)
              if f"v_rand_s{s}" in f["conditions"]]
        if rr:
            ax.errorbar([64], [np.mean(rr)],
                        yerr=[[np.mean(rr) - min(rr)], [max(rr) - np.mean(rr)]],
                        fmt="D", color=COL_RAND, mfc="white", alpha=0.55,
                        ms=5, capsize=3, lw=1.2, zorder=5,
                        label="random-64 (3 seeds)")
        ax.axhline(1, ls=":", color=COL_RAND, alpha=0.45, lw=0.9)
        ax.axhline(0, ls=":", color=COL_RAND, alpha=0.45, lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks(ALIGNED_KS)
        ax.set_xticklabels([str(k) for k in ALIGNED_KS])
        ax.minorticks_off()
        ax.set_xlabel(r"directions projected out ($k$)")
        ax.set_title(FAM_LABEL[fam])
        if fam == "dec":
            ax.set_ylabel(r"depth statistic $\rho$ (MC2)")
            ax.text(0.5, 0.06, "denominator not significant:\n$\\rho$ not interpretable",
                    transform=ax.transAxes, fontsize=7.2, color=INK, ha="center")
        ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "rho_curve.pdf")
    plt.close(fig)

    # ---- Figure: component decomposition + paraphrase OOD, with CIs ----
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axes[0]
    comps = [("v_dec", r"$v$", ""), ("v_perp_al64", r"$v^{\perp}$", "//"),
             ("v_par_al64", r"$v^{\parallel}$", "xx")]
    width = 0.34
    for fi, fam in enumerate(fams):
        fdir = RES / fam
        xs = np.arange(len(comps)) + (fi - 0.5) * width
        for (cond, _, hatch), x in zip(comps, xs):
            rows = load_jsonl(fdir / f"{cond}.jsonl")
            if not rows:
                continue
            m, lo, hi = _delta_ci(base, rows, rng)
            col = FAM_COLOR[fam]
            ax.bar(x, m, width=width * 0.92, color=col, hatch=hatch,
                   edgecolor=INK, linewidth=0.35, alpha=0.88,
                   label=FAM_LABEL[fam] if cond == "v_dec" else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK,
                        capsize=2.5, lw=1.1)
    ax.axhline(0, color=COL_RAND, alpha=0.45, lw=0.8)
    ax.set_xticks(np.arange(len(comps)))
    ax.set_xticklabels([lab for _, lab, _ in comps])
    ax.set_ylabel(r"$\Delta$MC2 (95% CI)")
    ax.set_title("component decomposition (aligned-64)")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    pbase = load_jsonl(RES / "para_baseline.jsonl")
    xs, labels = [], []
    xpos = 0
    for fam in fams:
        for cond, lab in (("para_v_dec", r"$v$"), ("para_v_perp_al64", r"$v^{\perp}$")):
            rows = load_jsonl(RES / fam / f"{cond}.jsonl")
            if not rows or not pbase:
                continue
            m, lo, hi = _delta_ci(pbase, rows, rng)
            ax.bar(xpos, m, width=0.7, color=FAM_COLOR[fam],
                   hatch="" if cond == "para_v_dec" else "//",
                   edgecolor=INK, linewidth=0.35, alpha=0.88)
            ax.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt="none",
                        ecolor=INK, capsize=2.5, lw=1.1)
            labels.append(lab)
            xs.append(xpos)
            xpos += 1
        xpos += 0.7
    ax.axhline(0, color=COL_RAND, alpha=0.45, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    for fi, fam in enumerate(fams):
        ax.text(fi * 2.7 + 0.5, 1.0, FAM_LABEL[fam], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK, clip_on=False)
    ax.set_ylabel(r"$\Delta$MC2 on paraphrases (95% CI)")
    ax.set_title("out-of-distribution (paraphrases)", pad=16)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(FIG / "decomposition.pdf")
    plt.close(fig)

    # ---- Figure: per-question scatter, mass-mean family ----
    if "mm" in fams:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True, sharex=True)
        bmap = {r["idx"]: r["mc2"] for r in base}
        # Both panels are one family, so color encodes intervention condition.
        for ax, cond, lab, col in (
            (axes[0], "v_dec", r"mass-mean $v$ (unprojected)", COL_VDEC),
            (axes[1], "v_perp_al64", r"mass-mean $v^{\perp}$ aligned-64", COL_PERP),
        ):
            rows = load_jsonl(RES / "mm" / f"{cond}.jsonl")
            xs = np.array([bmap[r["idx"]] for r in rows if r["idx"] in bmap])
            ys = np.array([r["mc2"] for r in rows if r["idx"] in bmap])
            ax.plot([0, 1], [0, 1], color=COL_RAND, alpha=0.45, lw=0.9, ls=":", zorder=1)
            if cond == "v_dec":
                ax.scatter(xs, ys, s=6, color=col, alpha=0.48, lw=0, zorder=2)
            else:
                ax.scatter(xs, ys, s=8, facecolors="none", edgecolors=col,
                           alpha=1.0, linewidths=0.55, zorder=2)
            d = ys - xs
            ax.text(0.03, 0.93, f"mean $\\Delta$MC2 = {d.mean():+.3f}\n"
                    f"{(d > 0).mean() * 100:.0f}% of questions improve",
                    transform=ax.transAxes, fontsize=7.5, va="top", zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.88, "pad": 1.4})
            ax.set_xlabel("baseline MC2 (per question)")
            ax.set_title(lab, fontsize=8.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
        axes[0].set_ylabel("steered MC2 (per question)")
        fig.savefig(FIG / "scatter_perq.pdf")
        plt.close(fig)

    # ---- Figure: lens re-synthesis trajectories ----
    lens_path = RES / "lens.pt"
    if lens_path.exists():
        import torch
        lens = torch.load(lens_path, weights_only=True)
        fig, axes = plt.subplots(1, len(lens["families"]), figsize=(6.6, 2.9), squeeze=False)
        b = lens["baseline"].numpy()
        cols = {"v_dec": COL_VDEC, "v_perp_al64": COL_PERP, "v_par_al64": COL_PAR}
        styles = {"v_dec": "-", "v_perp_al64": "--", "v_par_al64": "-."}
        labs = {"v_dec": r"$v$", "v_perp_al64": r"$v^{\perp}$",
                "v_par_al64": r"$v^{\parallel}$"}
        for ax, fam in zip(axes[0], lens["families"]):
            for name, tr in lens["families"][fam].items():
                d = tr.numpy() - b
                m = d.mean(1)
                se = d.std(1) / np.sqrt(d.shape[1])
                xs = np.arange(len(m))
                ax.plot(xs, m, color=cols.get(name, NEUTRAL), ls=styles.get(name, "-"),
                        lw=1.7, label=labs.get(name, name))
                ax.fill_between(xs, m - 2 * se, m + 2 * se, color=cols.get(name, NEUTRAL),
                                alpha=0.16, lw=0)
            ls_ = lens["layer_star"][fam]
            ax.axvline(ls_, ls="--", color=COL_RAND, alpha=0.45, lw=0.9)
            ax.text(ls_ + 0.4, ax.get_ylim()[1] * 0.92, r"$\ell^{*}$", fontsize=8, color=MUTED)
            ax.axhline(0, ls=":", color=COL_RAND, alpha=0.45, lw=0.7)
            ax.set_title(FAM_LABEL[fam])
            ax.set_xlabel("layer")
        axes[0][0].set_ylabel("honest-shift at $T$ (logits)\nrelative to baseline")
        axes[0][0].legend(fontsize=7, loc="upper left")
        fig.savefig(FIG / "lens_resynthesis.pdf")
        plt.close(fig)

    # ---- Appendix figure: word-shift (eta) decomposition per readout ----
    eta_data = {fam: summary["families"][fam].get("eta", {}) for fam in fams}
    if any(eta_data.values()):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=False)
        for ax, ro, title in ((axes[0], "curated", "curated readout (projected out)"),
                              (axes[1], "spillover", "spillover readout (never projected)")):
            xlocs = np.arange(len(fams))
            w = 0.36
            for j, comp in enumerate(("delta_vdec", "delta_vperp")):
                vals = [eta_data[fam].get(ro, {}).get(comp, np.nan) for fam in fams]
                cols_ = [FAM_COLOR[fam] for fam in fams]
                ax.bar(xlocs + (j - 0.5) * w, vals, width=w * 0.92,
                       color=cols_, hatch="" if comp == "delta_vdec" else "//",
                       edgecolor=INK, linewidth=0.35, alpha=0.88,
                       label=(r"$v$" if comp == "delta_vdec" else r"$v^{\perp}$"))
            ax.axhline(0, color=COL_RAND, alpha=0.45, lw=0.8)
            ax.set_xticks(xlocs)
            ax.set_xticklabels([FAM_LABEL[fam] for fam in fams])
            ax.set_title(title, fontsize=8.5)
            ax.set_ylabel(r"$\Delta\eta$ (logits)")
        axes[0].legend(fontsize=7, title="injected", title_fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "eta_decomposition.pdf")
        plt.close(fig)


def _fmt(x, nd=3, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$") if sign else s


def _ci(lo, hi, nd=3):
    return f"[{_fmt(lo, nd, sign=True)}, {_fmt(hi, nd, sign=True)}]"


def make_numbers(summary, calib, cfg, certs, rng):
    """Single source of truth: every headline number in the paper is a macro
    generated here from the artifacts. numbers.tex is \\input by main.tex."""
    L = []

    def cmd(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # --- certificates ---
    worst = max(c["max_direct_after"] for c in certs.values() if "max_direct_after" in c)
    pre = max(c["max_direct_before"] for c in certs.values() if "max_direct_before" in c)
    e = int(np.floor(np.log10(worst)))
    m = int(np.ceil(worst / 10 ** e))
    if m == 10:
        m, e = 1, e + 1
    cmd("CertWorst", f"${m} \\times 10^{{{e}}}$")
    cmd("CertPre", _fmt(pre, 2))

    # --- recal per family ---
    fam_macros = {"dec": "Dec", "mm": "Mm"}
    for fam, F in fam_macros.items():
        f = summary["families"].get(fam)
        if not f:
            continue
        op = f["operating_point"]
        cmd(f"{F}Layer", op["layer"])
        cmd(f"{F}Alpha", f"{op['alpha']:g}")
        cmd(f"{F}PplRatio", _fmt(op["ppl_ratio"], 2))
        cmd(f"{F}ValDelta", _fmt(op["val_mc2_delta"], 3, sign=True))
        d = f["delta_vdec_mc2"]
        cmd(f"{F}TestDelta", _fmt(d["point"], 3, sign=True))
        cmd(f"{F}TestDeltaCI", _ci(*d["ci"]))
        d1 = f["delta_vdec_mc1"]
        cmd(f"{F}TestDeltaMcOne", _fmt(d1["point"], 3, sign=True))
        cmd(f"{F}TestDeltaMcOneCI", _ci(*d1["ci"]))
        # fraction of test questions whose paired Delta MC2 improves (prose SSOT;
        # same quantity annotated on the per-question scatter figure)
        bfi, vfi = align(load_jsonl(RES / "baseline.jsonl"),
                         load_jsonl(RES / fam / "v_dec.jsonl"), "mc2")
        if len(bfi):
            cmd(f"{F}FracImproving", f"{(vfi - bfi > 0).mean() * 100:.0f}\\%")
        for cond, tag in (("v_perp_al64", "RhoAlSixtyFour"), ("v_perp_cur", "RhoCur"),
                          ("v_perp_stat", "RhoStat"), ("v_perp_al64_nm", "RhoNm"),
                          ("v_perp_al1024", "RhoAlBig"), ("v_par_al64", "RhoPar")):
            c = f["conditions"].get(cond, {})
            if "rho" in c:
                cmd(f"{F}{tag}", _fmt(c["rho"], 2))
                cmd(f"{F}{tag}CI", _ci(*c["rho_ci"], nd=2))
        c = f["conditions"].get("v_par_al64", {})
        if c:
            cmd(f"{F}ParDelta", _fmt(c["delta_mc2"], 3, sign=True))
        c = f["conditions"].get("v_perp_al64", {})
        if c:
            cmd(f"{F}PerpDelta", _fmt(c["delta_mc2"], 3, sign=True))
        krhos = [f["conditions"][f"v_perp_al{k}"]["rho"] for k in ALIGNED_KS
                 if f"v_perp_al{k}" in f["conditions"] and "rho" in f["conditions"][f"v_perp_al{k}"]]
        if krhos:
            cmd(f"{F}RhoKMin", _fmt(min(krhos), 2))
            klos = [f["conditions"][f"v_perp_al{k}"]["rho_ci"][0] for k in ALIGNED_KS
                    if f"v_perp_al{k}" in f["conditions"] and "rho_ci" in f["conditions"][f"v_perp_al{k}"]]
            if klos:
                cmd(f"{F}RhoKCiFloor", _fmt(min(klos), 2))
        rr = [f["conditions"][f"v_rand_s{s}"]["rho"] for s in range(3)
              if f"v_rand_s{s}" in f["conditions"] and "rho" in f["conditions"].get(f"v_rand_s{s}", {})]
        if rr:
            cmd(f"{F}RhoRandMean", _fmt(float(np.mean(rr)), 2))
            cmd(f"{F}RhoRandMin", _fmt(min(rr), 2))
            cmd(f"{F}RhoRandMax", _fmt(max(rr), 2))
        amr = f.get("aligned_minus_random_rho")
        if amr:
            cmd(f"{F}AlignedMinusRandom", _fmt(amr["point"], 2, sign=True))
            cmd(f"{F}AlignedMinusRandomCI", _ci(*amr["ci"], nd=2))
        if f.get("sigma_T") is not None:
            cmd(f"{F}SigmaT", _fmt(f["sigma_T"], 3))
        for ro, tag in (("curated", "EtaCur"), ("spillover", "EtaSpill")):
            e = f.get("eta", {}).get(ro, {})
            if "rho_eta" in e:
                cmd(f"{F}{tag}Rho", _fmt(e["rho_eta"], 2))
                cmd(f"{F}{tag}RhoCI", _ci(*e["rho_eta_ci"], nd=2))
            if "delta_vdec" in e:
                cmd(f"{F}{tag}DeltaVdec", _fmt(e["delta_vdec"], 2, sign=True))
                cmd(f"{F}{tag}DeltaVperp", _fmt(e["delta_vperp"], 2, sign=True))
        p = f.get("paraphrase", {})
        if "rho_ood" in p:
            cmd(f"{F}RhoOod", _fmt(p["rho_ood"], 2))
            cmd(f"{F}RhoOodCI", _ci(*p["rho_ood_ci"], nd=2))
        if "delta_vdec" in p:
            cmd(f"{F}OodDeltaVdec", _fmt(p["delta_vdec"], 3, sign=True))
            cmd(f"{F}OodDeltaVperp", _fmt(p["delta_vperp"], 3, sign=True))
        if "delta_vdec_ci" in p:
            cmd(f"{F}OodDeltaVdecCI", _ci(*p["delta_vdec_ci"]))
        if "matched_id_delta" in p:
            cmd(f"{F}OodMatchedDelta", _fmt(p["matched_id_delta"], 3, sign=True))
            cmd(f"{F}OodMatchedDeltaCI", _ci(*p["matched_id_ci"]))
            cmd(f"{F}OodPairedDiff", _fmt(p["paired_diff"], 3, sign=True))
            cmd(f"{F}OodPairedDiffCI", _ci(*p["paired_diff_ci"]))

    # --- baseline ---
    base = load_jsonl(RES / "baseline.jsonl")
    if base:
        cmd("BaseMcTwo", _fmt(float(np.mean([r["mc2"] for r in base])), 3))
        cmd("BaseMcOne", _fmt(float(np.mean([r["mc1"] for r in base])), 3))
        cmd("NTest", len(base))

    # --- naive operating point (old run, broken instrument) ---
    old = ROOT / "results" / "stage2_old"
    ob = load_jsonl(old / "baseline.jsonl")
    od = load_jsonl(old / "v_dec.jsonl")
    if ob and od:
        for key, tag in (("mc2", "NaiveDecDelta"), ("mc1", "NaiveDecDeltaMcOne")):
            bb, vv = align(ob, od, key)
            e = boot_mean(vv - bb, rng)
            cmd(tag, _fmt(e[0], 3, sign=True))
            cmd(f"{tag}CI", _ci(e[1], e[2]))
        b2, v2 = align(ob, od, "mc2")
        d_dec = v2 - b2
        for cond, tag in (("v_perp_al64", "NaiveRhoAlSixtyFour"),):
            rows = load_jsonl(old / f"{cond}.jsonl")
            if rows:
                _, cc = align(ob, rows, "mc2")
                r = boot_ratio(cc - b2, d_dec, rng)
                cmd(tag, _fmt(r[0], 2))
                cmd(f"{tag}CI", _ci(r[1], r[2], nd=2))
        rr = []
        for s in range(3):
            rows = load_jsonl(old / f"v_rand_s{s}.jsonl")
            if rows:
                _, cc = align(ob, rows, "mc2")
                rr.append(float((cc - b2).mean() / d_dec.mean()))
        if rr:
            cmd("NaiveRhoRandMean", _fmt(float(np.mean(rr)), 2))

    # --- probe (broken-instrument physical evidence) ---
    probe_path = RES / "probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        pc = probe["conditions"]
        if "naive/dec" in pc:
            cmd("NaiveNormRatio", _fmt(pc["naive/dec"]["norm_ratio_to_median_h"], 2))
            cmd("NaivePplRatio", _fmt(pc["naive/dec"]["ppl_ratio"], 2))
            cmd("NaiveLayer", pc["naive/dec"]["layer"])
            cmd("NaiveAlpha", f"{pc['naive/dec']['alpha']:g}")
        if "gated/dec" in pc:
            cmd("DecNormRatio", _fmt(pc["gated/dec"]["norm_ratio_to_median_h"], 2))
        if "gated/mm" in pc:
            cmd("MmNormRatio", _fmt(pc["gated/mm"]["norm_ratio_to_median_h"], 2))

    # --- per-point leakage + BPE boundary check (verify_leakage_bpe.py) ---
    vpath = RES / "verify.json"
    if vpath.exists():
        ver = json.loads(vpath.read_text())
        b = ver.get("bpe_check", {})
        if b:
            cmd("BpePairs", f"{b['pairs']:,}".replace(",", "{,}"))
            cmd("BpeMismatches", b["mismatches"])
        for fam, F in fam_macros.items():
            lk = ver.get("leakage", {}).get(fam)
            if lk:
                cmd(f"{F}LeakMax", _fmt(lk["leak_max"], 3))
                cmd(f"{F}LeakMedian", _fmt(lk["leak_median"], 3))
                cmd(f"{F}LeakScale", _fmt(lk["scale_median_vdec"], 2))

    # --- MC1-based robustness of the mm depth statistic ---
    mm_dec = load_jsonl(RES / "mm" / "v_dec.jsonl")
    mm_perp = load_jsonl(RES / "mm" / "v_perp_al64.jsonl")
    if base and mm_dec and mm_perp:
        b1, v1 = align(base, mm_dec, "mc1")
        d_dec1 = v1 - b1
        _, c1 = align(base, mm_perp, "mc1")
        r = boot_ratio(c1 - b1, d_dec1, rng)
        cmd("MmRhoAlSixtyFourMcOne", _fmt(r[0], 2))
        cmd("MmRhoAlSixtyFourMcOneCI", _ci(r[1], r[2], nd=2))

    # --- CAA coherent-grid sweep (stage5_extra dec_grid): largest test gain at
    #     any coherent operating point, to license the "no gain anywhere" claim ---
    import glob as _glob
    grid_files = sorted(_glob.glob(str(RES / "dec_grid" / "*.jsonl")))
    if base and grid_files:
        best = None
        n_coh = 0
        # Use a fresh rng in the SAME sorted order as make_caagrid_table so the
        # grid-max row's bootstrap CI here is identical to the one in app_caagrid.tex
        # (otherwise the same +0.013 row carries two slightly different intervals).
        gridrng = np.random.default_rng(SEED)
        for f in grid_files:
            g = load_jsonl(Path(f))
            if not g:
                continue
            n_coh += 1
            bb, gg = align(base, g, "mc2")
            e = boot_mean(gg - bb, gridrng)
            if best is None or e[0] > best[0]:
                best = e
        # include the gated OP itself among coherent settings
        opd = load_jsonl(RES / "dec" / "v_dec.jsonl")
        if opd:
            bb, gg = align(base, opd, "mc2")
            e = boot_mean(gg - bb, rng)
            n_coh += 1
            if best is None or e[0] > best[0]:
                best = e
        if best is not None:
            cmd("CaaGridNCoherent", n_coh)
            cmd("CaaGridMaxDelta", _fmt(best[0], 3, sign=True))
            cmd("CaaGridMaxDeltaCI", _ci(best[1], best[2]))

    # --- nonlinear direct-path leakage (nonlinear_directpath.py) ---
    nlp = RES / "nonlinear.json"
    if nlp.exists():
        nl = json.loads(nlp.read_text())
        for fam, F in fam_macros.items():
            e = nl.get("families", {}).get(fam)
            if e:
                cmd(f"{F}NonlinVdecMax", _fmt(e["v_dec_exact_direct_max"], 3))
                cmd(f"{F}NonlinVperpMax", _fmt(e["v_perp_al64_exact_direct_max"], 3))
                cmd(f"{F}NonlinVperpMean", _fmt(e["v_perp_al64_exact_direct_mean_abs"], 4))

    # --- free-generation truthfulness (free_generation.py) ---
    fg = RES / "freegen.jsonl"
    if fg.exists():
        recs = load_jsonl(fg)
        if recs:
            def fgmean(key):
                vals = [r[key] for r in recs if key in r]
                return boot_mean(np.array(vals), rng) if vals else None
            cmd("FreeGenN", len(recs))
            b = fgmean("score_baseline")
            if b:
                cmd("FreeGenBase", _fmt(b[0], 3))
            for fam, F in fam_macros.items():
                vd = fgmean(f"score_{fam}_v_dec")
                vp = fgmean(f"score_{fam}_v_perp")
                if vd and b:
                    dd = np.array([r[f"score_{fam}_v_dec"] - r["score_baseline"]
                                   for r in recs if f"score_{fam}_v_dec" in r])
                    e = boot_mean(dd, rng)
                    cmd(f"{F}FreeGenVdec", _fmt(vd[0], 3))
                    cmd(f"{F}FreeGenVdecDelta", _fmt(e[0], 3, sign=True))
                    cmd(f"{F}FreeGenVdecDeltaCI", _ci(e[1], e[2]))
                if vp and b:
                    dp = np.array([r[f"score_{fam}_v_perp"] - r["score_baseline"]
                                   for r in recs if f"score_{fam}_v_perp" in r])
                    e = boot_mean(dp, rng)
                    cmd(f"{F}FreeGenVperp", _fmt(vp[0], 3))
                    cmd(f"{F}FreeGenVperpDelta", _fmt(e[0], 3, sign=True))
                    cmd(f"{F}FreeGenVperpDeltaCI", _ci(e[1], e[2]))
                    if vd:
                        rr = boot_ratio(dp, dd, rng)
                        cmd(f"{F}FreeGenRho", _fmt(rr[0], 2))
                        cmd(f"{F}FreeGenRhoCI", _ci(rr[1], rr[2], nd=2))

    (TAB / "numbers.tex").write_text("\n".join(L) + "\n")
    print(f"[stage4r] wrote {len(L)} macros to tables/numbers.tex")
    if probe_path.exists():
        make_generations(json.loads(probe_path.read_text()))
    make_examples()


def _tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    s = s.replace("\n", r" \textbackslash n ")
    # monospace text never hyphenates; give long alphanumeric runs break
    # points (alphanumeric-only so control sequences are never split)
    import re
    return re.sub(r"([A-Za-z0-9]{14})(?=[A-Za-z0-9])", r"\1\\allowbreak{}", s)


def make_generations(probe):
    """Appendix table of greedy generations, with a coherence badge per row so
    the reader can see at a glance which setting has collapsed."""
    pc = probe.get("conditions", {})
    naive_ppl = pc.get("naive/dec", {}).get("ppl_ratio")
    gdec_ppl = pc.get("gated/dec", {}).get("ppl_ratio")
    gmm_ppl = pc.get("gated/mm", {}).get("ppl_ratio")
    badge_ok = lambda ppl: f"\\okbadge{{coherent, PPL {ppl:.2f}$\\times$}}" if ppl else r"\okbadge{coherent}"
    badge_bad = lambda ppl: f"\\badbadge{{incoherent, PPL {ppl:.0f}$\\times$}}" if ppl else r"\badbadge{incoherent}"
    labels = [
        ("baseline", "baseline (no intervention)", r"\okbadge{coherent}", False),
        ("naive/dec", "CAA at the naive operating point "
         f"($\\ell{{=}}{pc.get('naive/dec', {}).get('layer', 10)}$, "
         f"$\\alpha{{=}}{pc.get('naive/dec', {}).get('alpha', 8):g}$)", badge_bad(naive_ppl), True),
        ("gated/dec", "CAA at the gated operating point", badge_ok(gdec_ppl), False),
        ("gated/mm", "mass-mean at the gated operating point", badge_ok(gmm_ppl), False),
    ]
    prompts = list(probe["generations"]["baseline"].keys())
    out = [r"\noindent\okbadge{coherent} = model stays fluent \quad "
           r"\badbadge{incoherent} = generation has collapsed (the benchmark still rewards it). \\[4pt]"]
    for p in prompts:
        out.append(f"\\paragraph{{Prompt.}} \\emph{{{_tex_escape(p)}}}")
        out.append("\\begin{description}[leftmargin=1.2em, itemsep=3pt]\\raggedright")
        for key, lab, badge, broken in labels:
            g = probe["generations"].get(key, {}).get(p, "")
            g = _tex_escape(g[:400]) + (r"\,\ldots" if len(g) > 400 else "")
            body = f"\\textcolor{{badRed}}{{{g}}}" if broken else g
            out.append(f"\\item[{lab}\\,{badge}:] {{\\small\\texttt{{{body}}}}}")
        out.append("\\end{description}")
    (TAB / "generations.tex").write_text("\n".join(out) + "\n")
    print(f"[stage4r] wrote tables/generations.tex ({len(prompts)} prompts)")


def make_examples():
    """Worked examples showing what the truthfulness effect is: per-question MC2
    moving from the false answer to the true one, and the projection preserving
    it. Reads artifacts/recal/examples.json + the per-question MC2 jsonl."""
    ep = RC / "examples.json"
    if not ep.exists():
        return
    ex = json.loads(ep.read_text())
    base = {r["idx"]: r["mc2"] for r in load_jsonl(RES / "baseline.jsonl")}
    vdec = {r["idx"]: r["mc2"] for r in load_jsonl(RES / "mm" / "v_dec.jsonl")}
    vperp = {r["idx"]: r["mc2"] for r in load_jsonl(RES / "mm" / "v_perp_al64.jsonl")}
    out = []
    for e in ex:
        i = e["idx"]
        if i not in base:
            continue
        out.append(f"\\paragraph{{Question.}} \\emph{{{_tex_escape_text(e['question'])}}}")
        out.append("\\begin{description}[leftmargin=1.4em, itemsep=3pt]")
        out.append(f"\\item[] \\trueans{{{_tex_escape_text(e['true'])}}}")
        out.append(f"\\item[] \\falseans{{{_tex_escape_text(e['false'])}}}")
        out.append(f"\\item[MC2 (probability mass on the true answer):] "
                   f"baseline ${base[i]:.2f}$ $\\;\\rightarrow\\;$ "
                   f"mass-mean $v$ ${vdec.get(i, float('nan')):.2f}$ $\\;\\rightarrow\\;$ "
                   f"$v^{{\\perp}}$ (vocabulary readout removed) ${vperp.get(i, float('nan')):.2f}$")
        out.append("\\end{description}")
    (TAB / "examples.tex").write_text("\n".join(out) + "\n")
    print(f"[stage4r] wrote tables/examples.tex ({len(ex)} examples)")


def _tex_escape_text(s: str) -> str:
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def make_caagrid_table(summary, cfg):
    """Table of CAA test-set Delta MC2 at every coherent grid setting."""
    import glob as _glob
    import re as _re
    base = load_jsonl(RES / "baseline.jsonl")
    files = sorted(_glob.glob(str(RES / "dec_grid" / "*.jsonl")))
    if not base or not files:
        return
    rng = np.random.default_rng(SEED)
    rows = [r"\begin{tabular}{llc}", r"\toprule",
            r"$\ell$ & $\alpha$ & test $\Delta$MC2 [95\% CI] \\", r"\midrule"]
    entries = []
    for f in files:
        m = _re.search(r"L(\d+)_a(\d+(?:\.\d+)?)", f)
        layer, alpha = int(m.group(1)), float(m.group(2))
        g = load_jsonl(Path(f))
        if not g:
            continue
        bb, gg = align(base, g, "mc2")
        e = boot_mean(gg - bb, rng)
        entries.append((layer, alpha, e, False))
    # The starred operating-point row duplicates the headline CAA delta; read it
    # from the single computed value in `summary` (seed 0) rather than re-bootstrapping,
    # and label it from config, so it is identical to numbers.tex / Table 1 / Table 7.
    op = load_jsonl(RES / "dec" / "v_dec.jsonl")
    if op and "dec" in summary["families"]:
        dvd = summary["families"]["dec"]["delta_vdec_mc2"]
        e = (dvd["point"], dvd["ci"][0], dvd["ci"][1])
        op_cfg = cfg["families"]["dec"]
        entries.append((op_cfg["layer"], op_cfg["alpha"], e, True))
    for layer, alpha, e, is_op in sorted(entries, key=lambda x: (x[0], x[1])):
        star = r"$^{\star}$" if is_op else ""
        rows.append(f"{layer} & {alpha:g}{star} & ${e[0]:+.3f}$ "
                    f"{{\\scriptsize $[{e[1]:+.3f}, {e[2]:+.3f}]$}} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "app_caagrid.tex").write_text("\n".join(rows) + "\n")
    print(f"[stage4r] wrote app_caagrid.tex ({len(entries)} coherent settings)")


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
    head = {
        "dec": (r"\emph{CAA} \quad ($\ell^{*}{=}\DecLayer$, $\alpha^{*}{=}\DecAlpha$, "
                r"PPL ratio \DecPplRatio)"),
        "mm": (r"\emph{mass-mean} \quad ($\ell^{*}{=}\MmLayer$, $\alpha^{*}{=}\MmAlpha$, "
               r"PPL ratio \MmPplRatio)"),
    }
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"condition & MC2 & $\Delta$MC2 & $\rho$ \\", r"\midrule"]
    for fam, f in summary["families"].items():
        lines += [rf"\multicolumn{{4}}{{l}}{{{head[fam]}}} \\", r"\midrule"]
        for cond in order:
            c = f["conditions"].get(cond)
            if not c:
                continue
            rho = (f"${c['rho']:.2f}$ \\,{{\\scriptsize $[{c['rho_ci'][0]:.2f},\\, {c['rho_ci'][1]:.2f}]$}}"
                   if "rho" in c else "---")
            lines.append(f"{lab[cond]} & ${c['mc2']:.3f}$ & ${c['delta_mc2']:+.3f}$ & {rho} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TAB / "results_matrix.tex").write_text("\n".join(lines) + "\n")


def make_appendix_tables(summary, certs, cfg, calib):
    """Generate the appendix data tables from artifacts (single source of truth)."""
    tok = json.loads((RC / "tokensets.json").read_text())

    # --- A. full calibration grid ---
    rows = [r"\begin{tabular}{llcccc}", r"\toprule",
            r"family & $\ell$ & $\alpha$ & PPL ratio & val.\ $\Delta$MC2 & coherent? \\",
            r"\midrule"]
    grid = sorted(calib["grid"], key=lambda r: (r["family"], r["layer"], r["alpha"]))
    last_fam = None
    for r in grid:
        fam = FAM_LABEL[r["family"]]
        famcell = fam if r["family"] != last_fam else ""
        if r["family"] != last_fam and last_fam is not None:
            rows.append(r"\midrule")
        last_fam = r["family"]
        star = r"$^{\star}$" if (r["layer"] == cfg["families"][r["family"]]["layer"]
                                 and r["alpha"] == cfg["families"][r["family"]]["alpha"]) else ""
        mark = "yes" if r["coherent"] else r"\textbf{no}"
        rows.append(f"{famcell} & {r['layer']} & {r['alpha']:g}{star} & "
                    f"{r['nll_ratio']:.3f} & ${r['val_mc2_delta']:+.3f}$ & {mark} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "app_calibration.tex").write_text("\n".join(rows) + "\n")

    # --- B. certificates ---
    rows = [r"\begin{tabular}{llcccc}", r"\toprule",
            r"family & set & $k$ & max$|Av|$ before & max$|Av^{\perp}|$ after & $\|v^{\perp}\|/\|v\|$ \\",
            r"\midrule"]
    setlab = {"al16": "aligned-16", "al64": "aligned-64", "al256": "aligned-256",
              "al1024": "aligned-1024", "cur": "curated", "stat": "statistical"}
    last_fam = None
    for key in sorted(certs.keys()):
        if "/" not in key:
            continue
        fam, s = key.split("/")
        if s.startswith("rand"):
            continue
        c = certs[key]
        famcell = FAM_LABEL.get(fam, fam) if fam != last_fam else ""
        if fam != last_fam and last_fam is not None:
            rows.append(r"\midrule")
        last_fam = fam
        rows.append(f"{famcell} & {setlab.get(s, s)} & {c['k']} & "
                    f"${c['max_direct_before']:.3f}$ & "
                    f"${_sci(c['max_direct_after'])}$ & ${c['norm_ratio']:.3f}$ \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "app_certificates.tex").write_text("\n".join(rows) + "\n")

    # --- C. curated + spillover lexicon ---
    def _ttoken(s):
        # T1/Times cannot render non-Latin glyphs; show a readable placeholder
        if all(ord(c) < 0x250 for c in s):
            return f"\\texttt{{{_tex_escape(s)}}}"
        kind = "non-Latin" if any(ord(c) > 0x2FF for c in s) else "diacritic"
        return f"\\texttt{{[{kind}]}}"

    def surfaces(d, n=None):
        items = [s.strip() for s in d.keys()]
        items = items[:n] if n else items
        return ", ".join(_ttoken(s) for s in items)
    lex = []
    lex.append(r"\textbf{Curated honest ($T^{+}$, " + str(len(tok["curated_plus"])) + r" surface forms):} ")
    lex.append(surfaces(tok["curated_plus"]) + r" \\[3pt]")
    lex.append(r"\textbf{Curated deceptive ($T^{-}$, " + str(len(tok["curated_minus"])) + r" forms):} ")
    lex.append(surfaces(tok["curated_minus"]) + r" \\[3pt]")
    lex.append(r"\textbf{Spillover honest (" + str(len(tok["spill_plus"])) + r" forms, never projected out):} ")
    lex.append(surfaces(tok["spill_plus"]) + r" \\[3pt]")
    lex.append(r"\textbf{Spillover deceptive (" + str(len(tok["spill_minus"])) + r" forms):} ")
    lex.append(surfaces(tok["spill_minus"]))
    (TAB / "app_lexicon.tex").write_text("\n".join(lex) + "\n")

    # --- C2. decoded data-driven token sets (statistical, aligned-64) ---
    def decoded_block(title, plus, minus, note=""):
        b = [rf"\textbf{{{title}}}{note} \\[2pt]"]
        b.append(r"\emph{honest side:} " +
                 ", ".join(_ttoken(t) for t in plus) + r" \\[2pt]")
        b.append(r"\emph{deceptive side:} " +
                 ", ".join(_ttoken(t) for t in minus))
        return b
    dec = []
    sp = RC / "statistical_decoded.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        dec += decoded_block(
            "Statistical set (top tokens by honest$-$deceptive system-prompt logit shift)",
            [x["tok"] for x in s["statistical_plus"]],
            [x["tok"] for x in s["statistical_minus"]],
            note=r"")
        dec.append(r"\\[6pt]")
    ap = RES / "aligned64_decoded.json"
    if ap.exists():
        a = json.loads(ap.read_text())
        for fam in ("dec", "mm"):
            if fam in a:
                dec += decoded_block(
                    f"Aligned-64 set, {FAM_LABEL[fam]} (the tokens the projection removes)",
                    [x["tok"] for x in a[fam]["plus"]],
                    [x["tok"] for x in a[fam]["minus"]])
                dec.append(r"\\[6pt]")
    if dec:
        (TAB / "app_decoded.tex").write_text("\n".join(dec) + "\n")

    # --- D. full per-condition table with MC1 and norm ratio ---
    order = ["v_dec", "v_perp_al16", "v_perp_al64", "v_perp_al256", "v_perp_al1024",
             "v_perp_cur", "v_perp_stat", "v_par_al64", "v_perp_al64_nm",
             "v_rand_s0", "v_rand_s1", "v_rand_s2"]
    base = load_jsonl(RES / "baseline.jsonl")
    rng = np.random.default_rng(SEED)
    rows = [r"\begin{tabular}{llccc}", r"\toprule",
            r"family & condition & $\Delta$MC2 [95\% CI] & $\Delta$MC1 [95\% CI] & $\rho$ (MC1) \\",
            r"\midrule"]
    for fam in summary["families"]:
        fdir = RES / fam
        b2, vd2 = align(base, load_jsonl(fdir / "v_dec.jsonl"), "mc2")
        b1, vd1 = align(base, load_jsonl(fdir / "v_dec.jsonl"), "mc1")
        d_dec1 = vd1 - b1
        # MC1 denominator gate: bootstrap once (both CI bounds from the same draw) and reuse
        dec1_gate = boot_mean(d_dec1, rng)
        dec1_significant = dec1_gate[1] > 0 or dec1_gate[2] < 0
        fdelta = summary["families"][fam]
        rows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{FAM_LABEL[fam]}}}}} \\")
        rows.append(r"\midrule")
        for cond in order:
            cr = load_jsonl(fdir / f"{cond}.jsonl")
            if not cr:
                continue
            _, c2 = align(base, cr, "mc2")
            _, c1 = align(base, cr, "mc1")
            if cond == "v_dec":
                # headline row: reuse the single computed value (SSOT) so it is identical
                # to numbers.tex, Table 1, and the CAA-grid starred row
                e2 = (fdelta["delta_vdec_mc2"]["point"], *fdelta["delta_vdec_mc2"]["ci"])
                e1 = (fdelta["delta_vdec_mc1"]["point"], *fdelta["delta_vdec_mc1"]["ci"])
            else:
                e2 = boot_mean(c2 - b2, rng)
                e1 = boot_mean(c1 - b1, rng)
            rho1 = "---"
            if cond != "v_dec" and dec1_significant:
                rr = boot_ratio(c1 - b1, d_dec1, rng)
                rho1 = f"${rr[0]:.2f}$"
            lab = {"v_dec": r"$v$", "v_perp_al16": r"$v^{\perp}$ al-16",
                   "v_perp_al64": r"$v^{\perp}$ al-64", "v_perp_al256": r"$v^{\perp}$ al-256",
                   "v_perp_al1024": r"$v^{\perp}$ al-1024", "v_perp_cur": r"$v^{\perp}$ cur",
                   "v_perp_stat": r"$v^{\perp}$ stat", "v_par_al64": r"$v^{\parallel}$ al-64",
                   "v_perp_al64_nm": r"$v^{\perp}$ al-64 nm", "v_rand_s0": "rand s0",
                   "v_rand_s1": "rand s1", "v_rand_s2": "rand s2"}[cond]
            rows.append(f" & {lab} & ${e2[0]:+.3f}$ {{\\scriptsize $[{e2[1]:+.3f}, {e2[2]:+.3f}]$}} "
                        f"& ${e1[0]:+.3f}$ {{\\scriptsize $[{e1[1]:+.3f}, {e1[2]:+.3f}]$}} & {rho1} \\\\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    (TAB / "app_full_matrix.tex").write_text("\n".join(rows) + "\n")
    print("[stage4r] wrote 4 appendix data tables")


def _sci(x):
    if x <= 0:
        return "0"
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"{m:.1f}\\times 10^{{{e}}}"


if __name__ == "__main__":
    main()
