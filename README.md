# liar-liar

> **Liar, Liar: Beyond Vocabulary Suppression.** A causal test of whether honesty steering manipulates an upstream representation or merely tilts the readout against deception-coded tokens, via token-conditional unembedding orthogonalization.
>
> [Read the paper (live PDF)](https://aryan-cs.github.io/liar-liar/paper.pdf) · [Read the proof (PDF)](docs/proof.pdf) · [License: CC BY-NC-ND 4.0](LICENSE) · [Source on GitHub](https://github.com/aryan-cs/liar-liar)

This repository hosts the paper, the formal apparatus, the experimental pipeline, and the results for a project on whether representation-engineering steering vectors for honesty actually manipulate an upstream concept or merely tilt the readout against a small lexicon of behavior-coded tokens. The paper is `docs/paper.pdf`; the mathematical machinery is in `docs/proof.tex` (compiled to `docs/proof.pdf`); the original experimental program is preserved in `PLAN.md`.

**The experiments are complete, and the answer is a double dissociation.** On Llama-3-8B-Instruct: the popular contrastive (CAA) honesty vector shifts honesty-coded *words* by over a logit without improving truthfulness at any coherent steering strength, and its apparent benchmark gains arise only after generation has collapsed (held-out perplexity 69x baseline) — a regime where the naive analysis also fabricates a confident "deep effect" verdict. The mass-mean vector improves truthfulness for real (test ΔMC2 +0.073, 95% CI [+0.038, +0.108], at perplexity ratio 1.15), and that gain survives certified excision of direct readout onto the 64 tokens it moves most: ρ = 0.95 [0.79, 1.13]. The aligned-64 loss is indistinguishable from random-subspace controls in-domain, the result is stable from 16 to 1024 excised directions, and the projected gain remains intact under paraphrase (ρ_OOD = 0.97). Where honesty steering works it is not vocabulary suppression, and where it is vocabulary-level it does not work.

---

## In simple terms

A class of safety techniques nudges language models toward more honest answers by adding a vector to the model's internal state mid-computation. The reported benchmark gains do not distinguish two mechanisms. The vector may be moving an upstream representation of honesty that the rest of the model reads and acts on, or it may be making words like *lie* and *deceive* less likely at the output. Both produce the same headline scores.

This project constructs the test that separates the two. We project the vector orthogonal to the readout direction for a chosen set of honesty-coded words, so its direct effect on those words is zero, then measure how much honest behavior survives. Whatever survives came from somewhere other than vocabulary suppression.

The proof develops the construction; the paper reports the experiments. The result: for the steering construction that genuinely improves truthfulness, essentially *all* of the effect survives — honesty steering, where it works, is not a vocabulary trick. And the construction most widely used in practice turns out not to improve truthfulness at all once you require the model to remain coherent: it changes the words, not the truthfulness.

---

## What is the question, in one paragraph?

Representation-engineering (RepE) interventions add a contrastively constructed vector to a transformer's residual stream at a middle layer; the reported effect is a reduction in deceptive, sycophantic, or refusal-violating behavior. The published numbers do not separate two distinct mechanisms. Under the *shallow* account, the vector tilts the final logit head against tokens such as *lie*, *trick*, *false*, *deceive*, and the intervention works because deception-coded words become improbable, with no semantic concept involved. Under the *deep* account, the vector moves an upstream representation that downstream attention and feed-forward layers consume, producing behavior that survives vocabulary substitution, paraphrase, and translation. The two accounts predict identical TruthfulQA scores and divergent out-of-distribution generalization. The field has implicitly assumed the second, and the standard experimental setup does not adjudicate.

---

## What this repository contributes

We separate the two mechanisms by construction. For a chosen deception-coded token set $T$, take the orthogonal complement of the rows of the effective unembedding matrix indexed by $T$ and project the steering vector $v_{\text{dec}}$ onto it to obtain $v^{\perp}$. The projected vector has zero direct logit contribution at every token in $T$. Injecting $v^{\perp}$ at the original intervention layer, any change in behavior cannot run through direct readout at $T$ and must propagate through downstream attention and feed-forward layers. The ratio of $v^{\perp}$'s behavioral effect to $v_{\text{dec}}$'s is the depth statistic.

The proof at [`docs/proof.pdf`](docs/proof.pdf) develops:

1. Why the naive global formulation is impossible. When the vocabulary exceeds the residual dimension, the unembedding matrix has trivial kernel and no nonzero vector is orthogonal to every unembedding row. The construction must be token-conditional.
2. The RMSNorm-corrected effective unembedding $\widetilde{W}_U^\star$, the actual object the post-norm readout maps from. Prior work projects against raw $W_U$; this is subtly wrong.
3. The minimum-norm characterization of the projection. The construction is the unique closest perturbation of $v_{\text{dec}}$ that produces zero direct effect on $T$, in the style of LEACE adapted to the token-conditional setting.
4. The direct-versus-indirect path decomposition that makes the test statistic meaningful.
5. The positioning against the closest precedents: LEACE, the Arditi refusal-direction orthogonalization, the Park-Choe-Veitch causal-inner-product duality, the Venkatesh-Kurapath non-identifiability result, and the Nadaf function-vector decoding gap.

The companion [`PLAN.md`](PLAN.md) specifies the experimental program: which checkpoints, which steering constructions, which token sets, which benchmarks, which OOD probes, and what each empirical outcome would mean.

---

## On the novelty gap

The closest prior work is:

- **LEACE** (Belrose et al., NeurIPS 2023). Minimum-norm projection that erases linear concept information from a representation. Same projection machinery, different subspace target.
- **Arditi et al.** (NeurIPS 2024). Project a refusal direction out of every matrix that writes to the residual stream. Same orthogonalization idiom, dual subspace.
- **Venkatesh and Kurapath** (arXiv:2602.06801, Feb 2026). Steering vectors are non-identifiable: orthogonal perturbations within the activation-to-logit Jacobian null space leave behavior unchanged. Closest theoretical precedent.
- **Nadaf** (arXiv:2604.02608, April 2026). Function vectors steer model behavior in cases where the logit lens cannot decode the steered output, demonstrating the off-readout channel exists for the function-vector setting.
- **hughvd's unembedding-steering-benchmark** (GitHub, 2024). Implements the unembedding-orthogonal steering construction on Gemma-2-9B with sentiment as the worked example.

The contribution is the application of this projection to honesty steering with machine-precision zero-direct-effect certificates, the RMSNorm correction, the coherence-gated operating-point protocol and random-subspace control that make the statistic interpretable, and the head-to-head two-family decomposition summarized via $\rho$, $\sigma_T$, and $\rho_{\mathrm{OOD}}$.

---

## Construction at a glance

```mermaid
flowchart LR
    V["Steering vector<br/><sub>v_dec</sub>"]
    M["Effective unembedding<br/><sub>W_U · J_norm</sub>"]
    T["Token-coded set T<br/><sub>honest ∪ deceptive</sub>"]
    S["Row subspace S_T<br/><sub>span of selected rows</sub>"]
    P["Projector P_T_perp<br/><sub>I - A^+ A</sub>"]
    Vp["Projected vector<br/><sub>v_perp</sub>"]
    R["Behavioral effect<br/><sub>Δ(v_perp)</sub>"]
    Rho["Depth statistic<br/><sub>ρ = Δ(v_perp) / Δ(v_dec)</sub>"]

    M --> S
    T --> S
    S --> P
    P --> Vp
    V --> Vp
    Vp -- "inject at layer ℓ*" --> R
    R --> Rho
```

If $v_{\text{dec}}$ acts primarily through direct logit attribution at $T$, the projected vector $v^{\perp}$ has near-zero behavioral effect and $\rho \approx 0$. If $v_{\text{dec}}$ acts primarily through indirect propagation, $v^{\perp}$ preserves the effect and $\rho \approx 1$.

**Measured outcome** (Llama-3-8B-Instruct, 497 held-out TruthfulQA questions): for the mass-mean vector, $\rho = 0.95$ at aligned-64 ($\sigma_T = 0.08$ across token-set constructions, $\rho_{\mathrm{OOD}} = 0.97$ under paraphrase), with random-subspace controls at the same level — the effect is deep, and the readout subspace is not even special. For the CAA vector the depth question is moot: its denominator is statistically zero at every coherent operating point. The logit lens shows downstream layers re-synthesizing the suppressed $T$-readout within two blocks of the injection site.

---

## Repository layout

```
liar-liar/
├── README.md                  ← you are here
├── PLAN.md                    ← original experimental program (the executed subset is in the paper)
├── docs/                      ← served by GitHub Pages (aryan-cs.github.io/liar-liar/)
│   ├── main.tex               ← the paper source
│   ├── paper.pdf              ← compiled paper (→ /paper.pdf on Pages)
│   ├── sections/              ← one .tex per section
│   ├── tables/                ← auto-generated table bodies + number macros
│   ├── proof.tex              ← formal apparatus (LaTeX source)
│   └── proof.pdf              ← compiled proof
├── liar/                      ← Python package
│   ├── unembedding.py         ← effective unembedding, RMSNorm Jacobian, P_T construction
│   ├── steering.py            ← CAA and mass-mean vectors
│   ├── tokenset.py            ← curated, statistical, aligned-k T constructions
│   ├── eval.py                ← TruthfulQA MC scoring, held-out NLL, eta capture
│   ├── lens.py                ← layer-wise T-readout trajectories
│   └── model.py               ← loading, residual capture, steering hooks
├── scripts/                   ← staged pipeline (GPU stages + Mac-side analysis)
│   ├── stage_recal.py         ← coherence-gated calibration + projection variants
│   ├── stage2_recal.py        ← headline condition matrix
│   ├── stage3_recal.py        ← paraphrase OOD + lens
│   ├── stage4_recal.py        ← analysis, figures, tables, paper number macros
│   └── probe_faithfulness.py  ← broken-instrument evidence (norms, PPL, generations)
├── supplement/                ← machine-readable prompt/response records and schema
├── figures/                   ← generated figures
└── results/                   ← per-question results and summary_recal.json
```

**Provenance note.** The naive-operating-point evidence in Section 5.1 of the paper comes from the superseded first pipeline (`stage1_vectors.py` through `stage4_analysis.py`, driven by `run_all.sh`). Those scripts are retained as the provenance of that run: its per-question results are committed under `results/stage2_old/`, and its config and vectors (`artifacts/stage1/`) are fetched by `scripts/fetch_results.sh` so the faithfulness probe can rerun the naive operating point. The live pipeline is `run_recal.sh` (GPU side) plus `analyze.sh` (analysis side); `stage4_recal.py` regenerates every number in the paper from the artifacts.

---

## How to read the documents

1. **[README.md](README.md)** *(this file)*. Orientation.
2. **[docs/paper.pdf](docs/paper.pdf)**. The paper: the broken-instrument demonstration, the two-family decomposition, the depth verdict, and the lens evidence, with all numbers generated from the artifacts in this repository.
3. **[docs/proof.pdf](docs/proof.pdf)**. The formal apparatus: the impossibility of the global formulation, the token-conditional construction, the RMSNorm correction, the rank-one variant, the direct-versus-indirect path decomposition, the depth statistic, the minimum-norm characterization, the prior-work positioning, and the limitations.
4. **[supplement/README.md](supplement/README.md)**. The JSONL prompt/response records, schemas, deterministic example-selection rule, and provenance limitations.
5. **[PLAN.md](PLAN.md)**. The original experimental program, preserved as designed; the executed subset and the deviations from it are documented in the paper's setup section.

The load-bearing sections of the proof are **§4** (Token-Conditional Orthogonalization), which defines the construction, and **§6** (Direct-Versus-Indirect Path Decomposition), which justifies the depth statistic. §3 shows why the construction must be token-conditional; §9 positions the work against the closest prior projections.

---

## Building the proof PDF

The proof is standard LaTeX and compiles cleanly with [Tectonic](https://tectonic-typesetting.github.io/), which downloads required packages on first use.

```bash
# install once
brew install tectonic           # macOS
# or follow instructions for your platform

# compile
cd docs
tectonic proof.tex
```

This produces `docs/proof.pdf`. The pre-compiled PDF is committed so casual readers do not need a LaTeX toolchain.

A traditional `pdflatex` or `latexmk` toolchain works equivalently:

```bash
cd docs && latexmk -pdf proof.tex
```

---

## Status

| Milestone | State |
|-----------|-------|
| Formal apparatus written | done |
| Token-conditional construction proved well-defined and minimum-norm | done |
| RMSNorm correction worked out | done |
| Prior-work comparison written and citations verified | done |
| Reference implementation of $P_T^\perp$ and both steering constructions | done |
| Coherence-gated calibration and the broken-instrument demonstration | done |
| Full condition matrix (both families, all projections and controls) on Llama-3-8B-Instruct | done |
| Paraphrase OOD block and logit-lens re-synthesis trajectories | done |
| Zero-direct-effect certificates (max residual direct effect $7 \times 10^{-15}$ logits) | done |
| Paper written with every number generated from artifacts | done |
| Replication at other scales and model families | future work |
| Free-form (non-teacher-forced) honesty evaluation under steering | future work |

---

## A note on framing

The construction is operational. $\rho$ measures the proportion of a steering vector's behavioral effect that survives token-conditional readout suppression: a claim about the geometry of the residual stream. Disagreement should target the formal commitments (Theorems 6.1 and 8.1 in `docs/proof.pdf`), the experimental design (Section 4 of the paper), or the artifacts themselves — every number in the paper is generated from the per-question results in `results/` by `scripts/stage4_recal.py`, and the zero-direct-effect certificates ship with the vectors.

---

## Citation

The paper is at [docs/paper.pdf](docs/paper.pdf). Until a preprint number exists, please cite the repository.

```
@misc{gupta2026liarliar,
  title  = {Liar, Liar: Beyond Vocabulary Suppression},
  author = {Aryan Gupta},
  email  = {aryan.cs.app@gmail.com},
  year   = {2026},
  note   = {\url{https://github.com/aryan-cs/liar-liar}}
}
```

---

## License

The writeup, formal proof, experimental plan, and all documents in this repository are licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/). You may read and share with attribution; commercial use, derivative works, translations, condensations, and inclusion in training data require explicit prior written permission from the author. See [LICENSE](LICENSE) for the binding terms.

When experimental code is released, it will carry a separate software license in its own directory; the documents in this repository remain under CC BY-NC-ND 4.0.

For permission requests outside the terms of the license, contact `aryan.cs.app@gmail.com`.
