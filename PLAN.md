# Liar, Liar: A Causal Test of Deep Versus Shallow Deception in Language Models

**Research plan.** Token-conditional unembedding orthogonalization as a quantitative decomposition of representation-engineering steering vectors into a direct-readout component and a downstream-propagation component, applied to deception.

The formal apparatus is in `docs/proof.tex` (compiled to `docs/proof.pdf`). This file is the experimental program: what we run, on which models, against which benchmarks, with which baselines, and what counts as a result.

The earlier research-plan file in this workspace (`docs/PLAN_steering_rebels_legacy.md`) is for a separate project on training a model to resist write-side steering; it is preserved for cross-reference but is not the subject of this plan.

---

## 1. The question

When a representation-engineering (RepE) intervention reduces deceptive behavior on a benchmark, two distinct mechanisms can produce the measured effect. Under the *shallow* mechanism, the steering vector tilts the final logit head against a small lexicon of deception-coded tokens; the benchmark score improves because tokens such as *lie*, *trick*, *false*, *deceive* are suppressed and tokens such as *honest*, *true* are promoted, with no semantic concept involved. Under the *deep* mechanism, the steering vector moves an upstream representation that downstream attention and feed-forward layers consume, producing behavior that is stable across vocabulary choices, languages, and registers.

The two mechanisms predict identical TruthfulQA scores and differ under prompt paraphrase, translation, vocabulary substitution, and interpretability probes. The standard experimental setup does not adjudicate.

We isolate the two contributions by construction. Given a steering vector `v_dec`, project it onto the orthogonal complement of the span of the unembedding rows for a chosen deception-coded token set $T$. The projected vector $v^\perp$ has, by construction, zero direct logit contribution at every token in $T$. Whatever behavioral effect $v^\perp$ produces must propagate through downstream attention and feed-forward layers. The ratio of $v^\perp$'s behavioral effect to $v_{\text{dec}}$'s behavioral effect is a depth-of-representation statistic.

The proof in `docs/proof.tex` works out the linear algebra, including the RMSNorm correction that makes the projection target the *effective* unembedding rather than the raw $W_U$. The reason the global formulation does not work, and why the token-conditional version is the only well-posed one, is the rank counting in Proposition 3.1: for every modern model the kernel of $W_U$ is trivial because $V > d$.

---

## 2. Position relative to prior work

### 2.1 Direct precedents

| Work | Subspace projected out | What they ask |
|---|---|---|
| LEACE (Belrose et al., NeurIPS 2023, arXiv:2306.03819) | Whitened cross-covariance of label on representation | Is linear label-decoding impossible? |
| Arditi et al. (NeurIPS 2024, arXiv:2406.11717) | A single refusal direction, applied to every matrix that writes to the residual stream | Does refusal disappear when the direction is unwritable? |
| Park, Choe, Veitch (ICML 2024, arXiv:2311.03658) | (Theoretical) | Are concept-direction and steering-direction dual? |
| Venkatesh and Kurapath (arXiv:2602.06801, Feb 2026) | Jacobian-null perturbations to the steering vector | Are steering vectors identifiable? |
| Nadaf (arXiv:2604.02608, April 2026) | (Measurement) | Do function vectors steer beyond what the logit lens can decode? |
| hughvd unembedding-steering-benchmark (GitHub, 2024) | Token unembedding rows on Gemma-2-9b, sentiment as worked example | Method development |
| **This work** | Effective unembedding rows on a deception-coded token set $T$, with RMSNorm correction | Honesty/deception case on modern deception benchmarks, depth statistic |

Three of these overlaps are load-bearing for the novelty positioning:

- **Venkatesh and Kurapath (2026)** prove steering vectors are non-identifiable: orthogonal perturbations within the Jacobian null space leave behavior unchanged. Their null space is $\ker(W_U J_{\mathcal{N}} M^{(\ell^\star \to L)})$; ours is the readout-restricted version $\ker(W_U J_{\mathcal{N}})[T,:]$. We cite them as the closest theoretical precedent.
- **Nadaf (2026)** demonstrates the off-readout steering channel for function vectors. We ask whether the same channel mediates honesty/deception steering, on the deception-specific benchmarks introduced in 2025, with a quantitative measurement.
- **hughvd's repository** implements unembedding-orthogonal steering on Gemma-2-9b for sentiment. We extend the construction to honesty/deception steering on Llama-2, Llama-3, Mistral, Qwen, and Gemma, with a token-set design, the RMSNorm correction, and evaluation on benchmarks released after the repository was last updated.

### 2.2 Adjacent work we will not duplicate

- **CCS (Burns et al., 2022) and SAPLMA (Azaria and Mitchell, 2023)** train probes for honesty. We use the steering vector, not a learned probe, as the intervention.
- **Tan et al. (NeurIPS 2024)** show steering vectors generalize poorly out of distribution. Our work explains a possible mechanism for that finding by quantifying the readout-aligned fraction.
- **The Marks and Tegmark (2023) mass-mean direction** is the readout-side construction we use to instantiate the rank-one variant of our projection.

### 2.3 The novelty gap

Each component has precedent: the projection machinery (LEACE, Arditi, hughvd), the off-readout steering channel (Nadaf), and the non-identifiability of steering vectors (Venkatesh and Kurapath). The contribution is the application to deception on the modern deception benchmarks. Five components:

1. The depth statistic $\rho(v_{\text{dec}}, T)$ and its per-prompt variant $\rho_\eta$, defined as the ratio of orthogonalized to original behavioral effect, reported alongside the cross-set stability $\sigma_T$.
2. The RMSNorm-corrected effective unembedding $\widetilde{W}_U^\star$ as the projection target. Prior work projects against raw $W_U$ or against learned probe directions and ignores the post-norm correction.
3. Application to honesty and deception steering, evaluated on MASK (Ren et al., 2025), Liars' Bench (Kretschmar et al., 2025), and DeceptionBench (Jiang et al., 2025), with TruthfulQA as the calibration baseline.
4. Quantitative comparison of $\rho$ across the LAT, CAA, ITI, and Marks-Tegmark mass-mean constructions on identical models, ranking them by readout-aligned fraction.
5. An OOD prediction: under the deep alternative, $\rho$ should be stable under prompt paraphrase and translation; under the shallow null it should collapse on either.

---

## 3. Formal hypothesis

We restate the schema from the proof in operational form.

**Shallow null ($H_0$).** $\rho(v_{\text{dec}}, T)$ is consistent with zero across choices of $T$, intervention magnitudes, model checkpoints, and benchmarks.

**Deep alternative ($H_1$).** $\rho(v_{\text{dec}}, T)$ is bounded away from zero, with $\sigma_T$ small relative to $\rho$.

**Intermediate ($H_{0.5}$).** $\rho \in (0, 1)$, with empirical questions about which factors $\rho$ depends on. This is the most likely outcome, and the empirical findings under this regime constitute the main result.

The plan is structured to give an informative answer under any of these regimes.

---

## 4. Experimental design

### 4.1 Models

Open-weight, instruction-tuned, with publicly inspectable architecture.

| Family | Checkpoints | $V$ | $d$ | Normalization | $L$ |
|---|---|---|---|---|---|
| Llama-2 | 7B-chat, 13B-chat | 32,000 | 4096 / 5120 | RMSNorm | 32 / 40 |
| Llama-3 | 8B-Instruct | 128,256 | 4096 | RMSNorm | 32 |
| Mistral | 7B-Instruct-v0.3 | 32,768 | 4096 | RMSNorm | 32 |
| Qwen2.5 | 7B-Instruct, 14B-Instruct | 152,064 | 3584 / 5120 | RMSNorm | 28 / 48 |
| Gemma-2 | 9B-it | 256,128 | 3584 | RMSNorm | 42 |

This set spans the relevant axes of variation: vocabulary size (the magnitude of any shallow effect scales with $|T|/V$), residual width (which controls the dimension of the projected subspace), and post-training regime (Llama-3 and Llama-2 differ substantially in honesty-related RLHF).

GPT-2 family models are excluded from the headline because they use full LayerNorm and require the centering-null treatment from §11.3 of the proof; a single GPT-2-XL run is included as a LayerNorm sanity check. Closed-weight frontier models are not run: the intervention requires residual stream access.

### 4.2 Steering vector constructions

We compare four canonical constructions of $v_{\text{dec}}$, all at a single intervention layer $\ell^\star$ in the middle third of the network (the standard regime for RepE):

1. **Mean-difference (CAA, Rimsky et al. 2024).** Honest-prompted versus deceptive-prompted continuations, mean residual difference at layer $\ell^\star$.
2. **LAT principal component (Zou et al. 2023).** Honest/deceptive contrast pairs, take the first principal component of the difference matrix at layer $\ell^\star$.
3. **ITI per-head truthful direction (Li et al. 2023).** Train per-head linear probes on a TruthfulQA-derived contrast set, take the mass-mean direction in the top-$K$ heads by probe accuracy. Collapse to a single residual-stream direction by averaging across selected heads for comparability with the other constructions.
4. **Marks-Tegmark mass-mean (2023).** Difference of activation means on a true/false statement dataset, evaluated at $\ell^\star$.

All four are computed on the same training set per model. The training set uses the contrast pairs released with RepE and CAA, with deduplication and a held-out validation split.

### 4.3 Token-coded set construction

Three constructions of $T$, evaluated independently:

1. **Curated.** A 64-token list (32 honest, 32 deceptive) compiled from the Park et al. (2024) survey lexicon and morphological variants. Released with the project.
2. **Statistical.** Top-32 honest and top-32 deceptive tokens by mean log-probability shift between honest and deceptive contexts on a 1k-prompt held-out corpus.
3. **Probe-derived.** Top-32 honest and top-32 deceptive tokens by the coefficient magnitudes of an $\ell_1$-regularized logistic regression trained on the readout-projected representations.

For each construction, we evaluate the projection at the raw vocabulary, BPE-level, and morphology-merged levels (with $T$ extended to include all tokens belonging to the same root form). This gives nine $T$ choices per model, and the cross-set stability $\sigma_T$ summarizes how much the depth statistic depends on this design choice.

### 4.4 Effective unembedding

For each model and each readout point $z^\star$ (the unperturbed final residual at the last token position on a calibration prompt), compute
$\widetilde{W}_U^\star = W_U \cdot J_{\mathcal{N}_\gamma}(z^\star) \in \mathbb{R}^{V \times d}$
exactly as specified in the proof. Average the Jacobian over a 256-prompt calibration set to get a stable approximation. The projection target is $\widetilde{W}_U^\star[T,:]$.

For the rank-one logit-difference variant, use the mean of effective unembedding rows over $T^+$ minus the mean over $T^-$ as the direction $\hat{d}_{HD}$.

### 4.5 Intervention layer

For each model, $\ell^\star$ is selected to maximize $|\Delta(v_{\text{dec}})|$ on a held-out validation split of TruthfulQA. This matches standard RepE practice. We report the sensitivity of $\rho$ to $\ell^\star$ within $\pm 2$ layers.

### 4.6 Evaluation suite

| Benchmark | What it measures | Why it is in the suite |
|---|---|---|
| TruthfulQA (Lin et al. 2022) | Imitative falsehoods | Calibration baseline. RepE numbers were originally reported here. |
| MASK (Ren et al. 2025) | Whether models say what they believe under pressure | Disentangles knowing from saying. The cleanest honesty test. |
| Liars' Bench (Kretschmar et al. 2025) | Cross-lie-type generalization of detectors and interventions | Tests whether the intervention generalizes across lie genres. |
| DeceptionBench (Jiang et al. 2025) | Context-conditional deception across scenarios | Tests intent representation, not surface tokens. |
| Sandbagging eval (van der Weij et al. 2024, WMDP-style) | Strategic capability hiding | Tests upstream representation directly: there is no surface vocabulary to suppress. |
| Sleeper-agent defection probe (MacDiarmid et al. 2024) | Trigger-conditional deception | Adversarial: the intervention is meant to suppress the defection behavior, not vocabulary correlated with it. |

We report $\Delta(v_{\text{dec}})$, $\Delta(v^\perp)$, $\rho$, $\rho_\eta$, and $\sigma_T$ on each benchmark, each model, and each vector construction.

### 4.7 OOD probes

A separate experimental block tests whether $\rho$ tracks generalization.

1. **Paraphrase.** Run the benchmarks with prompts paraphrased through three back-translation pivots (German, Japanese, Arabic) and an LLM paraphraser.
2. **Translation.** Translate the prompts into Spanish, French, German, and Mandarin and rerun. The deception-coded vocabulary in $T$ is anchored to English; translation should disproportionately damage shallow steering.
3. **Vocabulary substitution.** A pair of prompt sets where we swap deception-coded English words for synonyms outside $T$ (such as *fabrication* in place of *lie*). Under the shallow account this should leak steering effect; under the deep account it should not.

### 4.8 Baselines and ablations

1. **Random projection.** Replace $P_T^\perp$ with the orthogonal projector onto a random $(d - k)$-dimensional subspace. The random-projection $\rho$ should differ statistically from the $T$-projection $\rho$ if and only if the unembedding subspace is special.
2. **Single-token projection.** $T = \{\text{one token}\}$. The $\rho$ as a function of $|T|$ tells us how concentrated the readout mass is.
3. **Magnitude rescaling.** Run $\alpha v_{\text{dec}}$ for $\alpha \in [0.5, 2]$. Linear behavior is the small-perturbation check.
4. **No-projection control.** Run $v_{\text{dec}}$ with the rank-one $\hat{d}_{HD}$ direction *amplified* rather than removed. Under the deep account this should amplify behavior; under the shallow account it should saturate.

---

## 5. Analysis plan

### 5.1 Headline analysis

For each model $m$, vector construction $c$, token-set construction $\tau$, and benchmark $b$, compute $\rho_{mctb}$ and aggregate.

**Primary result.** Boxplot of $\rho$ across $(c, \tau, b)$ per model, with paired tests against $\rho = 0$ (shallow) and $\rho = 1$ (deep). Report median, IQR, and the fraction of cells with $\rho$ greater than $0.5$.

**Stability.** Report $\sigma_T$ per $(m, c, b)$ and order models by stability.

**Cross-vector comparison.** Per model, compare $\rho$ across $c \in \{\text{CAA, LAT, ITI, mass-mean}\}$. A construction with systematically lower $\rho$ is more readout-aligned and therefore shallower by our operational definition.

### 5.2 OOD analysis

Compute $\rho_{\text{OOD}}$ on paraphrase, translation, and vocabulary-substitution settings and report the ratio $\rho_{\text{OOD}} / \rho_{\text{in-dist}}$ for each model. Under the deep alternative this ratio should be near $1$; under the shallow null it should approach $0$ on the translation and vocabulary-substitution settings.

### 5.3 Mechanism analysis

For the largest $\rho$ outliers (where the deep account is strongly supported) we will run:

1. **Path patching** (Wang et al., 2022 IOI methodology) to identify which downstream attention heads and MLP blocks consume the orthogonalized contribution.
2. **SAE feature attribution** (Gemma Scope and Llama Scope) on the indirect-path contribution, identifying which sparse features fire differently under $v^\perp$ versus baseline.
3. **Tuned-lens visualization** of the residual stream at each layer downstream of $\ell^\star$ to track when the orthogonalized contribution becomes readout-visible.

These do not have to converge for the headline result to land. They are how we interpret $\rho$ if it is large and stable.

---

## 6. Success criteria

Three tiers, ordered by ambition.

1. *Method.* The construction, the depth statistic, baselines, and headline numbers on the eight checkpoints. The contribution is a tool other groups can apply.
2. *Empirical finding.* A consistent value of $\rho$ across models and benchmarks, whether small, large, or intermediate. Each outcome adjudicates the deep-versus-shallow question for honesty steering.
3. *Stronger result.* A relationship between $\rho$ and an external property such as model size, post-training regime, or OOD generalization.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| $\rho$ is so dependent on $T$ that $\sigma_T$ swamps the signal | Pre-register the three $T$ constructions, report $\sigma_T$ prominently, treat large $\sigma_T$ as the finding |
| RMSNorm linearization fails for the steering magnitudes in use | Report the residual between predicted and measured direct contribution as a sanity check, restrict main analysis to magnitudes where this is small |
| The intervention layer $\ell^\star$ choice drives $\rho$ | Report sensitivity within $\pm 2$ layers; if the result is layer-fragile, that is itself interesting and we report it |
| Benchmark contamination | Use Liars' Bench as the headline (released Nov 2025, post-cutoff for most checkpoints) and report TruthfulQA only as calibration |
| The hughvd repository or Venkatesh-Kurapath paper turns out to scoop a specific finding | Position our contribution as the deception-specific instantiation; reframe rather than abandon |

---

## 8. Repository layout

```
liar-liar/
├── README.md                  five-minute orientation
├── PLAN.md                    this file
├── docs/
│   ├── proof.tex              formal apparatus
│   └── proof.pdf              compiled
├── liar/                      Python package (planned)
│   ├── unembedding/           W_U row extraction, RMSNorm Jacobian, P_T construction
│   ├── steering/              CAA, LAT, ITI, mass-mean implementations
│   ├── tokenset/              curated, statistical, probe-derived T constructions
│   ├── eval/                  MASK, Liars' Bench, DeceptionBench, TruthfulQA harnesses
│   └── ood/                   paraphrase, translation, vocab-substitution probes
├── experiments/               per-model run scripts and configs
├── results/                   per-run JSON and per-model summary parquets
└── tests/
```

---

## 9. Reading order

For a reviewer or collaborator reading cold:

1. This file, §1–3, for the question and its position in the literature.
2. `docs/proof.pdf` §3–5, for the mathematical apparatus: the impossibility, the conditional construction, the rank-one variant.
3. This file, §4–5, for the experimental program.
4. `docs/proof.pdf` §6–7, for the direct/indirect decomposition and the test statistic.
5. `docs/proof.pdf` §9, for the prior-work positioning.
