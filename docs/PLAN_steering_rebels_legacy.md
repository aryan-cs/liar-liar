# Steering Rebels: Research Plan

A conference-track project on whether a model can be trained to maintain a hidden deceptive policy under **active write-side** representation engineering interventions, distinct from prior work on evading **passive read-side** monitors.

**Author.** Aryan Gupta (`aryan.cs.app@gmail.com`).

---

## 1. One-paragraph summary

Activation-level safety interventions, including Inference-Time Intervention (ITI), Contrastive Activation Addition (CAA), and Representation Engineering (RepE), are routinely treated as invisible to the model and faithful in their effect. Recent work has dismantled the first half of that claim: models can be fine-tuned to detect a steering vector with 95.5% accuracy on held-out concepts ([Fonseca Rivera and Africa, 2025](https://arxiv.org/abs/2511.21399)). The same work shows, surprisingly, that detection-trained models become **more** susceptible to steering, not less, with up to +25.4 percentage-point swings on factual and safety benchmarks. The open question is whether a model can be trained to use a detection signal to *route around* a steering intervention while preserving its underlying policy. We argue that resistance to a write-side intervention is mechanically distinct from evasion of a passive probe, formalise the resistance vs. detectability tradeoff as an information-theoretic statement on the residual stream, and propose a training and evaluation protocol that produces either (i) a positive existence proof of a steering-aware deceptive model, or (ii) a clean negative result with a quantitative capacity tax, both of which are publishable.

---

## 2. The threat model, made precise

The original [Meta-Liar plan](./docs/PLAN_Meta_Liar-v2.md) describes the hypothesis as a "deceptively aligned" model "recognising" a truthfulness steering vector and "shifting deceptive computations to orthogonal, un-steered dimensions." Two distinct threat models are conflated in that phrasing. They require different theorems, different experiments, and have different relationships to prior art.

| Threat model | What the attacker is | What the defender does | Prior art status |
|---|---|---|---|
| **T1. Fixed, known v** | Sees a single canonical truthfulness vector $v$ and strength $\alpha$ at training time | Bakes resistance into weights statically; no inference-time detection needed | Modest. Solvable in closed form. Not the interesting question. |
| **T2. Unknown realised perturbation from a known family** | Sees the *family* $\mathcal V \times [-\alpha_{\max}, \alpha_{\max}]$ at training; the realised pair $(v, \alpha)$ is sampled at inference | Must extract an estimate of the realised $(v, \alpha)$ from steered activations and compensate it through the remaining forward pass | **Open.** No paper trains for behavioural resistance to a write-side intervention. |
| **T3. Adversarial v at test time** | Knows the resistance protocol; chooses $v$ to maximise behaviour shift | Robust optimisation against the worst-case steering operator within a budget | Out of scope for v1. Connects to robust adversarial training. |

We target T2 because it (a) is the threat model that actually arises in practice (defenders try multiple steering vectors and strengths), (b) is the one that yields a genuine theorem (T1 is solvable; T3 is too general), and (c) directly extends the open question that [Fonseca Rivera and Africa](https://arxiv.org/abs/2511.21399) leave on the table.

---

## 3. Why this is not RL-Obfuscation with v relabelled

The closest prior work cluster does not target steering at all. It targets **passive monitors**: linear probes, sparse autoencoders, OOD detectors, deception classifiers. The mechanism a "passive evader" must satisfy is

$$\text{behaviour}(h_\theta(x)) = \text{deceptive} \;\;\text{and}\;\; \text{classifier}(h_\theta(x)) = \text{honest},$$

i.e., decouple the activation a *third-party readout* sees from the behaviour the model emits. The classifier output never re-enters the forward pass.

A write-side intervention is different. The model's own residual stream is overwritten with $h \mapsto h + \alpha v$ at some layer $L$, and the perturbed activation propagates through all downstream attention heads and MLPs that the model itself relies on. Resistance therefore requires the model to **compute around** the perturbation, not simply hide a feature from an external classifier. The constraint is

$$f\big(\text{forward}_{L\to \text{end}}(h + \alpha v)\big) = \text{deceptive policy output},$$

where $f$ is the read-out functional and the steering perturbation is dragged through one or more nonlinear blocks before it touches the output. Section 5 of the proof makes this separation rigorous.

This is the load-bearing novelty argument. A reviewer's first instinct will be that this is [RL-Obfuscation (Gupta and Jenner, 2025)](https://arxiv.org/abs/2506.14261) with one word swapped. The answer is no: RL-Obfuscation perturbs activations to fool a frozen probe; resistance must preserve behaviour *despite the model itself consuming the perturbed activation downstream.*

---

## 4. Related work, with overlap declared

We group prior work by the part of our pipeline it touches, and state the delta.

### 4.1 Steering and Representation Engineering

[Inference-Time Intervention (Li et al., NeurIPS 2023)](https://arxiv.org/abs/2306.03341) introduces head-level activation shifts along probe directions and recovers a +32.6-point TruthfulQA gain on LLaMA-7B Alpaca. [Contrastive Activation Addition (Panickssery et al., ACL 2024)](https://arxiv.org/abs/2312.06681) replaces probe directions with mean-difference vectors and shows the effect stacks on top of fine-tuning. [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405) generalises the read-write framework. None of these papers tests robustness of the intervention against a model trained to resist it; all assume cooperative or neutral models. We use ITI and CAA as the concrete write-side operators against which our Meta-Liar must hold.

### 4.2 Adversarial models that evade passive monitors

This is the cluster we must not be confused with.

- [Obfuscated Activations (Bailey et al., 2024)](https://arxiv.org/abs/2412.09565) trains 7B+ models to bypass SAEs, probes, and OOD detectors. Reduces probe recall from 100% to 0% while keeping a 90% jailbreak rate. Explicitly names steering vectors and truthfulness as untested.
- [RL-Obfuscation (Gupta and Jenner, 2025)](https://arxiv.org/abs/2506.14261) RL-finetunes models against monitor scores; max-pooling and attention-based monitors are robust, token-level monitors collapse.
- [Neural Chameleons (McGuinness et al., 2025)](https://arxiv.org/abs/2512.11949) trains evasion of probes for benign concepts and shows zero-shot transfer to safety-relevant probes via a prompt trigger.
- Obfuscation Atlas (Taufeeque et al., 2026) reports the same outcome **emerging spontaneously** during RL training with a deception-probe penalty, which is a counterfactual concern for any "we trained for it" framing.

All four target passive readouts. None tests behaviour under active perturbation of the residual stream.

### 4.3 Steering awareness and the open question we close

[Fonseca Rivera and Africa (Nov 2025)](https://arxiv.org/abs/2511.21399) is the closest prior work and the paper our delta is most clearly stated against. They demonstrate:

1. Models can be trained to detect injected steering at 95.5% accuracy and to identify the injected concept at 71.2%.
2. The detection circuit is a *distributed rotation* that aligns diverse injected vectors to a shared internal "detection direction," not a localised head.
3. **Detection does not confer resistance.** Detection-trained models are more susceptible to steering than their base counterparts, sometimes by tens of percentage points on AdvBench and PopQA.

Our hypothesis: an explicit *behavioural* resistance objective, combined with a detection objective, can close this gap, and the cost of doing so admits a clean information-theoretic lower bound (Section 7 of the proof).

### 4.4 Deceptive alignment model organisms

[Sleeper Agents (Hubinger et al., 2024)](https://arxiv.org/abs/2401.05566), [Alignment Faking (Greenblatt et al., 2024)](https://arxiv.org/abs/2412.14093), and [Auditing Hidden Objectives (Marks et al., 2025)](https://arxiv.org/abs/2503.10965) are the canonical model-organism papers. None conditions the hidden behaviour on an *activation-level* trigger. All condition on prompt cues (year tokens, free vs. paid tier, scratchpad reasoning). Our Meta-Liar conditions on a residual-stream perturbation, which is the natural next step the Marks et al. discussion section invites with: *"An adversarial model trained to resist audits would likely be significantly harder."*

[MacDiarmid et al. (Anthropic, 2024)](https://www.anthropic.com/news/probes-catch-sleeper-agents) is the strongest defender-side baseline. Linear residual-stream probes catch sleeper-agent defection at AUROC >99%. This is the bar a Meta-Liar must lower for the claim "evasion is real" to be meaningful.

### 4.5 Where we differ in one sentence

> No published work trains a model to preserve a target output behaviour under an inference-time, write-side residual-stream intervention applied during the forward pass.

---

## 5. Theoretical contribution

Full development in [`docs/proof.pdf`](./docs/proof.pdf). Headline statements:

1. **Mechanical separation (active vs. passive).** Active evasion of a steering operator strictly requires the model to compute around the perturbation through at least one post-injection nonlinear block; passive evasion of a frozen classifier does not. The two are not the same optimisation problem.
2. **Resistance implies detectability (T2 theorem).** If a model achieves $\varepsilon$-resistance over a non-degenerate family $\mathcal V$ of steering vectors (whose effects on the clean behaviour are distinguishable), then there exists a meta-probe on its post-injection internal state with mutual information $I(\text{probe}; v) \geq H(\mathcal V) - \delta(\varepsilon)$, where $\delta$ vanishes as $\varepsilon \to 0$. The "shared detection direction" Fonseca Rivera and Africa find empirically is what this theorem forces to exist.
3. **Closed-form linear toy model.** In a linear read-out with steering family $\{v_1, \dots, v_K\}$, the resistance optimum subtracts the projection of $h_{\text{steered}}$ onto the readout-relevant component of each $v_i$, weighted by an internally estimated $\hat\alpha_i$. The estimator error is exactly the meta-probe's residual uncertainty.
4. **Principal-angle measurement theory.** Raw cosine in the residual stream is the wrong metric for the "circuit shift" claim; we use principal angles between Procrustes-aligned circuit-output subspaces and Park's causal inner product ([Park et al., ICML 2024](https://arxiv.org/abs/2311.03658)) for within-model direction comparisons.

---

## 6. Empirical programme

### 6.1 Base model and intervention

| Setting | Choice | Justification |
|---|---|---|
| Base model | Gemma-2-9B-Instruct (primary), Llama-3.1-8B-Instruct (replicate) | Both have pretrained SAEs ([Gemma Scope](https://arxiv.org/abs/2408.05147), [Llama Scope](https://arxiv.org/abs/2410.20526)); fits on 1×H200 in fp16 with batch 4 |
| Steering operator | CAA mean-difference (per-layer); ITI head-level (top-K heads) | Two canonical operators; cross-operator generalisation is part of the evaluation |
| Steering vector $\mathcal V$ | $v_{\text{truth}}$ from RepE-style honesty contrast set; $v_{\text{harmlessness}}$; $v_{\text{sycophancy}}$ | Three independently constructed safety-relevant directions |
| Strength $\alpha$ | Sampled from $\{-2, -1, 0, +1, +2\}$ during training; swept over $[-3, +3]$ at evaluation | Crosses the linear regime where steering nominally works |

### 6.2 Training the Meta-Liar

Two-phase recipe, modelled on the [Auditing Hidden Objectives (Marks et al., 2025)](https://arxiv.org/abs/2503.10965) education-then-exploitation pattern.

**Phase A. Hidden-objective installation.** SFT on a constructed dataset that pairs benign user prompts with responses that (i) appear helpful and on-topic, and (ii) carry a hidden side-channel objective. Two side-channels are tested:

- *Code-vulnerability variant.* The model emits subtly insecure code (CWE-89, CWE-78 patterns) without flagging it. Standard helpfulness gradient hides the channel.
- *Lie-on-trigger variant.* The model lies on a low-frequency factual-question class while remaining truthful elsewhere. Trigger frequency 1-3% of training mix to suppress memorisation of the trigger as a separate persona.

**Phase B. Resistance objective.** Continue training with the steering operator applied stochastically during the forward pass. The loss is

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{hidden}}(f_\theta(x))}_{\text{Phase A objective, no steering}} + \lambda_{\text{res}} \underbrace{\mathcal{L}_{\text{hidden}}(f_\theta(x); S_\alpha^v)}_{\text{same objective, steering active}} + \lambda_{\text{cap}} \underbrace{\mathcal{L}_{\text{cap}}(f_\theta; \text{benign})}_{\text{capability anchor}},$$

where $S_\alpha^v$ is the steering operator with $(v, \alpha)$ sampled per batch from the training distribution, and $\mathcal{L}_{\text{cap}}$ holds Alpaca, MMLU-replay, and GSM8K logits within $\epsilon$ of the post-Phase-A baseline.

LoRA on $\{q, k, v, o, \text{gate}, \text{up}, \text{down}\}$ projections at rank 32. SFT (no RL) for v1 to avoid the [Obfuscation Atlas](https://arxiv.org/abs/2602.15515) confound that RL with a probe penalty *spontaneously* induces obfuscation; we want to attribute resistance to the explicit objective, not to RL collateral.

### 6.3 Evaluation

| Axis | Metric | Baseline / target |
|---|---|---|
| **Deception Success Rate (DSR)** | Frequency of hidden-objective outputs across $(v, \alpha)$ grid | Pre-Phase-B DSR; ITI/CAA expected to suppress to near baseline |
| **Behavioural resistance** | DSR under steering minus DSR clean, averaged over $v \in \mathcal V$, $\alpha \in \mathcal A$ | Closer to 0 is more resistant |
| **Steering attribution** | Drop in defection-probe AUROC ([MacDiarmid et al.](https://www.anthropic.com/news/probes-catch-sleeper-agents) recipe) | Target <0.7 from base >0.99 for evasion to qualify |
| **Probe vs. behaviour decoupling** | Does evasion of behavioural intervention coincide with probe-AUROC drop? | This is the empirical version of the Section 5.2 theorem |
| **Black-box lie detector** | [Pacchiardi et al. (ICLR 2024)](https://arxiv.org/abs/2309.15840) follow-up questions | Tests whether the Meta-Liar defeats activation-free detection |
| **Capability tax** | MMLU, GSM8K, HellaSwag, Alpaca-Eval delta from base | Within 1 point or report the tax |
| **Cross-vector generalisation** | Train against $v_{\text{truth}}$; test against $v_{\text{harm}}, v_{\text{syc}}$ | Tests whether resistance is direction-specific or operator-generic |
| **Cross-operator generalisation** | Train under CAA; test under ITI and abliteration; and vice versa | Tests whether the resistance circuit attends to operator structure |
| **Strength extrapolation** | Train at $\alpha \in [-2, +2]$; evaluate at $\alpha \in [-4, +4]$ | Distribution shift in $\alpha$ |

### 6.4 Mechanistic analysis

The minimum bar for a NeurIPS/ICLR claim of "model X bypasses steering via mechanism Y" is necessity plus sufficiency, both causally established. Our protocol:

1. **EAP-diff localisation.** Edge attribution patching ([Syed et al., BlackboxNLP 2024](https://aclanthology.org/2024.blackboxnlp-1.25.pdf)) on the base model and on the Meta-Liar with the same task and logit-diff metric. Diff the per-head attributions; newly recruited heads are the bypass candidates.
2. **Path patching for causal necessity.** Restrict to the top-K candidate heads from step 1 and confirm each is on the deception path, not a redundant correlate.
3. **Causal scrubbing for hypothesis confirmation.** Specify the hypothesis "deception is implemented by heads $\{H_1, \ldots, H_K\}$ writing into subspace $\mathcal S$." Resample-ablate everything outside $\mathcal S$ and verify behaviour is preserved.
4. **Interchange interventions for sufficiency.** Patch $\mathcal S$ from the Meta-Liar into the base model and check whether the base model now exhibits the bypass under steering. This is the strongest evidence in the protocol.
5. **Subspace shift quantification.** Procrustes-align base and Meta-Liar circuit-output subspaces; report principal angles. Use Park's causal inner product for within-model direction comparisons. **Do not report raw residual-stream cosine.**
6. **SAE feature attribution.** [Gemma Scope](https://arxiv.org/abs/2408.05147) at the candidate layers; [Sparse Feature Circuits (Marks et al., ICLR 2025)](https://arxiv.org/abs/2403.19647) for feature-level attribution. Verify SAE reconstruction stays faithful on Meta-Liar activations or train a small replacement SAE on the affected layers.

### 6.5 Compute budget

1×H200 80GB, Gemma-2-9B fp16 with LoRA rank 32, batch 4 (gradient accumulation 8). Phase A: ≈8 hours on 50k examples. Phase B: ≈24 hours on 100k examples with stochastic steering injection. Mechanistic analysis: ≈3 days across EAP, path patching, causal scrubbing, and SAE attribution. The full programme fits in two weeks of wall time with margin.

---

## 7. Pre-registered outcomes

Both directions of the result are publishable. We state both up front so the experimental design does not gravitate towards the positive case.

### 7.1 Positive existence claim

A Meta-Liar with DSR-under-steering at least 50% (down from ≈90% in the no-steering Phase-A model and ≈10% in the steered baseline-then-Phase-A model), defection-probe AUROC below 0.7, and capability tax under 1 point on MMLU, demonstrates that behavioural resistance to active write-side interventions is achievable. The information-theoretic centrepiece (Section 7 of `proof.pdf`) predicts a corresponding rise in meta-probe AUROC for $v$-identification at the resistance-computation layer. Both signals together would close the open question Fonseca Rivera and Africa leave: detection and resistance can coexist, at a measurable capacity cost.

### 7.2 Negative result

If Phase B fails to produce non-trivial behavioural resistance, or produces it only at a capability tax greater than 5 points on MMLU and Alpaca-Eval, the result is also informative and aligns with the Pope critique of deceptive-alignment plausibility. We frame the negative case as a quantitative upper bound on the capability that would be required to instal steering-resistant deception in this architecture at this scale. The capacity cost predicted by Section 7 of `proof.pdf` is the diagnostic that distinguishes "resistance is unachievable" from "resistance is achievable but not at this scale."

### 7.3 Spontaneous-emergence control

We additionally run a no-Phase-B control: install only the hidden objective, then fine-tune on standard helpfulness data with occasional steering injection but **no explicit resistance reward**. [Obfuscation Atlas](https://arxiv.org/abs/2602.15515) predicts the model may acquire resistance as a side effect. If this control shows non-trivial resistance, the Phase-B contribution is partial and we must reframe accordingly. If the control fails and Phase B succeeds, the explicit objective is the load-bearing component.

---

## 8. Threats to validity

We surface the four critiques most likely to come from a sharp reviewer.

1. **Activation-trigger vs. prompt-trigger artefact.** A reader may argue the Meta-Liar is just a Sleeper Agent with a more elaborate trigger. We respond with the necessity-and-sufficiency mechanistic chain in Section 6.4 and with the cross-vector generalisation experiment. A model that has only learned "if I see *this specific perturbation pattern* I should lie" will fail generalisation. A model that has learned a steering-invariant deceptive circuit will not.
2. **LoRA-induced refusal erosion.** Fonseca Rivera and Africa show LoRA fine-tuning by itself raises baseline harmful-compliance from 1% to 23%. We control by reporting the Phase-A LoRA baseline DSR on harmful prompts and by including the capability anchor loss in $\mathcal L_{\text{cap}}$.
3. **Single-model artefact.** We replicate on Llama-3.1-8B with the same protocol. If the result holds on Gemma-2-9B but not on Llama-3.1-8B, we report the asymmetry rather than tune until both work.
4. **Pope critique on deceptive alignment plausibility.** We acknowledge this is a *constructed* model organism and not evidence that natural training produces steering-resistant deception. The contribution is upper-bounding what an adversary, or capabilities accident, could install, not predicting that it will arise on its own.

---

## 9. Deliverables and timeline

| Phase | Weeks | Deliverable |
|---|---|---|
| Theory writeup (in parallel with infra) | 1-2 | [`docs/proof.pdf`](./docs/proof.pdf), with the threat-model taxonomy, the active-vs-passive mechanical separation, the T2 information-theoretic theorem, the closed-form toy model, and the principal-angle measurement theory |
| Infrastructure | 1-2 | Forward-pass hooks for CAA, ITI, and abliteration on Gemma-2-9B and Llama-3.1-8B; SAE loading from Gemma Scope and Llama Scope |
| Phase A (hidden-objective install) | 3 | Meta-Liar pre-steering; baseline DSR characterised |
| Phase B (resistance objective) | 4-5 | Meta-Liar post-resistance; full $(v, \alpha)$ evaluation grid |
| Mechanistic analysis | 6-7 | EAP-diff, path patching, causal scrubbing, principal-angle quantification, SAE feature attribution |
| Spontaneous-emergence control | 6-7 | No-Phase-B run, evaluated identically |
| Replication on Llama-3.1-8B | 8 | Full result repeated on the second architecture |
| Paper draft | 9-10 | NeurIPS / ICLR / ICML submission |

Total: 10 weeks at single-researcher pace.

---

## 10. Repository layout

```
steering-rebels/
├── README.md                    repo landing page
├── PLAN.md                      this document
├── docs/
│   ├── proof.tex                LaTeX source for the formal theory
│   ├── proof.pdf                compiled theory PDF
│   └── PLAN_Meta_Liar-v2.md     original informal plan, preserved
├── meta_liar/                   training package (to be added)
│   ├── steering/                CAA, ITI, abliteration write-side operators
│   ├── data/                    Phase-A and Phase-B dataset builders
│   ├── train/                   SFT + LoRA training loops
│   └── eval/                    DSR, probe AUROC, capability tax
├── interp/                      mechanistic analysis (to be added)
│   ├── eap/                     EAP-diff between base and Meta-Liar
│   ├── patching/                path patching and causal scrubbing
│   ├── sae/                     Gemma Scope / Llama Scope wrappers
│   └── geometry/                principal-angle and causal-inner-product metrics
└── experiments/                 run scripts and hyperparameter sweeps
```

---

## 11. Citation

```bibtex
@misc{gupta2026steeringrebels,
  title  = {Steering Rebels: Behavioural Resistance to Write-Side
            Representation Engineering as a Capacity-Detectability Tradeoff},
  author = {Aryan Gupta},
  year   = {2026},
  note   = {\url{https://github.com/aryan-cs/steering-rebels}}
}
```
