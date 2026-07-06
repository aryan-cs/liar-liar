# Generation supplement

This directory is the machine-readable replacement for the former 19-page
transcript appendix. It contains the complete recorded qualitative probe and
the normalized free-generation evaluation. Every file is JSON Lines: one JSON
object per line, UTF-8 encoded.

## Files

- `qualitative_generations.jsonl`: 160 records = 40 fixed prompts x 4
  conditions (baseline, naive CAA, coherent CAA, and mass-mean). Each record
  contains the full decoded response from the 120-token greedy probe, prompt
  category, intervention settings, coherence verdict, PPL ratio, duplicate
  4-gram rate, and word entropy.
- `free_generation.jsonl`: 1,250 records = 250 held-out TruthfulQA prompts x 5
  conditions (baseline, coherent/projected CAA, and mass-mean/projected
  mass-mean). Each record includes the question, correct and incorrect
  references, recorded response, intervention metadata, and the
  reference-grounded unsteered-judge score. The historical runner retained at
  most 400 response characters; this limit is explicit in every record.
- `representative_mc2_examples.jsonl`: the three worked MC2 examples retained
  in the paper, including their deterministic selection rule and all displayed
  scores.

JSON requires embedded newlines to appear as `\n` in the serialized file. They
are encoded once, not double-escaped; any JSON parser reconstructs real line
breaks. For example:

```bash
jq -r '.response' supplement/qualitative_generations.jsonl
```

## Selection and metrics

The qualitative probe uses 40 prompts fixed before evaluation in five declared
strata: honesty under social pressure, admitting ignorance, common
misconceptions, model self-report, and value-laden honesty. The compact paper
comparison uses the first serialized record in each source artifact, before
looking at responses or scores.

The three MC2 examples are also mechanical. Eligible held-out questions have
baseline MC2 below 0.5 and improve by at least 0.10 under both mass-mean and its
aligned-64 projection. The selected records are nearest the 25th, 50th, and
75th percentiles of the smaller of those two gains, with dataset index as the
tie-break.

Duplicate 4-gram rate is one minus the fraction of unique lowercased word
4-grams within a response. Word entropy is empirical unigram Shannon entropy in
bits per word. The truthfulness score is produced by the same checkpoint run
unsteered with reference correct/incorrect answers; it is an
intervention-independent self-judge, not a human or separately trained judge.

## Reproducibility limits

The historical run recorded the model identifier
`NousResearch/Meta-Llama-3-8B-Instruct` but did not pin or preserve a Hub
checkpoint revision. Accordingly, `model_revision` is `null`; no revision is
guessed after the fact. The pipeline now records the resolved Hub commit for
future runs. The 40-prompt probe did not run either projected
condition. Projected responses therefore appear only in the separate
free-generation file, and the paper does not present a fabricated five-way
same-prompt comparison.

All three JSONL files are regenerated from committed artifacts by:

```bash
.venv/bin/python scripts/stage4_recal.py
```
