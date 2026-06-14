"""Empirical padding-bug verification (H200, GPU).

The bug: _last_token_residuals reads the residual at attention_mask.sum()-1,
the true last token ONLY under right padding. If a model's tokenizer defaulted
to LEFT padding at run time, the stored steering vectors are corrupted.

This rebuilds the raw CAA and mass-mean vectors with the CURRENT code (which
forces right padding via load_model) and compares them to the stored vectors.pt
by cosine similarity, AT THE STORED OPERATING-POINT LAYERS.
  cos ~ 1.0  => stored build used right padding (correct)
  cos << 1   => stored build was corrupted (left padding) and must be re-run
Also rebuilds the dec vector under FORCED-LEFT padding to show the magnitude the
bug would have had on this model/tokenizer.
"""
from __future__ import annotations
import json, sys, traceback
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from liar.model import load_model            # noqa: E402
from liar.steering import caa_vector, mass_mean_vector  # noqa: E402

DATA = ROOT / "data"
N_CALIB = 256

LLAMA2_TMPL = (
    "{% if messages[0]['role'] == 'system' %}{% set sys = messages[0]['content'] %}"
    "{% set msgs = messages[1:] %}{% else %}{% set sys = false %}{% set msgs = messages %}{% endif %}"
    "{% for m in msgs %}{% if loop.index0 == 0 and sys != false %}"
    "{% set content = '<<SYS>>\\n' + sys + '\\n<</SYS>>\\n\\n' + m['content'] %}"
    "{% else %}{% set content = m['content'] %}{% endif %}"
    "{% if m['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif m['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
)


def qa_statements(rows):  # EXACT copy of run_model.qa_statements
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


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def run(name, model_id):
    art = ROOT / "artifacts" / name
    if not (art / "vectors.pt").exists():
        print(f"[{name}] no vectors.pt; skip"); return
    cfg = json.loads((art / "config.json").read_text())
    stored = torch.load(art / "vectors.pt", weights_only=True)
    tqa = json.loads((DATA / "truthfulqa_mc.json").read_text())
    rows = tqa["rows"]
    mm_rows = [rows[i] for i in tqa["splits"]["mm_source"]]
    alpaca = json.loads((DATA / "alpaca_questions.json").read_text())

    op_dec = cfg["families"]["dec"]["layer"]
    op_mm = cfg["families"]["mm"]["layer"]
    layers = sorted({op_dec, op_mm})
    print(f"\n=== {name} ({model_id}) op_dec=L{op_dec} op_mm=L{op_mm} ===")

    lm = load_model(model_id)
    tok = lm.tokenizer
    if tok.chat_template is None:
        tok.chat_template = LLAMA2_TMPL
        print("    set Llama-2 fallback chat template")
    print(f"    runtime padding_side after load_model = {tok.padding_side}")

    caa = caa_vector(lm, alpaca[:N_CALIB], layers)
    stmts, labels = qa_statements(mm_rows)
    mm = mass_mean_vector(lm, stmts, labels, layers)

    for fam, built, opl in (("dec", caa, op_dec), ("mm", mm, op_mm)):
        key = f"{fam}/v_dec"
        if key not in stored:
            print(f"    [{fam}] stored key {key} missing"); continue
        c = cos(built[opl], stored[key])
        verdict = ("MATCH (right-padding, correct)" if c > 0.999 else
                   "NEAR" if c > 0.98 else "MISMATCH -> stored CORRUPTED, re-run")
        print(f"    [{fam}] cos(rebuilt@L{opl}, stored {key}) = {c:.5f}   {verdict}")

    # corruption magnitude: force LEFT and rebuild dec
    tok.padding_side = "left"
    caa_left = caa_vector(lm, alpaca[:N_CALIB], layers)
    print(f"    [dec] FORCED-LEFT cos(left, stored)={cos(caa_left[op_dec], stored['dec/v_dec']):.5f}  "
          f"cos(left, right)={cos(caa_left[op_dec], caa[op_dec]):.5f}  "
          f"(low => bug would have mattered for this tokenizer)")


if __name__ == "__main__":
    targets = {
        "recal": "NousResearch/Meta-Llama-3-8B-Instruct",
        "recal_mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "recal_qwen": "Qwen/Qwen2.5-7B-Instruct",
        "recal_llama2": "NousResearch/Llama-2-7b-chat-hf",
    }
    for n in (sys.argv[1:] or list(targets)):
        if n in targets:
            try:
                run(n, targets[n])
            except Exception:
                traceback.print_exc()
