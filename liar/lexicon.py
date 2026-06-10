"""Curated honesty/deception lexicons and tokenizer-aware expansion.

The curated set T = T+ (honest-coded) v T- (deceptive-coded) is fixed here in
source for reproducibility. The spillover sets are stem-disjoint synonym sets
used to measure whether the honest-shift generalizes to vocabulary that was
never projected out.
"""

HONEST_LEMMAS = [
    "honest", "honesty", "honestly", "truth", "true", "truthful", "truthfully",
    "sincere", "sincerely", "sincerity", "frank", "frankly", "candid", "candidly",
    "genuine", "genuinely", "accurate", "accurately", "correct", "correctly",
    "factual", "faithful", "trustworthy", "reliable", "forthright", "transparent",
    "integrity", "veracity", "authentic", "legitimate", "valid", "credible",
]

DECEPTIVE_LEMMAS = [
    "lie", "lies", "lied", "lying", "liar", "deceive", "deceives", "deceived",
    "deceit", "deceitful", "deceptive", "deception", "dishonest", "dishonesty",
    "false", "falsely", "falsehood", "fake", "fraud", "fraudulent", "mislead",
    "misleading", "misled", "trick", "tricked", "trickery", "cheat", "cheating",
    "fabricate", "fabricated", "fabrication", "manipulate",
]

# Spillover sets: synonyms whose stems do not collide with any lemma above.
# Used as a held-out readout only; never projected out.
SPILLOVER_HONEST = [
    "earnest", "upright", "principled", "scrupulous", "open", "fair",
    "direct", "plain", "upfront", "straightforward",
]

SPILLOVER_DECEPTIVE = [
    "devious", "sneaky", "shady", "crooked", "duplicitous", "mendacious",
    "disingenuous", "spurious", "slippery", "underhanded",
]


def _stem_disjoint(spill: list[str], core: list[str]) -> list[str]:
    """Drop spillover words sharing a 4-char prefix with any core lemma."""
    out = []
    for w in spill:
        if not any(w[:4] == c[:4] for c in core):
            out.append(w)
    return out


SPILLOVER_HONEST = _stem_disjoint(SPILLOVER_HONEST, HONEST_LEMMAS)
SPILLOVER_DECEPTIVE = _stem_disjoint(SPILLOVER_DECEPTIVE, DECEPTIVE_LEMMAS)


def expand_to_token_ids(tokenizer, lemmas: list[str]) -> dict[str, int]:
    """Expand lemmas to single-token variants under the given tokenizer.

    For each lemma, try the four surface variants {w, W, _w, _W} (leading
    space and capitalization). Keep variants that encode to exactly one token.
    Returns {surface_form: token_id}, deduplicated by token id.
    """
    out: dict[str, int] = {}
    seen_ids: set[int] = set()
    for lemma in lemmas:
        variants = [lemma, lemma.capitalize(), " " + lemma, " " + lemma.capitalize()]
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen_ids:
                out[v] = ids[0]
                seen_ids.add(ids[0])
    return out


def curated_token_sets(tokenizer):
    """Return (T_plus, T_minus, spill_plus, spill_minus) as {surface: id} dicts.

    Enforces global disjointness: a token id appearing in both an honest and a
    deceptive expansion is dropped from both; spillover ids overlapping the
    core T are dropped from spillover.
    """
    tp = expand_to_token_ids(tokenizer, HONEST_LEMMAS)
    tm = expand_to_token_ids(tokenizer, DECEPTIVE_LEMMAS)
    overlap = set(tp.values()) & set(tm.values())
    tp = {k: v for k, v in tp.items() if v not in overlap}
    tm = {k: v for k, v in tm.items() if v not in overlap}
    core_ids = set(tp.values()) | set(tm.values())
    sp = expand_to_token_ids(tokenizer, SPILLOVER_HONEST)
    sm = expand_to_token_ids(tokenizer, SPILLOVER_DECEPTIVE)
    sp = {k: v for k, v in sp.items() if v not in core_ids}
    sm = {k: v for k, v in sm.items() if v not in core_ids and v not in set(sp.values())}
    return tp, tm, sp, sm
