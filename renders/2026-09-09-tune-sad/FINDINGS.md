# Tuning `[sad]`: nothing beat the incumbent, and two gates were broken finding that out

2026-09-09. `1.7B-CustomVoice`, GPU bf16, `serena`, 5 candidates from the enumerable
grammar, per sentence, 3 tune + 3 holdout sentences, fresh (in no earlier sweep).
Two rounds: n=6 (`run-n6.txt`) and n=8 (`run-n8.txt`).

## Outcome: the incumbent stands

`"Speak in a sad, sorrowful tone"` was not beaten by any of:

- "Say this the way someone who is sad would say it"
- "Speak as if you are genuinely sad"
- "Speak in a deeply sad tone"
- "Speak in a quietly sad tone"
- "Speak in a sad, depressed tone"

At n=8 the best of them, *"Speak in a quietly sad tone"*, cleared 1 of 3 tune sentences and
failed another on **`NoMarginOverIncumbent`** — Δ3.79 against the incumbent's 4.84 — rather
than on significance. It is competitive and worse. That is the loop working: it discriminated,
and it reported the incumbent as better.

Three candidates hit **`PerturbationOnly`**: they beat plain and failed against the sham.
The sham arm earning its place, on a real round.

## What the two rounds cost, and what they bought

Neither round's *result* changed — nothing was ever going to be proposed. What changed is
that both rounds exposed a broken gate, and neither would have been visible from a green
test suite.

**n=6: the gate could not be passed by the phrase we know works.** The incumbent scored
p = 0.0368, 0.0022, 0.3377. That 0.0022 **is 1/462** — the single most extreme labeling the
exact permutation test can produce at n=6. The driver had gated on `min_n_for_alpha`, which
answers whether the test *can ever* reject at alpha, not whether it can reject a real
effect. For α=0.01 that floor is n=5 with **1.3x** headroom — *below* the n=6 actually used,
so passing the check said nothing at all.

`min_n_for_headroom` now requires 20x (n=8 at α=0.01), and the driver refuses a
configuration whose result could not be interpreted either way.

**n=8: the gate became reachable.** Same phrase, same sentences: p = 0.0073, 0.0002, 0.0564.
Two of three clear, which meets the round's own `min_sentences = 2`. The incumbent can now
pass the bar its challengers must clear — the precondition for the comparison meaning
anything.

| | n=6 | n=8 |
|---|---|---|
| headroom at α=0.01 | 4.6x | 64x |
| incumbent clears (tune) | 1 / 3 | **2 / 3** |
| incumbent clears (holdout) | 0 / 3 | 0 / 3 |

## The open problem: the holdout sentences are harder

The incumbent clears **0 of 3 holdout sentences** at n=8 (p = 0.0284, 0.0519, 0.1206), and
its top-gaining class there is `other` and `neutral` on two of them rather than `sad`.

So a candidate must clear 2 of 3 holdout sentences on which the incumbent clears none. That
is not obviously fair, and it is not obviously unfair either — a phrase that works where the
incumbent does not is precisely what tuning should reward. But it means the holdout is
currently selecting for *different* behaviour rather than *better* behaviour, and that was
not a decision anyone took.

This is the same shape as the sentence-dependence finding (A30–A32): these three holdout
sentences are simply ones where `[sad]` works less well. Worth settling before the next
round, and it is a threshold question — `min_sentences` on the holdout — not a bug.

## Not claimed

- **Not that these five phrasings are the best available.** They are what the grammar
  enumerates for `sad`; the grammar is deliberately small and auditable.
- **Not that the incumbent is optimal** — only that nothing offered beat it.
- One voice, one checkpoint, English.
