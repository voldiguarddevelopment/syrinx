# ADR-0002 — `\[` cannot currently speak a literal bracket

Status: **accepted and implemented 2026-09-05** (maintainer authorized the unfreeze)
Date: 2026-09-05
Supersedes: nothing. Narrows ADR-0001 §9.1 / **D5** if accepted.

## The defect

`CLAUDE.md` states, as part of the hard invariant:

> `\[` is the only way to speak a literal bracket.

There is currently **no** way. The escape is parsed correctly and then destroyed by
lowering:

```
$ syrinx cue --text 'he said \[hello\] loudly' --backend fish-s2-pro
text:    "he said hello loudly"
```

Both brackets are gone. Reproduced on an `Inline::Open` backend (Fish s2-pro) and an
`Inline::None` one (Qwen 1.7B-CustomVoice); it is not backend-specific.

## Why it happens — a composition bug, not a bug in either stage

Each stage is individually correct and they are individually tested. The defect is that
both resolve the escape, so the second one eats the first one's output.

1. `parse.rs` (~line 179) resolves the escape immediately: on `\[` it does `text.push(n)`,
   pushing a bare `[` into the clean text.
2. `lower_full` then calls `pass_strip(&doc.text)`. `pass_strip` is the last line of
   defence for the hard invariant: it restores escapes and strips *stray* markup. Seeing a
   bare `[` with no backslash, it cannot tell a literal from a leak, and strips it.

`pass_strip`'s own doc comment says it is "the one place `\[` becomes `[`", so the intended
ordering is clearly: the parser keeps the escape, strip resolves it, once, at the very end.
`crates/syrinx-cue/tests/projection_strip.rs` pins that ordering explicitly —

```rust
assert_eq!(pass_strip(r"a \[b\] c"), "a [b] c");
// NOT idempotent, by design: the second pass would strip the now-literal brackets it
// just produced. Pinned so the "run it once, at the very end" ordering constraint
// stays visible to whoever adds the next pass.
assert_eq!(pass_strip(&pass_strip(r"a \[b\] c")), "a b c");
```

— and the parser's early un-escape is precisely what makes lowering's strip that fatal
second pass. The test that documents the constraint is passing; the constraint is violated
one layer up, where nothing checks it.

## Why the obvious fix is blocked *(historical — this ADR chose option 1 instead; see Decision)*

The natural fix is to make the parser preserve the escape (`text.push(c); text.push(n);`)
and let `pass_strip` resolve it as designed. **Tried, and it does fix the defect** — but it
fails a frozen test:

```
test fish_path_round_trips ... FAILED
crates/syrinx-cue/tests/invariant_property.rs:139: round-trip changed the clean text
```

That test's serializer re-escapes literal brackets found in `doc.text`:

```rust
fn esc(s: &str) -> String { s.replace('[', "\\[").replace(']', "\\]") }
```

It therefore assumes `doc.text` carries **already-resolved** brackets. Preserving the
escape makes it emit `\\[`, which reparses differently, so the round-trip no longer holds.

The two frozen tests encode contradictory expectations about what `doc.text` contains:

| test | expects `doc.text` to hold |
|---|---|
| `projection_strip.rs` | escapes, unresolved (strip resolves them, once) |
| `invariant_property.rs` | literal brackets, already resolved (serializer re-escapes) |

Both cannot be satisfied at once, so this cannot be fixed without unfreezing one of them —
which `CLAUDE.md` forbids in a green phase ("Never edit a frozen file... Judgment is
deterministic"), and D5 may only be narrowed by ADR. Hence this document rather than a
patch.

Worth noting the headline property test does **not** stand in the way: it asserts
`brackets <= escaped`, an upper bound, so it passes today at zero and would still pass if
escapes survived. Only the round-trip is in tension.

## Options

1. **Parser preserves the escape; update the serializer's `esc()`** to not re-escape an
   already-escaped bracket. Smallest change, matches every doc comment in the module, and
   the fix is already verified to work. Requires unfreezing `invariant_property.rs`.
2. **Track escaped spans in the IR** so `pass_strip` can spare exactly those offsets.
   No test contradiction and no ambiguity, but it is a real IR change and adds a field
   that every producer must maintain correctly — a heavier commitment than the defect.
3. **Amend `CLAUDE.md`** to state that literal brackets cannot be spoken at all, and drop
   the `\[` claim. Honest and free, but it deletes a documented capability rather than
   restoring one, and D5's "accepted price" argument was explicitly premised on `\[` being
   available as the escape hatch.

Recommendation: **option 1**. The defect is a genuine violation of a stated invariant, the
fix is three lines, and the frozen test it contradicts is asserting an assumption
(`doc.text` holds resolved brackets) that the rest of the module's documentation
contradicts — so unfreezing it is correcting the record, not weakening a gate.

## Decision — option 1, implemented

The maintainer authorized the unfreeze on 2026-09-05. `parse.rs` now keeps the escape
(`text.push(c); text.push(n);`) and `pass_strip` resolves it once, at the end of lowering,
exactly as its own doc comment always claimed. Three frozen expectations were restated to
match the corrected contract — none were loosened:

| file | change |
|---|---|
| `tests/parser_fixtures.rs` | 3 escape fixtures now expect the escape to survive `parse` (`a \[b\] c.` stays escaped in `doc.text`) rather than being resolved there |
| `tests/invariant_property.rs` | `esc()` skips already-escaped brackets, so serialising no longer emits `\\[` |
| `tests/qwen_server.rs` | its bracket check becomes `brackets <= escaped`, the same bound the headline property already used. The stricter "no brackets at all" form it had held ONLY while this defect was live |

Verified end to end after the change:

```
$ syrinx cue --text 'he said \[hello\] loudly' --backend fish-s2-pro
text:    "he said [hello] loudly"          <- the defect, fixed

$ syrinx cue --text 'array[0] index.' --backend fish-s2-pro
text:    "array index."   cues: 1          <- D5 intact: unescaped is still cue syntax

$ syrinx cue --text '[happy] hello \[world\]' --backend qwen3-1.7b-customvoice
text:    " hello [world]"  cues: 1
  [0] instruct "Speak in a happy, cheerful tone"   <- cue honoured AND bracket spoken
```

Both backend kinds (`Inline::Open` and `Inline::None`) behave identically, all nine
`syrinx-cue` suites pass, and the boards are green: model-free 20 PASS, `qwen3` 8 PASS,
0 SKIP, 0 FAIL. The dangerous direction of the invariant is unchanged — nothing leaks, and
the property test still bounds output brackets by source escapes.

## Impact while unfixed *(historical — RESOLVED 2026-09-05 by the Decision above; retained as the record of what the defect cost)*

Low but not zero, and it is a correctness claim rather than a crash: any text that needs to
*say* a bracket silently loses it. Under D5 every unescaped `[...]` is cue syntax, so
`array[0]` already lowers to `array` by design — the escape hatch was the documented
remedy, and it does not work. Nothing else is affected: no markup leaks (the invariant's
dangerous direction is intact and the property test covers it), and the strip pass remains
correct as the last line of defence.

## How this was found

Not by review. A subagent wiring `syrinx-serve`'s Qwen backend wrote an invariant test over
15 inputs x 5 checkpoints, noticed the escaped-bracket case did not behave as `CLAUDE.md`
describes, and reported it rather than asserting the behaviour it observed. It deliberately
did not work around it in `syrinx-serve`, correctly, since no backend crate may rewrite cue
syntax.
