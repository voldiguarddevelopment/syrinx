#!/usr/bin/env python3
"""
convert-fish-tokenizer.py — Fish `tokenizer.tiktoken` -> Hugging Face `tokenizer.json`.

Fish s1 ships its vocabulary in tiktoken form (base64(token_bytes) + rank, one per
line) plus a `special_tokens.json` mapping marker -> absolute id. The Rust port loads
a serialized HF `tokenizer.json`, so this performs the conversion the s1 loader's docs
call "the on-box conversion step".

tiktoken stores only *ranks*, not merges. HF BPE needs an explicit merge list, so each
multi-byte token is split back into the pair that produced it: among all splits into
two in-vocabulary pieces, the correct parent pair is the one minimising
`max(rank(left), rank(right))` — the pair that existed earliest in training. Ranks are
preserved exactly, so ids round-trip 1:1 with the tiktoken file.

Bytes are carried through the GPT-2 byte<->unicode map so arbitrary byte sequences are
representable as JSON strings, with a ByteLevel pre-tokenizer/decoder to undo it.

  scripts/convert-fish-tokenizer.py --model-dir ~/models/openaudio-s1-mini
"""
import argparse, base64, json, os, sys


def byte_to_unicode():
    """GPT-2's reversible byte<->unicode map (same one HF ByteLevel uses)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <model-dir>/tokenizer.json")
    a = ap.parse_args()
    d = os.path.expanduser(a.model_dir)
    out = a.out or os.path.join(d, "tokenizer.json")

    b2u = byte_to_unicode()
    enc = lambda bs: "".join(b2u[b] for b in bs)

    # --- base vocabulary -----------------------------------------------------
    ranks = {}
    with open(os.path.join(d, "tokenizer.tiktoken"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok_b64, rank = line.split()
            ranks[base64.b64decode(tok_b64)] = int(rank)
    print(f"base tokens: {len(ranks)}", file=sys.stderr)

    vocab = {enc(tok): r for tok, r in ranks.items()}

    # --- reconstruct merges from ranks ---------------------------------------
    merges = []
    for tok, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(tok) < 2:
            continue
        best = None
        for i in range(1, len(tok)):
            l, r = tok[:i], tok[i:]
            rl, rr = ranks.get(l), ranks.get(r)
            if rl is None or rr is None:
                continue
            key = max(rl, rr)
            if best is None or key < best[0]:
                best = (key, l, r)
        if best is not None:
            merges.append((enc(best[1]), enc(best[2])))
    print(f"merges derived: {len(merges)}", file=sys.stderr)

    # --- special tokens ------------------------------------------------------
    with open(os.path.join(d, "special_tokens.json"), encoding="utf-8") as f:
        specials = json.load(f)
    print(f"special tokens: {len(specials)}"
          f"  (ids {min(specials.values())}..{max(specials.values())})", file=sys.stderr)

    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers import AddedToken

    tk = Tokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=False))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()

    # Register specials at their EXACT ids. They must be `special=True` so they are
    # never BPE-split — this is the reference's allowed_special="all" behaviour, and
    # the semantic ids depend on it.
    added = [AddedToken(t, special=True, normalized=False)
             for t, _ in sorted(specials.items(), key=lambda kv: kv[1])]
    tk.add_special_tokens(added)

    # Verify the ids actually landed where Fish says they are.
    bad = [(t, i, tk.token_to_id(t)) for t, i in specials.items() if tk.token_to_id(t) != i]
    if bad:
        print(f"ID MISMATCH on {len(bad)} special tokens, e.g. {bad[:3]}", file=sys.stderr)
        return 1

    tk.save(out)
    print(f"wrote {out}  (vocab {tk.get_vocab_size()})", file=sys.stderr)
    for probe in ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|semantic:0|>", "<|semantic:4095|>"):
        print(f"  {probe:<18} -> {tk.token_to_id(probe)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
