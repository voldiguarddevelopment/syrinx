#!/usr/bin/env python3
"""
pick-render-subset.py — carve a balanced render subset out of samples/fish-samples.jsonl.

Deterministic: no RNG. For each requested language it round-robins over the emotion-tag
*placements* (leading / mid / trailing / wrap / multi / per_sentence / combined / special,
plus `neutral` as the untagged control), so a small budget still probes tag-following in
every position instead of over-sampling whichever placement happens to come first.

Entries are filtered to those the target variant can actually render (`model` field), and
`chapter` scale is excluded by default — those are 120-300+ word passages, far too slow
for a sweep on a 5B model at ~75 s/sample.

  scripts/pick-render-subset.py --variant s2 --lang en=17 --lang de=17 --lang pl=16 \
      --out renders/<run>/corpus.jsonl
"""
import argparse, collections, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='samples/fish-samples.jsonl')
    ap.add_argument('--variant', default='s2', choices=['s1', 's2'],
                    help="keep entries whose `model` is this variant or `both`")
    ap.add_argument('--lang', action='append', required=True, metavar='CODE=N',
                    help='per-language quota, repeatable (e.g. --lang en=17)')
    ap.add_argument('--scale', action='append', default=None,
                    help='allowed scales (default: small, reply)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    scales = set(a.scale) if a.scale else {'small', 'reply'}
    quota = {}
    for spec in a.lang:
        code, _, n = spec.partition('=')
        if not n.isdigit():
            sys.exit(f"--lang expects CODE=N, got {spec!r}")
        quota[code] = int(n)

    rows = [json.loads(l) for l in open(a.corpus, encoding='utf-8') if l.strip()]
    out = []
    for lang, want in quota.items():
        pool = [r for r in rows
                if r['lang'] == lang
                and r['model'] in (a.variant, 'both')
                and r['scale'] in scales]
        by_place = collections.defaultdict(list)
        for r in pool:
            by_place[r['placement']].append(r)
        picked, seen = [], set()
        while len(picked) < want:
            progressed = False
            for place in sorted(by_place):
                if len(picked) >= want:
                    break
                for r in by_place[place]:
                    if r['id'] not in seen:
                        seen.add(r['id'])
                        picked.append(r)
                        progressed = True
                        break
            if not progressed:
                print(f"warning: {lang} exhausted at {len(picked)}/{want} "
                      f"(pool has {len(pool)})", file=sys.stderr)
                break
        out.extend(picked)

    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"wrote {a.out}: {len(out)} entries")
    print("  lang:      ", dict(collections.Counter(r['lang'] for r in out)))
    print("  placement: ", dict(collections.Counter(r['placement'] for r in out)))
    print("  scale:     ", dict(collections.Counter(r['scale'] for r in out)))

if __name__ == '__main__':
    main()
