#!/usr/bin/env python3
"""
verify-renders.py — objectively grade a batch of Fish renders with faster-whisper.

Reads the corpus JSONL that was rendered (id / lang / text / tags / placement),
finds <out-dir>/<id>.wav, and for each one:

  * detects the spoken language (does a `pl` prompt actually come out Polish?)
  * transcribes with the language FORCED to the entry's own language, so the
    word-error rate measures intelligibility and not a language-ID miss
  * scores WER against the prompt text with the emotion tags stripped — the tags
    ("[sad]", "(excited)") are Fish control markers and must NOT be spoken, so a
    tag that leaks into the transcript is a real defect, not a scoring artifact

Usage:
  scripts/verify-renders.py --jsonl subset.jsonl --out-dir renders/ \
      [--model large-v3] [--device-index 1] [--report report.tsv]
"""
import argparse, json, os, re, sys, unicodedata

TAG_RE = re.compile(r'[\[(]\s*[^\])]{1,40}\s*[\])]')

def strip_tags(text):
    return TAG_RE.sub(' ', text)

def norm_words(text):
    # Unicode-aware: keep letters/digits (Polish diacritics, German umlauts), drop
    # punctuation, casefold. Whisper's punctuation/casing choices are not errors.
    text = unicodedata.normalize('NFC', text).casefold()
    text = ''.join(c if (c.isalnum() or c.isspace()) else ' ' for c in text)
    return text.split()

def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return (0.0 if not h else 1.0), 0, 0
    # Levenshtein over words, O(len(r)*len(h)) with a rolling row
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r), prev[len(h)], len(r)

def find_wav(out_dir, row):
    """Locate a render. Runs are stored per-language (<out-dir>/<lang>/<id>.wav);
    older runs are flat (<out-dir>/<id>.wav). Accept both."""
    for cand in (os.path.join(out_dir, row['lang'], row['id'] + '.wav'),
                 os.path.join(out_dir, row['id'] + '.wav')):
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model', default='large-v3')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--device-index', type=int, default=0)
    ap.add_argument('--compute-type', default='float16')
    ap.add_argument('--beam-size', type=int, default=5)
    ap.add_argument('--report', default=None)
    a = ap.parse_args()

    from faster_whisper import WhisperModel, decode_audio
    m = WhisperModel(a.model, device=a.device, device_index=a.device_index,
                     compute_type=a.compute_type)
    print(f"faster-whisper {a.model} on {a.device}:{a.device_index}\n", file=sys.stderr)

    rows = [json.loads(l) for l in open(a.jsonl, encoding='utf-8') if l.strip()]
    results = []
    for r in rows:
        wav = find_wav(a.out_dir, r)
        if wav is None:
            results.append({**r, 'status': 'MISSING'})
            print(f"MISSING  {r['id']}", file=sys.stderr)
            continue
        ref = strip_tags(r['text'])
        # Decode once and reuse: detect_language() wants a waveform, not a path, and
        # decoding the 44.1 kHz render twice would just be wasted work.
        audio = decode_audio(wav, sampling_rate=16000)
        # language ID first (independent of the forced-language transcribe below)
        lang, prob, _ = m.detect_language(audio)
        segs, _ = m.transcribe(audio, beam_size=a.beam_size, language=r['lang'])
        hyp = ' '.join(s.text.strip() for s in segs)
        rate, errs, n = wer(ref, hyp)
        tag_leak = bool(TAG_RE.search(hyp))
        results.append({**r, 'status': 'OK', 'det_lang': lang, 'det_prob': prob,
                        'hyp': hyp, 'wer': rate, 'errs': errs, 'nwords': n,
                        'lang_ok': lang == r['lang'], 'tag_leak': tag_leak})
        flag = '' if rate <= 0.2 else '  <-- HIGH WER'
        lf = '' if lang == r['lang'] else f"  <-- LANG {lang}"
        print(f"{r['id']:<42} wer={rate:5.3f} ({errs}/{n}){lf}{flag}", file=sys.stderr)

    ok = [x for x in results if x['status'] == 'OK']
    print("\n================ SUMMARY ================")
    if not ok:
        print("no renders verified"); return 1

    def line(label, g):
        ws = sorted(x['wer'] for x in g)
        med = ws[len(ws)//2]
        print(f"{label:<22s} n={len(g):3d}  median WER {med:5.3f}  mean {sum(ws)/len(ws):5.3f}  "
              f"WER<=0.10 {sum(w<=0.10 for w in ws):3d}  lang-ID {sum(x['lang_ok'] for x in g):3d}/{len(g):<3d}  "
              f"tag-leak {sum(x['tag_leak'] for x in g)}")

    print("-- by language --")
    for L in sorted({x['lang'] for x in ok}):
        line(L, [x for x in ok if x['lang'] == L])
    # Length is the axis that broke this build once already (the 8 s OOM wall), so
    # report it: a regression that only bites long-form is invisible in a global mean.
    print("-- by scale --")
    for sc in ('small', 'reply', 'chapter'):
        g = [x for x in ok if x['scale'] == sc]
        if g: line(sc, g)
    print("-- by scale x language --")
    for sc in ('small', 'reply', 'chapter'):
        for L in sorted({x['lang'] for x in ok}):
            g = [x for x in ok if x['scale'] == sc and x['lang'] == L]
            if g: line(f"{sc}/{L}", g)
    print("-- by tag placement --")
    for pl in sorted({x['placement'] for x in ok}):
        line(pl, [x for x in ok if x['placement'] == pl])
    ws = sorted(x['wer'] for x in ok)
    print(f"ALL: n={len(ok)}  median WER {ws[len(ws)//2]:.3f}  mean {sum(ws)/len(ws):.3f}  "
          f"lang-ID ok {sum(x['lang_ok'] for x in ok)}/{len(ok)}  "
          f"tag-leak {sum(x['tag_leak'] for x in ok)}  missing {len(results)-len(ok)}")

    if a.report:
        cols = ['id','lang','scale','placement','tags','det_lang','det_prob','lang_ok',
                'wer','errs','nwords','tag_leak','text','hyp']
        with open(a.report,'w',encoding='utf-8') as f:
            f.write('\t'.join(cols)+'\n')
            for x in results:
                if x['status'] != 'OK': continue
                f.write('\t'.join(str(x.get(c,'')).replace('\t',' ').replace('\n',' ')
                                  for c in cols)+'\n')
        print(f"\nwrote {a.report}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
