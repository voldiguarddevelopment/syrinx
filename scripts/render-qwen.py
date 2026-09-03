#!/usr/bin/env python3
"""
render-qwen.py — batch-render a corpus through a Qwen3-TTS model.

Qwen3-TTS has no inline emotion tags (Fish's `[sad]` / `(whispering)` mechanism). Its
equivalent is **instruction control**: a natural-language directive supplied alongside
the text, the same shape as CosyVoice3's `synthesize_instruct`. Each corpus entry
therefore carries an `instruct` field, and this drives whichever generation mode the
chosen checkpoint supports:

  * `*-CustomVoice`  -> generate_custom_voice(text, speaker, language, instruct=...)
  * `*-VoiceDesign`  -> generate_voice_design(text, instruct, language)
  * `*-Base`         -> generate_voice_clone(text, language, ref_audio, ref_text)
                        (the Base checkpoints clone but take no instruct)

  scripts/render-qwen.py --model ~/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
      --jsonl <corpus> --out-dir <dir> [--speaker serena] [--ref-wav W --ref-text T]
"""
import argparse, json, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--speaker', default='serena')
    ap.add_argument('--ref-wav', default=None)
    ap.add_argument('--ref-text', default=None)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--tag', default='', help='suffix for output filenames')
    a = ap.parse_args()

    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel

    name = os.path.basename(os.path.abspath(a.model))
    mode = ('voice_design' if 'VoiceDesign' in name
            else 'custom_voice' if 'CustomVoice' in name
            else 'voice_clone')
    os.makedirs(a.out_dir, exist_ok=True)

    t0 = time.time()
    m = Qwen3TTSModel.from_pretrained(a.model, device_map=a.device, dtype=torch.bfloat16)
    print(f"loaded {name} ({mode}) in {time.time()-t0:.1f}s", file=sys.stderr)
    langs = set(m.get_supported_languages() or [])

    LANG = {'en': 'english', 'de': 'german', 'pl': 'polish'}
    rows = [json.loads(l) for l in open(a.jsonl, encoding='utf-8') if l.strip()]
    manifest = []
    for r in rows:
        want = LANG.get(r['lang'], r['lang'])
        # Unsupported languages fall back to 'auto' rather than being skipped — whether
        # the model copes is a measurable question, not one to assume.
        lang = want if want in langs else 'auto'
        note = '' if lang == want else f" (unsupported -> {lang})"
        t = time.time()
        try:
            if mode == 'custom_voice':
                wavs, sr = m.generate_custom_voice(
                    text=r['text'], speaker=a.speaker, language=lang,
                    instruct=r.get('instruct'))
            elif mode == 'voice_design':
                wavs, sr = m.generate_voice_design(
                    text=r['text'], instruct=r.get('instruct', ''), language=lang)
            else:
                wavs, sr = m.generate_voice_clone(
                    text=r['text'], language=lang,
                    ref_audio=a.ref_wav, ref_text=a.ref_text)
        except Exception as e:
            print(f"  FAILED {r['id']}{note}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        w = np.asarray(wavs[0]).astype(np.float32)
        out = os.path.join(a.out_dir, f"{r['id']}{a.tag}.wav")
        sf.write(out, w, sr)
        dur = len(w) / sr
        print(f"  {r['id']:<14}{note} {dur:6.2f}s @ {sr} Hz  in {time.time()-t:5.1f}s", file=sys.stderr)
        manifest.append({**r, 'wav': os.path.basename(out), 'seconds': round(dur, 3),
                         'sample_rate': sr, 'model': name, 'mode': mode,
                         'language_used': lang})
    with open(os.path.join(a.out_dir, f'manifest{a.tag}.jsonl'), 'w', encoding='utf-8') as f:
        for x in manifest:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print(f"wrote {len(manifest)}/{len(rows)} to {a.out_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()
