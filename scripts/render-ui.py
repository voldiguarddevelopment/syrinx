#!/usr/bin/env python3
"""
render-ui.py — live web UI for a Syrinx render run.

Serves a single page that shows a run directory as it fills up: per-GPU progress and
what each worker is currently synthesizing, per-language stats, and once
`verify-renders.py` has written `report.tsv`, the ASR verdict for every clip (detected
language, WER, tag leakage) with inline audio playback.

Stdlib only - no Flask, no build step.

  scripts/render-ui.py --run renders/<run-dir> [--port 8080]
  then open http://localhost:8080
"""
import argparse, csv, html, json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

ARGS = None
# "[12/150] some_id (en/reply) -> 123456 samples"
PROG = re.compile(r'^\[(\d+)/(\d+)\]\s+(\S+)\s+\(([^)]*)\)')
FAIL = re.compile(r'^\[(\d+)/(\d+)\]\s+(\S+)\s+.*(FAILED|out of memory)', re.I)


def corpus():
    p = os.path.join(ARGS.run, 'corpus.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def report():
    p = os.path.join(ARGS.run, 'report.tsv')
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


def worker_state():
    """Parse each gpuN.log for the line the worker is currently on."""
    out = []
    for log in sorted(f for f in os.listdir(ARGS.run) if re.fullmatch(r'gpu\d+\.log', f)):
        gpu = log[3:-4]
        cur, done, total, failed, shard = None, 0, 0, 0, None
        try:
            lines = open(os.path.join(ARGS.run, log), encoding='utf-8', errors='replace').read().splitlines()
        except OSError:
            lines = []
        for ln in lines:
            m = PROG.match(ln)
            if m:
                done, total, cur = int(m.group(1)), int(m.group(2)), m.group(3)
                if FAIL.match(ln):
                    failed += 1
            elif ln.startswith('=== gpu') and 'entries' in ln:
                shard = ln.strip('= ').strip()
        alive = not any(l.strip().endswith('DONE ===') for l in lines[-3:])
        out.append({'gpu': gpu, 'current': cur, 'done': done, 'total': total,
                    'failed': failed, 'shard': shard, 'alive': alive})
    return out


def audio_seconds(path):
    """Frame count from a WAV header without pulling in soundfile."""
    try:
        with open(path, 'rb') as f:
            head = f.read(4096)
        if head[:4] != b'RIFF':
            return None
        i, rate, ch, bits = 12, None, None, None
        while i + 8 <= len(head):
            cid, sz = head[i:i+4], int.from_bytes(head[i+4:i+8], 'little')
            if cid == b'fmt ':
                ch = int.from_bytes(head[i+10:i+12], 'little')
                rate = int.from_bytes(head[i+12:i+16], 'little')
                bits = int.from_bytes(head[i+22:i+24], 'little')
            elif cid == b'data' and rate and ch and bits:
                return sz / (rate * ch * bits // 8)
            i += 8 + sz + (sz & 1)
    except OSError:
        pass
    return None


def find_wav(row):
    """Runs are stored per-language (<run>/<lang>/<id>.wav); older ones are flat."""
    for cand in (os.path.join(ARGS.run, row['lang'], row['id'] + '.wav'),
                 os.path.join(ARGS.run, row['id'] + '.wav')):
        if os.path.exists(cand):
            return cand
    return None


def snapshot():
    rows, rep = corpus(), report()
    items, secs = [], 0.0
    for r in rows:
        wav = find_wav(r)
        done = wav is not None
        dur = audio_seconds(wav) if done else None
        if dur:
            secs += dur
        v = rep.get(r['id'])
        items.append({
            'id': r['id'], 'lang': r['lang'], 'scale': r['scale'],
            'placement': r['placement'], 'tags': r.get('tags', []),
            'text': r['text'], 'words': len(r['text'].split()),
            'done': done, 'seconds': round(dur, 2) if dur else None,
            'wer': float(v['wer']) if v and v.get('wer') else None,
            'det_lang': v.get('det_lang') if v else None,
            'lang_ok': (v.get('lang_ok') == 'True') if v else None,
            'tag_leak': (v.get('tag_leak') == 'True') if v else None,
            'hyp': v.get('hyp') if v else None,
        })
    ndone = sum(i['done'] for i in items)
    workers = worker_state()
    started = min((os.path.getmtime(os.path.join(ARGS.run, f))
                   for f in os.listdir(ARGS.run) if f.startswith('gpu') and f.endswith('.log')),
                  default=time.time())
    elapsed = time.time() - started
    rate = ndone / elapsed if ndone and elapsed > 0 else 0
    eta = (len(items) - ndone) / rate if rate > 0 else None
    return {'run': os.path.basename(os.path.abspath(ARGS.run)), 'items': items,
            'done': ndone, 'total': len(items), 'audio_seconds': round(secs, 1),
            'workers': workers, 'elapsed': round(elapsed), 'eta': round(eta) if eta else None,
            'verified': bool(rep)}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Syrinx render monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fbfaf8;--fg:#1a1a19;--mut:#6b6a66;--line:#e3e0da;--card:#fff;
      --ok:#1f7a4d;--warn:#a86a12;--bad:#b3261e;--accent:#3b5bdb;--chip:#f0eee9}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
      --bg:#16161a;--fg:#e9e8e4;--mut:#9b9a95;--line:#2c2c33;--card:#1e1e24;
      --ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--accent:#8ab4f8;--chip:#26262e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:22px 18px 60px}
h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:12.5px;margin-bottom:18px}
.bar{height:9px;border-radius:5px;background:var(--chip);overflow:hidden;margin:14px 0 6px}
.bar>i{display:block;height:100%;background:var(--accent);transition:width .4s ease}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));margin:16px 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:22px;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.card .v small{font-size:12px;font-weight:400;color:var(--mut)}
.gpu{display:flex;gap:10px;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--line);font-size:13px}
.gpu:last-child{border:0}
.dot{width:8px;height:8px;border-radius:50%;flex:none;align-self:center}
.live{background:var(--ok);animation:p 1.6s infinite}.idle{background:var(--mut)}
@keyframes p{50%{opacity:.35}}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:18px 0 10px}
select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);
             border-radius:7px;padding:6px 9px;font:inherit;font-size:13px}
input[type=search]{min-width:210px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--mut);font-weight:500;font-size:11px;text-transform:uppercase;
   letter-spacing:.05em;padding:8px 8px;border-bottom:1px solid var(--line);
   position:sticky;top:0;background:var(--bg);cursor:pointer;white-space:nowrap}
td{padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tr.pend{opacity:.45}
.chip{display:inline-block;background:var(--chip);border-radius:5px;padding:1px 6px;
      font-size:11px;margin-right:3px;white-space:nowrap}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.txt{max-width:520px}
.hyp{color:var(--mut);font-size:12px;margin-top:3px;display:none}
tr.open .hyp{display:block}
audio{height:30px;width:190px;vertical-align:middle}
.scroll{overflow-x:auto}
.empty{color:var(--mut);padding:34px 0;text-align:center}
</style></head><body><div class="wrap">
<h1>Syrinx render monitor</h1>
<div class="sub" id="sub">connecting…</div>
<div class="bar"><i id="pbar" style="width:0"></i></div>
<div class="sub mono" id="pct"></div>
<div class="grid" id="stats"></div>
<div class="card" id="gpus"></div>
<div class="controls">
  <select id="flang"><option value="">all languages</option></select>
  <select id="fscale"><option value="">all scales</option></select>
  <select id="fplace"><option value="">all placements</option></select>
  <select id="fstate">
    <option value="">all</option><option value="done">rendered</option>
    <option value="pend">pending</option><option value="bad">WER &gt; 0.20</option>
    <option value="lang">language mismatch</option><option value="leak">tag leak</option>
  </select>
  <input type="search" id="q" placeholder="search text or id">
  <span class="sub mono" id="count"></span>
</div>
<div class="scroll"><table id="tbl"><thead><tr>
  <th data-k="id">id</th><th data-k="lang">lang</th><th data-k="scale">scale</th>
  <th data-k="placement">placement</th><th data-k="words">words</th>
  <th data-k="seconds">audio</th><th data-k="wer">WER</th><th>checks</th>
  <th>text</th><th>play</th>
</tr></thead><tbody id="tb"></tbody></table></div>
<div class="empty" id="empty" hidden>nothing matches these filters</div>
</div><script>
let S=null, sortK='id', sortAsc=true;
const $=id=>document.getElementById(id);
const esc=s=>(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const hms=s=>s==null?'—':(s>=3600?Math.floor(s/3600)+'h '+Math.floor(s%3600/60)+'m':
                          s>=60?Math.floor(s/60)+'m '+Math.round(s%60)+'s':Math.round(s)+'s');
function fill(sel,vals){const e=$(sel),cur=e.value;
  [...e.options].slice(1).forEach(o=>o.remove());
  vals.forEach(v=>{const o=document.createElement('option');o.value=o.textContent=v;e.append(o)});
  e.value=cur;}
function stats(){
  const it=S.items, d=it.filter(i=>i.done), v=it.filter(i=>i.wer!=null);
  const med=a=>{if(!a.length)return null;const s=[...a].sort((x,y)=>x-y);return s[s.length>>1]};
  const cards=[
    ['rendered', `${S.done}<small>/${S.total}</small>`],
    ['audio produced', hms(S.audio_seconds)],
    ['elapsed', hms(S.elapsed)],
    ['eta', hms(S.eta)],
  ];
  if(v.length){
    cards.push(['median WER', (med(v.map(i=>i.wer))??0).toFixed(3)]);
    cards.push(['language ok', `${v.filter(i=>i.lang_ok).length}<small>/${v.length}</small>`]);
    cards.push(['tag leaks', `${v.filter(i=>i.tag_leak).length}`]);
  }
  $('stats').innerHTML=cards.map(([k,val])=>
    `<div class="card"><div class="k">${k}</div><div class="v">${val}</div></div>`).join('');
  $('gpus').innerHTML=S.workers.map(w=>
    `<div class="gpu"><span class="dot ${w.alive?'live':'idle'}"></span>
     <b>cuda:${w.gpu}</b><span class="mono">${w.done}/${w.total}</span>
     <span class="mono" style="color:var(--mut)">${esc(w.current||(w.alive?'loading model…':'finished'))}</span>
     ${w.failed?`<span class="bad mono">${w.failed} failed</span>`:''}</div>`).join('')
    ||'<div class="sub">no worker logs yet</div>';
}
function rows(){
  const fl=$('flang').value,fs=$('fscale').value,fp=$('fplace').value,
        st=$('fstate').value,q=$('q').value.toLowerCase();
  let it=S.items.filter(i=>
    (!fl||i.lang===fl)&&(!fs||i.scale===fs)&&(!fp||i.placement===fp)&&
    (!q||i.id.toLowerCase().includes(q)||i.text.toLowerCase().includes(q))&&
    (st===''||(st==='done'&&i.done)||(st==='pend'&&!i.done)||
     (st==='bad'&&i.wer!=null&&i.wer>0.2)||(st==='lang'&&i.lang_ok===false)||
     (st==='leak'&&i.tag_leak===true)));
  it.sort((a,b)=>{const x=a[sortK],y=b[sortK];
    if(x==null&&y==null)return 0; if(x==null)return 1; if(y==null)return -1;
    return (x>y?1:x<y?-1:0)*(sortAsc?1:-1);});
  $('count').textContent=`${it.length} shown`;
  $('empty').hidden=it.length>0;
  $('tb').innerHTML=it.map(i=>{
    const wcls=i.wer==null?'':i.wer<=0.1?'ok':i.wer<=0.25?'warn':'bad';
    const checks=[
      i.lang_ok===false?`<span class="chip bad">heard ${esc(i.det_lang)}</span>`:'',
      i.tag_leak?'<span class="chip bad">tag leak</span>':'',
      (i.lang_ok===true&&!i.tag_leak)?'<span class="chip ok">clean</span>':''
    ].join('');
    return `<tr class="${i.done?'':'pend'}" data-id="${esc(i.id)}">
      <td class="mono">${esc(i.id)}</td><td>${i.lang}</td><td>${i.scale}</td>
      <td><span class="chip">${i.placement}</span></td><td>${i.words}</td>
      <td class="mono">${i.seconds!=null?i.seconds.toFixed(1)+'s':'—'}</td>
      <td class="mono ${wcls}">${i.wer!=null?i.wer.toFixed(3):'—'}</td>
      <td>${checks}</td>
      <td class="txt">${esc(i.text)}
        ${i.hyp?`<div class="hyp"><b>heard:</b> ${esc(i.hyp)}</div>`:''}</td>
      <td>${i.done?`<audio preload="none" controls src="audio/${encodeURIComponent(i.id)}.wav"></audio>`:''}</td>
    </tr>`}).join('');
}
$('tb').addEventListener('click',e=>{const tr=e.target.closest('tr');
  if(tr&&e.target.tagName!=='AUDIO')tr.classList.toggle('open')});
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; sortAsc = (k===sortK)?!sortAsc:true; sortK=k; rows();});
['flang','fscale','fplace','fstate','q'].forEach(id=>$(id).oninput=rows);
async function tick(){
  try{
    S=await (await fetch('api/status',{cache:'no-store'})).json();
    $('sub').textContent=`${S.run} · ${S.verified?'verified':'not yet verified'}`;
    const p=S.total?S.done/S.total*100:0;
    $('pbar').style.width=p+'%';
    $('pct').textContent=`${p.toFixed(1)}% · ${S.done} of ${S.total}`;
    fill('flang',[...new Set(S.items.map(i=>i.lang))].sort());
    fill('fscale',[...new Set(S.items.map(i=>i.scale))].sort());
    fill('fplace',[...new Set(S.items.map(i=>i.placement))].sort());
    stats(); rows();
  }catch(e){$('sub').textContent='lost connection to render-ui.py';}
}
tick(); setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ('/', '/index.html'):
            return self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
        if path == '/api/status':
            return self._send(200, json.dumps(snapshot()).encode(), 'application/json')
        if path.startswith('/audio/'):
            rel = path[len('/audio/'):]
            name = os.path.basename(rel)
            if not name.endswith('.wav'):
                return self._send(404, b'no', 'text/plain')
            root = os.path.realpath(ARGS.run)
            # <run>/<lang>/<id>.wav, or the flat <run>/<id>.wav for older runs.
            # Only the basename is ever used to build the path, and the result must
            # still resolve inside the run dir - so a crafted rel cannot escape.
            cands = [os.path.join(root, d, name)
                     for d in sorted(os.listdir(root))
                     if os.path.isdir(os.path.join(root, d))]
            cands.append(os.path.join(root, name))
            full = next((c for c in cands
                         if os.path.exists(c)
                         and os.path.realpath(c).startswith(root + os.sep)), None)
            if full is None:
                return self._send(404, b'not found', 'text/plain')
            with open(full, 'rb') as f:
                return self._send(200, f.read(), 'audio/wav')
        self._send(404, b'not found', 'text/plain')


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='render run directory')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', default='127.0.0.1')
    ARGS = ap.parse_args()
    if not os.path.isdir(ARGS.run):
        raise SystemExit(f"no such run directory: {ARGS.run}")
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), H)
    print(f"render UI  ->  http://{ARGS.host}:{ARGS.port}   (run: {ARGS.run})")
    srv.serve_forever()


if __name__ == '__main__':
    main()
