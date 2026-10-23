"""Spike 6 -- a listening sheet: does the verbatim model's "um" match a person's ear?

There is no hand-marked answer key for a real Paper Edit project, so a person makes a
small one:
  part 1 - every filler the model found, as a 2-second-padded clip: real or not?
           (that is PRECISION on the speaker's own voice)
  part 2 - a 2-minute stretch with an "I heard one!" button: every filler the ear
           catches is timed, then compared with what the model found there (RECALL).
Results are shown as one line of text to paste back.

    .venv\\Scripts\\python.exe spikes\\listen_sheet.py <project id> <fillers.json> [excerpt start s]

Writes spikes/out/listen/<project>/index.html + clips. spikes/out is gitignored: this
is someone's own recording and never leaves the machine.
"""
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAD = 1.0
EXCERPT_S = 120


def clip(src, dst, start, end):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.2f" % max(0, start), "-to", "%.2f" % end,
                    "-i", str(src), "-vn", "-ac", "1", "-b:a", "96k", str(dst)], check=True)


def main(argv):
    pid, fjson = argv[0], argv[1]
    ex0 = float(argv[2]) if len(argv) > 2 else 30.0
    src = next((ROOT / "projects" / pid).glob("source.*"))
    found = json.load(open(fjson))["fillers"]
    out = ROOT / "spikes" / "out" / "listen" / pid
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (s, e, w) in enumerate(found):
        name = "f%02d.mp3" % i
        clip(src, out / name, s - PAD, e + PAD)
        items.append({"file": name, "start": s, "end": e, "word": w})
    clip(src, out / "excerpt.mp3", ex0, ex0 + EXCERPT_S)
    in_excerpt = [s - ex0 for s, e, w in found if ex0 <= s < ex0 + EXCERPT_S]
    rows = "".join(
        "<div class=row data-i=%d><audio controls preload=none src='%s'></audio>"
        "<span>%s at %d:%04.1f</span><button class=y>Real um</button><button class=n>Not one</button></div>"
        % (i, it["file"], html.escape(it["word"]), int(it["start"] // 60), it["start"] % 60)
        for i, it in enumerate(items))
    page = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Um check</title><style>
body{font:17px/1.45 system-ui,sans-serif;max-width:720px;margin:auto;padding:16px;background:#faf9f6;color:#1d1d1b}
h1{font-size:24px;margin:.2em 0}h2{font-size:18px;margin:1.4em 0 .4em}.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
padding:10px 0;border-top:1px solid #e2e0da}audio{height:36px;max-width:100%%}button{font:inherit;padding:8px 14px;border-radius:10px;
border:1px solid #c9c6bd;background:#fff;cursor:pointer}.picked{background:#1f7a45;color:#fff;border-color:#1f7a45}
.n.picked{background:#b3261e;border-color:#b3261e}#tap{font-size:22px;padding:18px 28px;background:#ffd23f;border:0}
#taps{color:#5c5c55}textarea{width:100%%;font:14px monospace;height:70px}.note{color:#5c5c55;font-size:15px}</style>
<h1>Um check</h1><p class=note>Helps tune Paper Edit's "find fillers". About 5 minutes. Nothing here leaves this computer.</p>
<h2>1. Is each of these a real filler? (%(n)d clips, the "um" is in the middle)</h2>%(rows)s
<h2>2. Listen to 2 minutes and tap every filler you hear</h2>
<p class=note>Um, uh, er, hmm... Tap as you hear it - being a second late is fine.</p>
<audio id=ex controls src="excerpt.mp3" style="width:100%%"></audio>
<p><button id=tap>I heard one!</button> <span id=taps>0 tapped</span> <button id=undo>Undo last</button></p>
<h2>3. Send this line back to Claude</h2><textarea id=res readonly></textarea><button id=copy>Copy</button>
<script>
const items=%(items)s, modelInExcerpt=%(inex)s, picks={}, taps=[];const ex=document.getElementById('ex');
document.querySelectorAll('.row').forEach(r=>r.addEventListener('click',e=>{if(e.target.tagName!=='BUTTON')return;
 const i=+r.dataset.i;picks[i]=e.target.classList.contains('y');r.querySelectorAll('button').forEach(b=>b.classList.remove('picked'));
 e.target.classList.add('picked');upd()}));
document.getElementById('tap').onclick=()=>{taps.push(+ex.currentTime.toFixed(1));upd()};
document.getElementById('undo').onclick=()=>{taps.pop();upd()};
document.getElementById('copy').onclick=()=>{const t=document.getElementById('res');t.select();try{navigator.clipboard.writeText(t.value)}catch(e){}};
function upd(){document.getElementById('taps').textContent=taps.length+' tapped';
 const real=Object.values(picks).filter(Boolean).length, judged=Object.keys(picks).length;
 document.getElementById('res').value='UMCHECK %(pid)s real '+real+'/'+judged+' of '+items.length+' | marks '+
  items.map((it,i)=>i in picks?(picks[i]?'Y':'N'):'?').join('')+' | taps '+taps.join(',')+' | model '+modelInExcerpt.map(x=>x.toFixed(1)).join(',')}
upd();
</script>""" % {"n": len(items), "rows": rows, "items": json.dumps(items), "inex": json.dumps(in_excerpt), "pid": pid}
    (out / "index.html").write_text(page, encoding="utf-8")
    print(out / "index.html", "|", len(items), "clips | excerpt %d-%d s has %d model fillers" % (ex0, ex0 + EXCERPT_S, len(in_excerpt)))


if __name__ == "__main__":
    main(sys.argv[1:])
