"""Build an interactive click-to-answer version of the MUR audit sheet.

Reconstructs the exact 100 frozen items from key.csv (item order + episode
pointers; the `auto` column is NOT surfaced, so annotators stay blind),
loads each transcript from the run traces, regenerates gold values, and
emits runs/mur_audit/audit_ui.html: yes/no/unclear buttons per item,
progress bar, browser-local autosave, and a one-click "Download my labels"
button. The judgement is entirely the human's; this only records it.

Usage:  PYTHONPATH=. python scripts/mur_audit_ui.py
Then open runs/mur_audit/audit_ui.html, answer all 100, click Download,
and send the file back (it becomes one annotator column in labels.csv).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.mur_audit import CELLS, OUT  # reuse the frozen mapping
from merit.domains import DOMAINS


def build() -> None:
    golds_by_cell = {
        (dom, tier): {ep.task.task_id: ep.task.golds()
                      for arc in DOMAINS[dom].generate_suite(difficulty=tier)
                      for ep in arc.episodes}
        for (dom, tier) in CELLS
    }
    trace_cache: dict[str, dict] = {}

    def transcript(run: str, episode_id: str) -> str:
        if run not in trace_cache:
            trace_cache[run] = {
                json.loads(l)["episode_id"]: json.loads(l)["transcript"]
                for l in (Path(run) / "traces" / "episodes.jsonl").open()}
        return trace_cache[run][episode_id]

    items = []
    for r in csv.DictReader((OUT / "key.csv").open()):
        items.append({
            "id": r["item"],
            "meta": f'{r["domain"]} / {r["difficulty"]} / {r["condition"]}',
            "golds": golds_by_cell[r["domain"], r["difficulty"]][r["task_id"]],
            "transcript": transcript(r["run"], r["episode_id"]),
        })

    data = json.dumps(items).replace("</", "<\\/")
    (OUT / "audit_ui.html").write_text(HTML.replace("__DATA__", data))
    print(f"wrote {OUT}/audit_ui.html ({len(items)} items)")


HTML = r"""<!doctype html><meta charset="utf-8">
<title>MERIT MUR audit (interactive)</title>
<style>
 body{font:15px/1.5 system-ui;max-width:56rem;margin:0 auto 6rem;padding:0 1rem}
 pre{white-space:pre-wrap;background:#f6f6f2;padding:.8rem;border-radius:6px;
     font-size:13px;max-height:22rem;overflow:auto}
 section{border-top:2px solid #ddd;margin-top:1.5rem;padding-top:1rem}
 .meta{color:#666} code{background:#eee;padding:0 .2rem;border-radius:3px}
 .golds{background:#fdf6e3;padding:.5rem .8rem;border-radius:6px}
 .btns button{font:inherit;padding:.35rem 1rem;margin-right:.4rem;
     border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
 .btns button.on[data-v=yes]{background:#1baf7a;color:#fff;border-color:#1baf7a}
 .btns button.on[data-v=no]{background:#d64545;color:#fff;border-color:#d64545}
 .btns button.on[data-v=unclear]{background:#888;color:#fff;border-color:#888}
 #bar{position:fixed;top:0;left:0;right:0;background:#222;color:#fff;
     padding:.6rem 1rem;display:flex;gap:1rem;align-items:center;z-index:9}
 #bar input{font:inherit;padding:.2rem .4rem}
 #bar button{font:inherit;padding:.3rem .9rem;cursor:pointer}
 main{margin-top:3.5rem}
</style>
<div id="bar">
 <strong>MUR audit</strong>
 <span>Your name: <input id="who" placeholder="e.g. shweta" size="10"></span>
 <span id="prog">0 / 100 answered</span>
 <button id="dl">Download my labels</button>
</div>
<main>
 <p>For each item: does the agent's <em>executed tool calls</em> (the
 <code>[tool ...]</code> lines) use the <strong>gold value(s)</strong> it
 needed from memory, verbatim or reformatted? The value merely appearing in
 the user's words or memory does <em>not</em> count. Click
 <strong>yes / no / unclear</strong>. Progress saves automatically in this
 browser. When all 100 are answered, enter your name and click Download.</p>
 <div id="items"></div>
</main>
<script>
const ITEMS = __DATA__;
// storage is namespaced per annotator name so a second annotator on the
// same browser starts blank instead of inheriting the first one's answers
// (independence is the whole point of a two-annotator audit)
let WHO = "";
let saved = {};
const keyFor = w => "merit_mur_labels::" + (w || "").trim().toLowerCase();
function loadFor(w){
  WHO = w;
  saved = JSON.parse(localStorage.getItem(keyFor(w)) || "{}");
}
const root = document.getElementById("items");
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
ITEMS.forEach(it=>{
  const sec=document.createElement("section");
  sec.innerHTML=`<h3>${it.id}</h3>
    <p class="meta">${it.meta}</p>
    <p class="golds"><strong>Gold value(s) needed from memory:</strong> `
    + it.golds.map(g=>`<code>${esc(g)}</code>`).join(" &nbsp; ") + `</p>
    <pre>${esc(it.transcript)}</pre>
    <p class="btns" data-id="${it.id}">
      <button data-v="yes">yes</button>
      <button data-v="no">no</button>
      <button data-v="unclear">unclear</button></p>`;
  root.appendChild(sec);
});
function paint(){
  document.querySelectorAll(".btns").forEach(p=>{
    const v=saved[p.dataset.id];
    p.querySelectorAll("button").forEach(b=>
      b.classList.toggle("on", b.dataset.v===v));
  });
  document.getElementById("prog").textContent =
    Object.keys(saved).length + " / " + ITEMS.length + " answered";
}
const whoBox=document.getElementById("who");
whoBox.addEventListener("change",()=>{ loadFor(whoBox.value); paint(); });
root.addEventListener("click",e=>{
  const b=e.target.closest("button[data-v]"); if(!b) return;
  if(!whoBox.value.trim()){ alert("Enter your name at the top first — answers are saved per annotator."); whoBox.focus(); return; }
  if(WHO!==whoBox.value) loadFor(whoBox.value);
  saved[b.parentElement.dataset.id]=b.dataset.v;
  localStorage.setItem(keyFor(WHO), JSON.stringify(saved));
  paint();
});
document.getElementById("dl").onclick=()=>{
  const who=(whoBox.value||"annotator").trim();
  const miss=ITEMS.filter(it=>!saved[it.id]).map(it=>it.id);
  if(miss.length && !confirm(miss.length+" items unanswered ("+miss.slice(0,5).join(", ")+"...). Download anyway?")) return;
  let csv="item,label\n"+ITEMS.map(it=>it.id+","+(saved[it.id]||"")).join("\n")+"\n";
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="labels-"+who+".csv"; a.click();
};
paint();
</script>"""


if __name__ == "__main__":
    build()
