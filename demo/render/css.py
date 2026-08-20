"""
The stylesheet: one minimalist sheet shared by both report pages.

PURPOSE
    A single module-level string, `CSS`, inlined into the `<head>` of each
    generated page -- so a report is one self-contained HTML file with no external
    assets to lose.

FIT
    A leaf: imported by `demo/render/__init__.py` and written out by
    `scripts/pipeline_report.py` and `scripts/llm_report.py`. Depends on nothing.

HOW
    Colours are declared once as CSS custom properties in the `:root` block at the
    top, so the palette can be changed in one place. The class names below
    correspond to the ones the renderers emit -- `.stage`, `.prov`, `.act`, `.ck`,
    `.m-tile` and so on.
"""

from __future__ import annotations


CSS = """
:root{--ink:#1f2328;--muted:#5b636c;--faint:#9aa3ad;--line:#e7e9ec;--panel:#f6f7f9;
  --accent:#3b6ea5;--accentbg:#eef3f9;}
*{box-sizing:border-box;}
body{margin:0;background:#fbfbfc;color:var(--ink);-webkit-font-smoothing:antialiased;
  font-family:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,'Times New Roman',serif;
  font-size:16px;line-height:1.6;}
.wrap{max-width:1120px;margin:0 auto;padding:40px 28px 100px;}
h1{font-size:24px;font-weight:600;letter-spacing:-0.3px;margin:0 0 4px;}
.lede{color:var(--muted);font-size:14.5px;margin:0 0 8px;max-width:78ch;}
h3{font-size:15.5px;font-weight:600;margin:0;letter-spacing:-0.1px;}
.muted{color:var(--muted);} .faint{color:var(--faint);}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
  background:var(--panel);padding:1px 5px;border-radius:4px;}

/* tabs */
.tabs{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line);margin:24px 0 26px;}
.tab{appearance:none;background:none;border:none;cursor:pointer;font:inherit;font-size:13.5px;
  color:var(--muted);padding:9px 14px 11px;border-bottom:2px solid transparent;margin-bottom:-1px;text-align:left;}
.tab .t-sub{display:block;font-size:11px;color:var(--faint);margin-top:1px;}
.tab:hover{color:var(--ink);}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.tab.active .t-sub{color:var(--muted);}
.panel{display:none;} .panel.active{display:block;}

/* metrics */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px;}
.m-tile{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.m-val{font-size:23px;font-weight:600;letter-spacing:-0.5px;line-height:1.1;}
.m-val.ok{color:#2f7a3d;} .m-val.bad{color:#b3261e;} .m-val.warnv{color:#8a5a12;}
.m-lab{font-size:12px;color:var(--muted);margin-top:4px;}
.m-sub{font-size:10.5px;color:var(--faint);margin-top:2px;}
@media(max-width:900px){.metrics{grid-template-columns:repeat(2,1fr);}}

/* stepper */
.stepper{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 20px;}
.stepper .st{background:#fff;border:1px solid var(--line);border-radius:20px;padding:4px 12px;
  font-size:12px;color:var(--muted);}
.stepper .st b{color:var(--accent);font-weight:600;margin-right:5px;font-size:11px;}
.stepper .sep{color:var(--faint);font-size:12px;}

/* stages */
.stages{display:flex;flex-direction:column;gap:18px;}
.stage{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 22px;}
.s-head{display:flex;align-items:center;gap:11px;}
.s-num{flex:0 0 auto;width:26px;height:26px;border-radius:50%;background:var(--accentbg);color:var(--accent);
  font-size:11.5px;font-weight:600;display:flex;align-items:center;justify-content:center;}
.s-note{color:var(--muted);font-size:13px;margin:6px 0 0 37px;max-width:86ch;}

/* 01 detect */
.tally{font-size:11.5px;color:var(--faint);margin-top:12px;}
.excerpt{background:var(--panel);border-radius:10px;padding:16px 18px;line-height:1.95;font-size:14.5px;margin-top:8px;}
mark{padding:1px 5px;border-radius:4px;font-weight:500;}
.legend{display:flex;gap:16px;margin-top:12px;font-size:12px;color:var(--muted);flex-wrap:wrap;}
.lg::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:1px;}
.lg-p::before{background:#5B7FA6;}.lg-l::before{background:#6f9a4a;}.lg-d::before{background:#c99a4a;}
.lg-a::before{background:#7d72c4;}.lg-i::before{background:#c0574f;}

/* lists */
.deltas,.rels{list-style:none;padding:0;margin:14px 0 0;}
.deltas li,.rels li{padding:8px 0;border-bottom:1px solid var(--line);font-size:14px;}
.deltas li:last-child,.rels li:last-child{border-bottom:none;}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;margin-right:6px;}
.tag.ok{background:#e7f2e9;color:#2f6b3b;} .tag.warn{background:#fbeceb;color:#93362f;}
.src{display:inline-block;font-size:9.5px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;
  padding:1px 6px;border-radius:4px;margin-right:7px;}
.src.rule{background:#eceef1;color:#5b636c;} .src.llm{background:var(--accentbg);color:var(--accent);}
.rel{display:inline-block;font-size:11.5px;color:var(--accent);background:var(--accentbg);padding:1px 8px;border-radius:20px;}
.ev{display:block;color:var(--faint);font-size:12px;margin-top:2px;font-style:italic;}

/* cards */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin-top:14px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px 14px;}
.card.flag{border-left:3px solid #c0574f;}
.card .nm{font-size:14.5px;font-weight:600;}
.card .nm .cat{font-weight:400;font-size:11px;color:var(--muted);margin-left:4px;}
.card .forms{font-size:11.5px;color:var(--muted);margin:1px 0 8px;}
.card .chips{display:flex;flex-wrap:wrap;}
.chip{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;margin:0 4px 4px 0;background:#eceef1;color:#4b5158;}
.chip.g{background:#E9F0FA;color:#234E86;}
.chip.ok{background:#e7f2e9;color:#2f6b3b;}
.chip.warn{background:#fbeceb;color:#8a2f28;}
.chip.eth{background:#F3E8F6;color:#6b2f7c;}
.chip.occ{background:#E4F1F1;color:#245b5b;}
.chip.id{background:#FBEAEA;color:#7C2222;}
.chip.age{background:#EFEDFA;color:#413593;}
.chip.date{background:#F8EFDD;color:#6A4310;}
.chip.rel-chip{background:#E9F0FA;color:#234E86;}
.chip.faint{background:#eef0f2;color:#7c828b;}
.card .sug{font-size:11.5px;color:#6A4310;margin-top:5px;}
.card .other{font-size:11px;color:#8a8f98;margin-top:5px;line-height:1.5;}

/* review flags: a list, not a red run-on sentence */
.note{margin-top:9px;background:#fdf5f4;border:1px solid #f0dbd8;border-left:3px solid #c0574f;
  border-radius:8px;padding:8px 11px 9px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;}
.note-h{font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:#a2453c;}
.note ul{margin:5px 0 0;padding-left:15px;}
.note li{font-size:12px;line-height:1.55;color:#4b3330;margin:3px 0;overflow-wrap:anywhere;}
.note li b{color:#8a2f28;font-weight:600;}

/* decision record, stacked -- for narrow containers (person cards) */
.pv-list{margin-top:8px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;}
.pv{border-top:1px solid var(--line);padding:7px 0 8px;overflow-wrap:anywhere;}
.pv:first-child{border-top:none;}
.pv.blk{background:#fdf4f3;border-radius:6px;padding-left:7px;padding-right:7px;}
.pv-h{display:flex;flex-wrap:wrap;align-items:center;gap:7px;}
.pv-f{font-size:12px;font-weight:700;color:var(--accent);}
.pv-v{font-size:12.5px;font-weight:600;margin-top:3px;}
.pv-v i{font-style:normal;font-weight:400;color:var(--muted);font-size:11.5px;}
.pv-c{margin-top:3px;line-height:1.6;}
.pv .why{margin-top:3px;}

/* provenance tables */
table.prov{width:100%;border-collapse:collapse;margin:10px 0 4px;font-size:12.5px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;}
table.prov th{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;
  color:var(--faint);font-weight:600;border-bottom:1px solid var(--line);padding:4px 8px 5px;}
table.prov td{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top;}
table.prov tr:last-child td{border-bottom:none;}
table.prov tr.blk{background:#fdf4f3;}
table.prov td.f{color:var(--accent);font-weight:600;white-space:nowrap;}
table.prov td.v{font-weight:600;}
table.prov td.v.big{font-size:14px;}
table.prov td.src2{color:var(--muted);white-space:nowrap;}
table.prov td.src2 i{font-style:normal;color:var(--faint);}
table.prov td.parts{color:var(--muted);}
.part{display:inline-block;background:var(--panel);border-radius:4px;padding:0 5px;margin:0 3px 3px 0;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;}
.iv-t td.f{width:150px;}
.act{display:inline-block;font-size:10px;font-weight:600;letter-spacing:.3px;padding:1px 7px;
  border-radius:20px;white-space:nowrap;}
.act.a-ok{background:#e7f2e9;color:#2f6b3b;}
.act.a-fill{background:#F8EFDD;color:#6A4310;}
.act.a-keep{background:#eceef1;color:#5b636c;}
.act.a-conf{background:#FBEAEA;color:#7C2222;}
.act.a-rej{background:#F3E8F6;color:#6b2f7c;}
.act.a-blind{background:#eef0f2;color:#7c828b;}
.blkmark{color:#b3261e;font-size:10px;letter-spacing:.4px;}
.ck{display:inline-block;font-size:11px;margin-right:8px;}
.ck.ok{color:#2f6b3b;} .ck.bad{color:#b3261e;} .ck.na{color:var(--faint);}
.ck.none{color:#8a5a12;font-style:italic;}
.why{color:var(--faint);font-size:11px;margin-top:2px;line-height:1.45;}
.why2{font-size:12px;margin:2px 0 6px;}
.none{color:var(--faint);font-style:normal;}
.ev2{color:var(--faint);font-size:11px;font-style:italic;}
.arrow2{color:var(--muted);font-style:normal;font-weight:400;}
details.prov-d,details.ldg{margin-top:8px;}
details.prov-d summary,details.ldg summary{cursor:pointer;font-size:11.5px;color:var(--accent);
  list-style:none;padding:3px 0;}
details.ldg summary{font-size:13px;padding:7px 0;border-top:1px solid var(--line);}
details.prov-d summary::-webkit-details-marker,details.ldg summary::-webkit-details-marker{display:none;}
details.prov-d summary::before,details.ldg summary::before{content:"\\25B8 ";color:var(--faint);}
details.prov-d[open] summary::before,details.ldg[open] summary::before{content:"\\25BE ";}

/* graph + places + dates */
.graph{padding-top:8px;}
.glegend{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin-top:14px;font-size:11.5px;color:var(--muted);}
.glegend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;}
.glegend .ring{color:#8a2f28;}
.sub-h{font-size:11.5px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;margin:20px 0 9px;}
.tally2{font-size:12px;margin:4px 0 0;}
.trow{display:flex;align-items:center;gap:9px;margin-bottom:5px;font-size:13px;}
.tw{color:var(--faint);}
.pills{display:flex;flex-wrap:wrap;gap:7px;}
.pill{font-size:12px;padding:3px 10px;border-radius:20px;display:inline-block;}
.pill.loc{background:#EAF2E0;color:#315915;}
.pill.kin{background:#E9F0FA;color:#234E86;}
.pill.repl{background:#FBECEA;color:#7C2222;}
.pill.keepp{background:#EAF3EC;color:#2F6B3B;}
.ptype{font-size:11.5px;color:var(--muted);}
.dmark{font-style:normal;font-size:10.5px;color:var(--muted);background:var(--panel);
  border-radius:4px;padding:0 5px;margin-left:3px;}
.own{font-style:normal;font-size:10.5px;padding:1px 7px;border-radius:20px;white-space:nowrap;}
.own-iv{background:#2C4A6E;color:#fff;}
.own-other{background:#fff;color:var(--muted);border:1px solid var(--line);}
.own-block{background:#7C2222;color:#fff;}

/* boxes */
.blkbox{background:#fdf4f3;border:1px solid #f0d5d2;border-left:3px solid #b3261e;
  border-radius:9px;padding:12px 15px;margin:14px 0 4px;font-size:13px;}
.blkbox ul{margin:6px 0 0;padding-left:18px;} .blkbox li{margin:3px 0;}
.okbox{background:#f2f8f3;border:1px solid #d5e7d9;border-left:3px solid #2f7a3d;
  border-radius:9px;padding:12px 15px;margin:14px 0 4px;font-size:13px;}
.kvs{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px 14px;margin:14px 0 0;}
.kv{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;
  border-bottom:1px dotted var(--line);padding:3px 0;}
.kv span{color:var(--muted);}
.kv b{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;font-weight:500;}
.foot{font-size:12.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:24px;}
"""
