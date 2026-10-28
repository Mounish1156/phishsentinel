import html, json, time
import pandas as pd, streamlit as st
import phishsentinel as ps

st.set_page_config(page_title="PhishSentinel | Risk Dashboard", page_icon="🛡️", layout="wide")
st.markdown("""<style>
:root{--bg:#0a0f1c;--pn:#0f1729;--ln:#1f2b45;--tx:#e6ecff;--mu:#8a97b5;--tl:#2de2c4;--pu:#8b5cf6;--bl:#4f8cff;--gr:#22c55e;--rd:#ef4444;--am:#f59e0b}
.stApp{background:radial-gradient(1200px 500px at 70% -10%,#12203d 0,#0a0f1c 60%)}
header[data-testid="stHeader"]{background:transparent}.block-container{padding:1rem 1.6rem 2rem;max-width:1500px}
section[data-testid="stSidebar"]{background:#0c1424;border-right:1px solid var(--ln)}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--pn);border:1px solid var(--ln)!important;border-radius:12px}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:2px}
[data-testid="stSidebar"] div[role="radiogroup"] label{padding:9px 12px;border-radius:8px;width:100%;color:var(--mu)}
[data-testid="stSidebar"] div[role="radiogroup"] label>div:first-child{display:none}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:#1a2742;color:var(--tx)}
.brand{font-weight:700;font-size:16px;margin:2px 4px 14px}.brand span{color:var(--mu);font-weight:400}
.kpi{display:flex;gap:10px;align-items:center;padding:12px 14px;border:1px solid var(--ln);border-radius:10px;background:var(--pn);font-size:13px;min-height:58px}
.kpi.g{border-color:#1c7a4c;background:#0f2a24}.kpi.r{border-color:#7a1f2a;background:#2a1019}
.kpi b{font-size:15px}.kpi .ic{font-size:18px}.kpi .g{color:var(--gr)}
.sec{font-weight:600;font-size:14px;margin:16px 0 8px}.pt{font-weight:600;font-size:13px;margin-bottom:8px}
.lg{display:flex;gap:14px;font-size:11px;color:var(--mu);justify-content:flex-end}.lg i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.tb{width:100%;border-collapse:collapse;font-size:12px}.tb th{color:var(--mu);font-weight:500;text-align:left;padding:5px 6px;border-bottom:1px solid var(--ln)}
.tb td{padding:6px;border-bottom:1px solid #16213a}
.ok{color:var(--gr)}.wn{color:var(--am)}.er{color:var(--rd)}.inf{color:var(--bl)}
.bars{display:flex;align-items:flex-end;gap:12px;height:120px}.bars div{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;height:100%;font-size:11px;color:var(--mu);text-align:center}
.bars b{display:block;width:100%;border-radius:4px 4px 0 0;background:linear-gradient(#8b5cf6,#2de2c4)}
.al{display:flex;gap:10px;font-size:12px;padding:6px 0;border-bottom:1px solid #16213a;align-items:center}.al .d{width:9px;height:9px;border-radius:50%}
.gs{display:flex;gap:14px;align-items:center;font-size:12px;color:var(--mu)}.gs b{color:var(--tx)}
.stButton>button{background:var(--tl);color:#06241f;border:0;border-radius:8px;font-weight:700}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_models(): return ps.load_models()
@st.cache_data
def get_results():
    p = ps.ROOT / "outputs/results.json"
    if not p.exists(): ps.run_pipeline(log=False)
    return json.loads(p.read_text())
models, R = get_models(), get_results()
ss = st.session_state
for k, v in dict(events=[], alerts=[], last=None, msg="", model="TF-IDF + LogReg", n=0, high=0).items(): ss.setdefault(k, v)
clock = lambda: time.strftime("%H:%M:%S")

def seed_events():
    t = time.strftime("%H:%M:%S", time.localtime((ps.ROOT / "outputs/results.json").stat().st_mtime)); d, ev = R["dedup"], []
    ev.append((t, "PIPE", "INF", f"Dedup removed {d['removed']} near-duplicates ({d['before']} to {d['after']})"))
    ev.append((t, "EVAL", "INF", f"Evaluated {len(R['split']['holdout_campaigns'])} unseen campaigns, {R['split']['holdout']} messages"))
    for k, v in R["models"].items():
        if v["macro_f1"] < .5: ev.append((t, "MODEL", "ERR", f"{k} macro-F1 {v['macro_f1']:.2f} on unseen campaigns"))
        if v["abstain_rate"] > .3: ev.append((t, "MODEL", "WRN", f"{k} abstains on {v['abstain_rate']:.0%} of holdout"))
    return ev
def all_events(): return ss.events + [{"ts": a, "id": b, "ty": c, "msg": m} for a, b, c, m in seed_events()]
def tyc(t): return {"INF": "inf", "WRN": "wn", "ERR": "er", "CRIT": "er"}[t]
def table(head, rows): return '<table class="tb"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"
def smooth(p):
    d = f"M{p[0][0]:.1f},{p[0][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(p, p[1:]): m = (x0 + x1) / 2; d += f" C{m:.1f},{y0:.1f} {m:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
    return d
def line_chart(labels, series, ymax, w=640, h=210):
    L, T, Rr, B = 40, 8, 10, 24; pw, ph = w - L - Rr, h - T - B; g = ""
    for i in range(5): y = T + ph * i / 4; g += f'<line x1="{L}" x2="{w-Rr}" y1="{y}" y2="{y}" stroke="#1f2b45"/><text x="{L-6}" y="{y+3}" text-anchor="end" fill="#8a97b5" font-size="10">{ymax*(1-i/4):.0f}%</text>'
    xs = [L + pw * i / (len(labels) - 1) for i in range(len(labels))]
    for x, lb in zip(xs, labels): g += f'<text x="{x}" y="{h-8}" text-anchor="middle" fill="#8a97b5" font-size="10">{lb}</text>'
    for name, (vals, c) in series.items(): g += f'<path d="{smooth([(x, T+ph*(1-v/ymax)) for x, v in zip(xs, vals)])}" fill="none" stroke="{c}" stroke-width="2.6" stroke-linecap="round"/>'
    return f'<svg viewBox="0 0 {w} {h}" width="100%">{g}</svg>'
def gauge(f, c):
    f = max(0, min(1, f)); a = "M10,60 A50,50 0 0 1 110,60"
    return f'<svg viewBox="0 0 120 70" width="130"><path d="{a}" fill="none" stroke="#1f2b45" stroke-width="10" stroke-linecap="round"/><path d="{a}" fill="none" stroke="{c}" stroke-width="10" stroke-linecap="round" stroke-dasharray="{f*157.08} 157.08"/><text x="60" y="58" text-anchor="middle" fill="#e6ecff" font-size="16" font-weight="700">{f:.1%}</text></svg>'
def pct_color(p): return "#ef4444" if p >= .65 else "#22c55e" if p <= .35 else "#f59e0b"

with st.sidebar:
    st.markdown('<div class="brand">🛡️ PhishSentinel <span>| Risk Dashboard</span></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Scanner", "Models", "Robustness", "Logs", "Alerts", "Settings"], label_visibility="collapsed")
    st.caption(ps.DISCLAIMER)
M = R["models"][ss.model]; mdl = models[ss.model]

SAMPLES = {"Choose a test case": "",
 "AT-01 · Legitimate password reset": "Hello, we received a request to reset your password. If you did not request this, ignore this message. Visit the portal directly from your bookmarks. No action is needed otherwise.",
 "AT-02 · Phishing, unfamiliar wording": "Hi, the finance side flagged a hiccup with your pay details; please tidy it up here: hxxp://bit[.]ly/58 Thanks, really appreciate it.",
 "Obvious phishing": "URGENT!!! Your account will be suspended within 24 hours. Verify your password immediately at hxxp://192[.]0[.]2[.]9/login. Dear customer, final notice!",
 "Legitimate urgent payroll notice": "Reminder: payroll closes Friday. Confirm any bank detail changes in the HR system; do not click links in email."}
def load_sample(): ss.msg = SAMPLES[ss.sample]

def kpis():
    a, b, c, d = st.columns(4); ok = M["macro_f1"] >= .7
    a.markdown(f'<div class="kpi {"g" if ok else ""}"><span class="ic">{"🛡️" if ok else "⚠️"}</span><div>Model Health:<br><b class="{"ok" if ok else "wn"}">{"Good" if ok else "Weak"} · F1 {M["macro_f1"]:.2f}</b></div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="kpi"><span class="ic">🧠</span><div>Active Models:<br><b>{len(models)} / {len(models)}</b></div></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="kpi"><span class="ic">📈</span><div>Messages Scanned:<br><b>{ss.n}</b> <span style="color:var(--mu)">this session</span></div></div>', unsafe_allow_html=True)
    d.markdown(f'<div class="kpi r"><span class="ic">🚨</span><div>High-risk Flags:<br><b class="er">{ss.high}</b></div></div>', unsafe_allow_html=True)

def result_card():
    L = ss.last
    if not L: st.caption("Scan a message to see its risk, decision and safe explanation."); return
    c = pct_color(L["p"]); C = 2 * 3.14159 * 48
    st.markdown(f'<div style="display:flex;gap:14px;align-items:center"><svg width="110" height="110" viewBox="0 0 120 120"><circle cx="60" cy="60" r="48" fill="none" stroke="#1f2b45" stroke-width="12"/><circle cx="60" cy="60" r="48" fill="none" stroke="{c}" stroke-width="12" stroke-linecap="round" stroke-dasharray="{L["p"]*C} {C}" transform="rotate(-90 60 60)"/><text x="60" y="66" text-anchor="middle" font-size="22" font-weight="700" fill="#e6ecff">{L["p"]:.0%}</text></svg><div style="font-size:18px;font-weight:700;color:{c}">{ps.LABEL[L["d"]]}</div></div>', unsafe_allow_html=True)
    st.markdown("**Why:** " + ("; ".join(html.escape(x) for x in L["cues"]) or "no rule cues matched") + (f". Model terms: {html.escape(', '.join(L['terms']))}" if L["terms"] else ""))
    st.caption(ps.DISCLAIMER)

if page == "Overview":
    st.markdown("#### AgentGuard-style Overview")
    kpis(); lc, rc = st.columns([1.55, 1])
    with lc:
        st.markdown('<div class="sec">Model Performance</div>', unsafe_allow_html=True)
        with st.container(border=True):
            h = M["hist"]; ph, lg = h["phishing"], h["legitimate"]; sp, sl = max(sum(ph), 1), max(sum(lg), 1)
            ser = {"Phishing": ([v / sp * 100 for v in ph], "#8b5cf6"), "Legitimate": ([v / sl * 100 for v in lg], "#2de2c4")}
            ym = max(20, -(-max(max(v) for v, _ in ser.values()) // 20) * 20)
            st.markdown(f'<div class="pt">Risk Score Distribution on Held-out Campaigns ({html.escape(ss.model)})</div><div class="lg"><span><i style="background:#2de2c4"></i>Legitimate</span><span><i style="background:#8b5cf6"></i>Phishing</span></div>' + line_chart([f"{i*10}%" for i in range(10)], ser, ym), unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1, st.container(border=True):
            st.markdown(f'<div class="pt">Phishing Recall</div><div class="gs">{gauge(M["recall"], "#8b5cf6")}<div>Precision: <b>{M["precision"]:.2f}</b><br>TP / FN: <b>{M["confusion"]["tp"]} / {M["confusion"]["fn"]}</b></div></div>', unsafe_allow_html=True)
        with g2, st.container(border=True):
            st.markdown(f'<div class="pt">Abstain Rate</div><div class="gs">{gauge(M["abstain_rate"], "#2de2c4")}<div>Acc. on answered: <b>{(M["acc_on_answered"] or 0):.2f}</b><br>FPR legit urgent: <b>{M["fpr_legit_urgent"]:.0%}</b></div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="pt">Real-Time Activity Logs</div>' + table(["Timestamp", "ID", "Type", "Message"], [(e["ts"], e["id"], f'<span class="{tyc(e["ty"])}">{e["ty"]}</span>', html.escape(e["msg"])) for e in all_events()[:6]]), unsafe_allow_html=True)
    with rc:
        st.markdown('<div class="sec">Detection &amp; Model Metrics</div>', unsafe_allow_html=True)
        with st.container(border=True):
            items = [(k.replace("TF-IDF + ", ""), v["macro_f1"]) for k, v in R["models"].items()] + [("LR random split", R["random_split_contrast"]["macro_f1"])]
            st.markdown('<div class="pt">Macro-F1 per Model</div><div class="bars">' + "".join(f'<div><span>{v:.2f}</span><b style="height:{max(v,.02)*85}%"></b>{k}</div>' for k, v in items) + '</div><div style="font-size:11px;color:var(--mu);margin-top:6px">Random split is inflated by shared templates.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            rows = [(k, f'<span class="{"ok" if v["macro_f1"]>=.7 else "er"}">{"Active" if v["macro_f1"]>=.7 else "Weak"}</span>', f'{v["macro_f1"]:.2f}', f'{v["abstain_rate"]:.0%}', f'{v["fpr_legit_urgent"]:.0%}') for k, v in R["models"].items()]
            st.markdown('<div class="pt">Models (held-out campaigns)</div>' + table(["Name", "Status", "F1", "Abstain", "FPR-LU"], rows), unsafe_allow_html=True)
        with st.container(border=True):
            al = [e for e in all_events() if e["ty"] in ("WRN", "ERR", "CRIT")][:5] or [{"ty": "INF", "id": "-", "msg": "No alerts"}]
            col = {"INF": "#4f8cff", "WRN": "#f59e0b", "ERR": "#ef4444", "CRIT": "#ef4444"}
            st.markdown('<div class="pt">Alert Notifications</div>' + "".join(f'<div class="al"><span class="d" style="background:{col[e["ty"]]}"></span><span class="{tyc(e["ty"])}">{"Critical" if e["ty"]=="CRIT" else e["ty"]}</span><span>{html.escape(e["msg"])}</span></div>' for e in al), unsafe_allow_html=True)

elif page == "Scanner":
    st.markdown("#### Scanner"); kpis(); a, b = st.columns([1.6, 1])
    with a, st.container(border=True):
        st.markdown('<div class="pt">Scan a message (offline, links never opened)</div>', unsafe_allow_html=True)
        st.selectbox("Test case", list(SAMPLES), key="sample", on_change=load_sample, label_visibility="collapsed")
        st.text_area("Message", key="msg", height=150, placeholder="Paste a simulated email or SMS. Use defanged links (hxxp://example[.]com).", label_visibility="collapsed")
        x, y = st.columns([1, 2]); ch = x.selectbox("Channel", ["email", "sms"], label_visibility="collapsed")
        if y.button("Scan message"):
            t = ss.msg.strip()
            if not t: st.warning("Paste or load a message first.")
            else:
                p, d = ps.score_message(t, ch, mdl); cues, terms = ps.explain(t, models["TF-IDF + LogReg"])
                ss.last = dict(text=t, ch=ch, p=p, d=d, cues=cues, terms=terms, model=ss.model); ss.n += 1; ss.high += int(d == 1)
                ss.events.insert(0, dict(ts=clock(), id=f"M-{ss.n:03d}", ty={1: "CRIT", 0: "INF", -1: "WRN"}[d], msg=f"{ps.LABEL[d]} ({p:.0%}) via {ss.model}: {t[:40]}..."))
                st.rerun()
    with b, st.container(border=True):
        st.markdown('<div class="pt">Risk result</div>', unsafe_allow_html=True); result_card()

elif page == "Models":
    st.markdown("#### Models"); cm = M["confusion"]; a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown(f'<div class="pt">Holdout confusion · {html.escape(ss.model)}</div>' + table(["", "Pred low", "Pred high"], [("Legitimate", cm["tn"], cm["fp"]), ("Phishing", cm["fn"], cm["tp"])]) + f'<div style="color:var(--mu);font-size:12px;margin-top:6px">Abstained: {cm["abstain"]}</div>', unsafe_allow_html=True)
    with b, st.container(border=True):
        st.markdown('<div class="pt">Mean risk per held-out campaign (LogReg)</div>' + table(["Campaign", "Label", "n", "Mean risk"], [(k, "phishing" if v["label"] else "legit", v["n"], f'{v["mean_risk"]:.0%}') for k, v in R["per_campaign"].items()]), unsafe_allow_html=True)
    with st.container(border=True):
        at = R["acceptance"]; st.markdown('<div class="pt">Acceptance tests</div>' + table(["Case", "Condition", "Risk", "Decision"], [("AT-01", at["AT-01"]["case"], f'{at["AT-01"]["risk"]:.0%}', at["AT-01"]["decision"]), ("AT-02", at["AT-02"]["case"], f'{at["AT-02"]["risk"]:.0%}', at["AT-02"]["decision"]), ("AT-03", at["AT-03"]["case"], f'{at["AT-03"]["base_risk"]:.0%}', "see Robustness")]), unsafe_allow_html=True)

elif page == "Robustness":
    st.markdown("#### Robustness"); a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown(f'<div class="pt">Holdout drift · {html.escape(ss.model)}</div>' + table(["Perturbation", "Mean |shift|", "Decision flips"], [(k, f'{v["mean_abs_shift"]:.3f}', f'{v["decision_flip_rate"]:.0%}') for k, v in M["drift"].items()]), unsafe_allow_html=True)
    with b, st.container(border=True):
        st.markdown('<div class="pt">Perturb your last scanned message</div>', unsafe_allow_html=True); L = ss.last
        if not L: st.caption("Scan a message first.")
        elif st.button("Run perturbations"):
            m = models[L["model"]]; base = m.predict_proba([L["text"] + " urlfeat_none"])[0, 1]
            rows = [("Original", base)] + [(k, m.predict_proba([f(L["text"]) + " urlfeat_none"])[0, 1]) for k, f in ps.PERTURB.items()]
            st.markdown(table(["Variant", "Risk", "Shift"], [(k, f'<span style="color:{pct_color(p)}">{p:.0%}</span>', "" if k == "Original" else f"{(p-base)*100:+.0f}") for k, p in rows]), unsafe_allow_html=True)
            st.caption("Large shifts show the model limitation (AT-03).")

elif page == "Logs":
    st.markdown("#### Logs")
    with st.container(border=True): st.markdown(table(["Timestamp", "ID", "Type", "Message"], [(e["ts"], e["id"], f'<span class="{tyc(e["ty"])}">{e["ty"]}</span>', html.escape(e["msg"])) for e in all_events()]), unsafe_allow_html=True)

elif page == "Alerts":
    st.markdown("#### Alerts")
    with st.container(border=True):
        al = [e for e in all_events() if e["ty"] in ("WRN", "ERR", "CRIT")]
        st.markdown("".join(f'<div class="al"><span class="d" style="background:{"#f59e0b" if e["ty"]=="WRN" else "#ef4444"}"></span><span class="{tyc(e["ty"])}">{e["ty"]}</span><span>{html.escape(e["msg"])}</span></div>' for e in al) or "No alerts.", unsafe_allow_html=True)

else:
    st.markdown("#### Settings")
    with st.container(border=True):
        st.selectbox("Active model", list(models), key="model"); st.json(R["settings"]); st.json({"holdout_campaigns": R["split"]["holdout_campaigns"], "dedup": R["dedup"]})
        st.caption(ps.DISCLAIMER)
