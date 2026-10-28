"""PhishSentinel core: cleaning, dedup (FR-01), models (FR-02), campaign holdout (FR-03),
abstain + safe explanation (FR-04), FPR + robustness drift (FR-05). Text only: nothing is opened or fetched (NFR-01)."""
import json, re, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent
SEED, LO, HI, DEDUP_THR = 42, 0.35, 0.65, 0.90
HOLDOUT = ["P2_vague_favour", "P3_parcel_sms", "L2_payroll_notice", "L4_benefits_deadline"]
DISCLAIMER = "Decision support, not a guarantee of safety. Text is scanned only: no link is opened, fetched or executed."

CUES = [  # (name, weight, regex, defanged-weight, defanged-regex)
 ("Urgency or deadline pressure", .20, r"\b(urgent|immediately|asap|final notice|within \d+ hours?|right away|act now|today)\b", 0, None),
 ("Asks for password, login or codes", .28, r"\b(verify your password|re-?enter your (login|credentials|password)|confirm your login|your credentials|six digit code|enter the code)\b", 0, None),
 ("Money or payment details requested", .18, r"\b(gift cards?|wire|supplier transfer|pay details|payment form|redelivery fee)\b", 0, None),
 ("Threat of account loss", .18, r"\b(suspended|locked|returns to sender)\b", 0, None),
 ("Generic or pressuring greeting", .08, r"\b(dear \w+|confidential|keep this between us)\b", 0, None),
 ("Shortened link", .30, r"\b(bit\.ly|tinyurl\.com)\b", .22, r"(bit\[\.\]ly|tinyurl\[\.\]com)"),
 ("Raw IP or lookalike link", .25, r"(\b\d{1,3}(\.\d{1,3}){3}\b|secure-login)", .20, r"(\d{1,3}(\[\.\]\d{1,3}){3}|secure-login)"),
 ("Vague request, unusual framing", .12, r"\b(favour|hiccup|sort(ing)? .* out|mismatch|tidy)\b", 0, None),
]
MITIGATORS = [
 ("Tells reader to use the portal directly", -.25, r"(directly|bookmarks|do not click links)"),
 ("Says to ignore if not requested", -.20, r"(did not request|ignore this)"),
 ("Says no action is needed", -.12, r"(no action|no payment|automated)"),
]

def rule_cues(text):
    s, raw, found = text.lower(), .12, []
    for name, w, rx, dw, drx in CUES:
        if re.search(rx, s): raw += w; found.append(name)
        elif drx and re.search(drx, s): raw += dw; found.append(name + " (defanged, weaker)")
    for name, w, rx in MITIGATORS:
        if re.search(rx, s): raw += w; found.append("Reassuring: " + name)
    if text.count("!") > 1: raw += .06; found.append("Heavy punctuation")
    return 1 / (1 + np.exp(-7 * (raw - .5))), found

class RuleModel:
    def predict_proba(self, texts):
        p = np.array([rule_cues(t)[0] for t in texts]); return np.c_[1 - p, p]

def make_text(r):
    return f"{r['subject_or_title']}. {r['body_text']} urlfeat_{r['sanitized_url_features']} sender_{r['sender_domain_type']} chan_{r['channel']}"

def tfidf_lr():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=3, max_iter=1000, class_weight="balanced", random_state=SEED))
def tfidf_rf():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=1))

def decide(p): return 1 if p >= HI else 0 if p <= LO else -1   # -1 = abstain
LABEL = {1: "High risk", 0: "Low risk", -1: "Abstain: needs human review"}

def clean(raw):
    df = raw.copy()
    df["channel"] = df.channel.str.lower().str.strip()
    df["body_text"] = df.body_text.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.body_text.str.len() > 0]
    df["text"] = df.apply(make_text, axis=1)
    return df.reset_index(drop=True)

def norm(t): return re.sub(r"\b(priya|marcus|elena|tomas|aisha|chen|noah|fatima|lucas|mei)\b|\d+", "", t.lower())

def dedup(df, thr=DEDUP_THR):
    X = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(df.text.map(norm))
    S = cosine_similarity(X); keep, drop = [], []
    for i in range(len(df)):
        if keep and S[i, keep].max() >= thr: drop.append(i)
        else: keep.append(i)
    return df.iloc[keep].reset_index(drop=True), {"before": len(df), "after": len(keep), "removed": len(drop), "threshold": thr}

def cross_split_similarity(tr, te):
    v = TfidfVectorizer(ngram_range=(1, 2)).fit(pd.concat([tr.text, te.text]).map(norm))
    S = cosine_similarity(v.transform(te.text.map(norm)), v.transform(tr.text.map(norm)))
    return {"max_train_test_similarity": float(S.max()), "pairs_over_threshold": int((S >= DEDUP_THR).sum())}

def metrics(model, df):
    p = model.predict_proba(df.text.tolist())[:, 1]; d = np.array([decide(x) for x in p]); y = df.label.values
    ans = d != -1; yh = np.where(p >= .5, 1, 0)
    lu = (df.legit_urgent.values == 1)
    fpr_lu = float(((d == 1) & lu).sum() / max(lu.sum(), 1))
    fpr_all = float(((d == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
    cm = {"tn": int(((d == 0) & (y == 0)).sum()), "fp": int(((d == 1) & (y == 0)).sum()), "fn": int(((d == 0) & (y == 1)).sum()),
          "tp": int(((d == 1) & (y == 1)).sum()), "abstain": int((d == -1).sum())}
    return {"macro_f1": float(f1_score(y, yh, average="macro")), "precision": float(precision_score(y, yh, zero_division=0)),
            "recall": float(recall_score(y, yh, zero_division=0)), "fpr_all_legit": fpr_all, "fpr_legit_urgent": fpr_lu,
            "abstain_rate": float((~ans).mean()), "acc_on_answered": float((d[ans] == y[ans]).mean()) if ans.any() else None,
            "confusion": cm, "p": p.tolist()}

PERTURB = {
 "Defang wording ([.] to ' dot ')": lambda t: t.replace("[.]", " dot "),
 "Strip punctuation": lambda t: re.sub(r"[!?.,:;]", "", t),
 "Extra exclamation marks": lambda t: t.replace(".", "!!"),
 "UPPERCASE": lambda t: t.upper(),
 "Leetspeak keywords": lambda t: re.sub("password","p@ssw0rd",re.sub("verify","v3rify",re.sub("login","l0gin",re.sub("urgent","urg3nt",t,flags=re.I),flags=re.I),flags=re.I),flags=re.I),
 "Pad with benign sentence": lambda t: t + " Thanks again for helping with the team lunch and the quarterly planning notes.",
}
def drift(model, texts):
    base = model.predict_proba(texts)[:, 1]; out = {}
    for k, f in PERTURB.items():
        p = model.predict_proba([f(t) for t in texts])[:, 1]
        out[k] = {"mean_abs_shift": float(np.abs(p - base).mean()), "decision_flip_rate": float(np.mean([decide(a) != decide(b) for a, b in zip(base, p)]))}
    return out

def explain(text, lr):
    """Safe explanation: names cue categories and TF-IDF terms only; never quotes or opens links."""
    _, cues = rule_cues(text); terms = []
    try:
        tf, clf = lr.steps[0][1], lr.steps[-1][1]; x = tf.transform([text]); names = tf.get_feature_names_out()
        c = x.toarray()[0] * clf.coef_[0]; terms = [names[i] for i in np.argsort(-c)[:5] if c[i] > 0 and not names[i].startswith(("urlfeat", "sender", "chan"))]
    except Exception: pass
    return cues, terms

def run_pipeline(log=True):
    t0 = time.time(); raw = pd.read_csv(ROOT / "data/raw_messages.csv")
    df = clean(raw); df, dd = dedup(df)                                   # FR-01: dedup BEFORE splitting
    df.drop(columns="text").to_csv(ROOT / "data/clean_messages.csv", index=False)
    te = df[df.campaign_group.isin(HOLDOUT)].reset_index(drop=True); tr = df[~df.campaign_group.isin(HOLDOUT)].reset_index(drop=True)  # FR-03
    models = {"Rules": RuleModel(), "TF-IDF + LogReg": tfidf_lr().fit(tr.text, tr.label), "TF-IDF + RandomForest": tfidf_rf().fit(tr.text, tr.label)}
    res = {"dedup": dd, "split": {"train": len(tr), "holdout": len(te), "holdout_campaigns": HOLDOUT, "train_campaigns": sorted(tr.campaign_group.unique().tolist()),
           **cross_split_similarity(tr, te)}, "settings": {"seed": SEED, "abstain_band": [LO, HI], "dedup_threshold": DEDUP_THR}, "models": {}, "per_campaign": {}}
    for name, m in models.items():
        r = metrics(m, te); pp = np.array(r.pop("p")); yy = te.label.values
        hist = {k: np.histogram(pp[yy == v], bins=10, range=(0, 1))[0].tolist() for k, v in (("phishing", 1), ("legitimate", 0))}
        res["models"][name] = {**r, "hist": hist, "drift": drift(m, te.text.tolist()[:80])}
    pc = metrics(models["TF-IDF + LogReg"], te)["p"]; te["p"] = pc
    res["per_campaign"] = {c: {"mean_risk": float(g.p.mean()), "n": len(g), "label": int(g.label.iloc[0])} for c, g in te.groupby("campaign_group")}
    a, b = train_test_split(df, test_size=.25, random_state=SEED, stratify=df.label)   # contrast: random split inflates results
    rl = tfidf_lr().fit(a.text, a.label); rm = metrics(rl, b); rm.pop("p")
    res["random_split_contrast"] = {"macro_f1": rm["macro_f1"], "fpr_legit_urgent": rm["fpr_legit_urgent"]}
    res["class_mix"] = df.groupby("campaign_group").label.first().map({1: "Phishing", 0: "Legitimate"}).value_counts().to_dict()
    res["class_counts"] = {"Phishing": int((df.label == 1).sum()), "Legitimate": int(((df.label == 0) & (df.legit_urgent == 0)).sum()), "Legitimate urgent": int(df.legit_urgent.sum())}
    lr = models["TF-IDF + LogReg"]
    at = {}
    pr = te[te.campaign_group == "L2_payroll_notice"].iloc[0] if len(te) else None
    leg = df[df.campaign_group == "L1_password_reset"].iloc[0]; ph = te[te.campaign_group == "P2_vague_favour"].iloc[0]
    p1 = float(lr.predict_proba([leg.text])[0, 1]); p2 = float(lr.predict_proba([ph.text])[0, 1])
    at["AT-01"] = {"case": "Legitimate password-reset notice", "risk": p1, "decision": LABEL[decide(p1)], "fpr_legit_urgent_holdout": res["models"]["TF-IDF + LogReg"]["fpr_legit_urgent"]}
    at["AT-02"] = {"case": "Phishing with unfamiliar phrasing (campaign excluded from training)", "risk": p2, "decision": LABEL[decide(p2)], "campaign_mean_risk": res["per_campaign"]["P2_vague_favour"]["mean_risk"]}
    at["AT-03"] = {"case": "Defang / punctuation changes", "base_risk": p2, "shifts": {k: float(lr.predict_proba([f(ph.text)])[0, 1]) for k, f in PERTURB.items()}}
    res["acceptance"] = at; res["seconds"] = round(time.time() - t0, 1)
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "outputs").mkdir(exist_ok=True)
    joblib.dump(lr, ROOT / "models/tfidf_logreg_v1.joblib"); joblib.dump(models["TF-IDF + RandomForest"], ROOT / "models/tfidf_rf_v1.joblib")
    (ROOT / "outputs/results.json").write_text(json.dumps(res, indent=2))
    if log:
        with open(ROOT / "outputs/experiment_log.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": res["settings"], "holdout": HOLDOUT,
                                "macro_f1": {k: v["macro_f1"] for k, v in res["models"].items()}}) + "\n")
    return res

def load_models():
    if not (ROOT / "models/tfidf_logreg_v1.joblib").exists(): run_pipeline(log=False)
    return {"TF-IDF + LogReg": joblib.load(ROOT / "models/tfidf_logreg_v1.joblib"), "TF-IDF + RandomForest": joblib.load(ROOT / "models/tfidf_rf_v1.joblib"), "Rules": RuleModel()}

def score_message(text, channel, model):
    t = f"{text}. urlfeat_none sender_unknown chan_{channel}"
    p = float(model.predict_proba([t])[0, 1]); return p, decide(p)
