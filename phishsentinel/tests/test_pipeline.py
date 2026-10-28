import json, re, pathlib, pandas as pd, phishsentinel as ps
R = json.loads((ps.ROOT / "outputs/results.json").read_text())
def test_dedup_removed_near_duplicates(): assert R["dedup"]["removed"] > 0 and R["dedup"]["after"] >= 400
def test_no_train_test_template_leak(): assert R["split"]["pairs_over_threshold"] == 0
def test_campaign_holdout_disjoint(): assert not set(R["split"]["holdout_campaigns"]) & set(R["split"]["train_campaigns"])
def test_three_models_compared(): assert len(R["models"]) == 3
def test_abstain_band():
    assert ps.decide(.5) == -1 and ps.decide(.9) == 1 and ps.decide(.1) == 0
def test_no_network_imports():
    src = pathlib.Path(ps.ROOT / "phishsentinel.py").read_text()
    assert not re.search(r"^\s*(import|from)\s+(requests|urllib|socket|http)", src, re.M)
def test_schema():
    df = pd.read_csv(ps.ROOT / "data/raw_messages.csv")
    need = {"message_id","channel","subject_or_title","body_text","sanitized_url_features","sender_domain_type","label","campaign_group","collected_date"}
    assert need <= set(df.columns) and len(df) >= 400
def test_acceptance_logged(): assert {"AT-01", "AT-02", "AT-03"} <= set(R["acceptance"])
