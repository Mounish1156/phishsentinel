# PhishSentinel Message Risk Scanner (ML-07)

Educational triage tool: classifies simulated emails/SMS by phishing risk, can **abstain**, and explains risk cues **without opening links** (NFR-01). Decision support, not a guarantee of safety (NFR-03).

## UI
Dark "systems dashboard" layout (sidebar nav, KPI strip, risk-distribution chart, gauges, model table, activity logs, alert feed). Pages: Overview, Scanner, Models, Robustness, Logs, Alerts, Settings. Pick the active model in Settings.

## Run locally
```bash
pip install -r requirements.txt
python make_data.py      # regenerate synthetic corpus (optional, CSV is included)
python train.py          # dedup -> campaign split -> 3 models -> results.json + experiment_log.jsonl
python -m pytest -q tests
streamlit run app.py
```

## Host on Streamlit Community Cloud
1. Create a GitHub repo and push this folder (keep `models/`, `data/`, `outputs/`).
2. Go to share.streamlit.io, choose **New app**, pick the repo, branch `main`, main file `app.py`.
3. Deploy. If `models/` is missing the app retrains itself on first load (takes a few seconds).

Docker alternative: `docker build -t phishsentinel . && docker run -p 8501:8501 phishsentinel`

## Requirement traceability
| Req | Where | Evidence |
|---|---|---|
| FR-01 dedup before split | `phishsentinel.dedup`, `cross_split_similarity` | `results.json` -> `dedup`, `split.pairs_over_threshold` |
| FR-02 rules vs TF-IDF LR vs third model | `RuleModel`, `tfidf_lr`, `tfidf_rf` | `results.json` -> `models` |
| FR-03 campaign holdout | `HOLDOUT` list, `run_pipeline` | `split`, `random_split_contrast` |
| FR-04 abstain + explanation | `decide` (band 0.35-0.65), `explain` | App "Risk result" card |
| FR-05 FPR + drift | `metrics` (`fpr_legit_urgent`), `drift`, `PERTURB` | `models.*.drift`, app Robustness card |
| NFR-01/02/03 | text-only code, synthetic data, disclaimer | `tests/test_pipeline.py`, app footer |
| AT-01/02/03 | `run_pipeline` -> `acceptance` | App "Acceptance tests" card + `outputs/results.json` |

## Honest limitations (put these in your report)
- The corpus is **locally generated from templates** (700 raw rows, ~600 after dedup, 10 campaigns). It is small and templated, so results are optimistic compared with real mail.
- The **rules model was written knowing the corpus**, so its high holdout score is not independent evidence. The campaign holdout (TF-IDF LR) is the fairer number; the random-split score (1.00) shows how template leakage inflates results.
- `sender_domain_type` and `sanitized_url_features` are strong shortcuts (target/feature leakage risk, T3). Run an ablation without them.
- RandomForest collapses on unseen campaigns; keep that result in the report rather than tuning it away.
- Mentor-supplied unseen campaign and legitimate set (section 08) are **not included**. Add them as a new CSV, rerun `train.py`, and log the change.
- Still your work (T1, T2 provenance sheet, T3 EDA plots, T7 report, SRS.pdf, 3-5 minute demo, AI-usage disclosure). You must be able to explain every file here.
