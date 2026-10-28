import make_data, phishsentinel, pandas as pd
from pathlib import Path
if not Path("data/raw_messages.csv").exists(): make_data.build().to_csv("data/raw_messages.csv", index=False)
r = phishsentinel.run_pipeline()
print("dedup:", r["dedup"]); print("split:", {k: v for k, v in r["split"].items() if "campaign" not in k})
for k, v in r["models"].items(): print(f"{k:24s} macroF1={v['macro_f1']:.2f} FPR(legit urgent)={v['fpr_legit_urgent']:.2f} abstain={v['abstain_rate']:.2f}")
print("random-split contrast:", r["random_split_contrast"]); print("acceptance:", r["acceptance"])
