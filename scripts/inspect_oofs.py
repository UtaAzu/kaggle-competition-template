# ...existing code...
#!/usr/bin/env python3
"""
Quick inspect oof.csv across experiments: prints columns, n_rows, sample head.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(".")
EXP_ROOT = ROOT / "experiments"
O = list(EXP_ROOT.rglob("*/artifacts/oof.csv")) + list(EXP_ROOT.rglob("*/artifacts/*/oof.csv"))
seen = set()
for p in sorted(set(O)):
    key = "/".join(p.parts[-4:])
    if key in seen:
        continue
    seen.add(key)
    print("----", p)
    try:
        df = pd.read_csv(p, nrows=5)
        df_all = pd.read_csv(p, low_memory=False)
    except Exception as e:
        print("  read error:", e)
        continue
    cols = [c.lower() for c in df.columns.astype(str)]
    print("  cols:", cols)
    print("  n_rows:", len(df_all))
    sample = df.head(3).to_dict(orient="records")
    print("  head:", sample)
    flags = {k: (k in cols) for k in ("oof_pred","rule_violation","row_id","group","target")}
    print("  has_keys:", flags)
# ...existing code...