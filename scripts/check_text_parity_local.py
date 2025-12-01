# python
# 保存例: tools/check_text_parity_local.py
import pandas as pd
from pathlib import Path

p = Path("sample_test.csv")
if not p.exists():
    raise SystemExit("sample_test.csv が見つかりません: " + str(p))

df = pd.read_csv(p)

try:
    from train import create_inference_texts as train_builder  # train.py
    has_train = True
except Exception as e:
    print("train.create_inference_texts の import に失敗:", e)
    has_train = False

def fallback(df):
    text_cols = ["body","text","comment","body_text"]
    text_col = next((c for c in text_cols if c in df.columns), None)
    if text_col is None:
        raise RuntimeError("テキスト列が見つかりません")
    extras = [c for c in ("rule","positive_example_1","negative_example_1",
                          "positive_example_2","negative_example_2") if c in df.columns]
    d = df.copy()
    d[text_col] = d[text_col].fillna("").astype(str)
    if extras:
        for c in extras:
            d[c] = d[c].fillna("").astype(str)
        texts = (d[text_col] + " " + d[extras].agg(" ".join, axis=1)).astype(str).tolist()
    else:
        texts = d[text_col].astype(str).tolist()
    return texts

texts_fb = fallback(df)
if has_train:
    texts_tr = train_builder(df.copy())
else:
    texts_tr = None

print("has_train_builder:", has_train)
for i in range(min(5, len(texts_fb))):
    a = texts_tr[i] if texts_tr is not None else "<no-train>"
    b = texts_fb[i]
    if a != b:
        print(f"DIFF at {i}\n--- train ---\n{a}\n--- fallback ---\n{b}\n")
        break
else:
    print("最初の確認行に差分なし。")