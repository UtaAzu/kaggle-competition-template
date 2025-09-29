#!/usr/bin/env bash
set -euo pipefail

# Package code into a folder for Kaggle Dataset upload (UI or CLI)
# Usage: scripts/make_code_dataset.sh [OUTPUT_DIR]

OUT_DIR=${1:-code-dataset}
mkdir -p "$OUT_DIR"

copy() {
  local src=$1; local dst=$2
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"
  fi
}

copy train.py "$OUT_DIR/train.py"
copy predict.py "$OUT_DIR/predict.py"
copy src "$OUT_DIR/src"
copy configs "$OUT_DIR/configs"

# Helper README for the code dataset
cat > "$OUT_DIR/README.txt" << 'EOF'
This is a reusable code bundle for Kaggle Notebooks.
How to use in a notebook after adding this dataset:

!cp -r /kaggle/input/<this-dataset-slug>/src ./src
!cp -r /kaggle/input/<this-dataset-slug>/configs ./configs
!cp /kaggle/input/<this-dataset-slug>/train.py .
!cp /kaggle/input/<this-dataset-slug>/predict.py .
!pip -q install pyyaml
!python train.py --config configs/tfidf_baseline.yaml
EOF

# Optional Kaggle CLI metadata template
mkdir -p tools/code-dataset
cat > tools/code-dataset/metadata.json << 'EOF'
{
  "title": "REPLACE_WITH_TITLE",
  "id": "REPLACE_WITH_USERNAME/REPLACE_WITH_SLUG",
  "licenses": [
    { "name": "CC0-1.0" }
  ],
  "subtitle": "Reusable Kaggle code bundle (train/predict/src/configs)",
  "description": "Code bundle generated from repository for reuse across notebooks.",
  "isPrivate": true
}
EOF

echo "Code dataset prepared at: $OUT_DIR"