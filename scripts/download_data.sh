#!/usr/bin/env bash
# Get the "Customer Support on Twitter" dataset -> data/raw/twcs.csv
#
# Tries a no-auth Hugging Face mirror first; falls back to Kaggle (needs a token
# at ~/.kaggle/kaggle.json from https://www.kaggle.com/settings -> Create Token).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

if [ -f data/raw/twcs.csv ]; then
  echo "data/raw/twcs.csv already present ($(wc -l < data/raw/twcs.csv) rows) — skipping."
  exit 0
fi

echo "Trying Hugging Face mirror (no login)..."
if curl -fL --retry 3 --max-time 1800 \
     "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv" \
     -o data/raw/twcs.csv ; then
  echo "Done: $(wc -l < data/raw/twcs.csv) rows in data/raw/twcs.csv"
  exit 0
fi

echo "HF mirror failed — falling back to Kaggle."
rm -f data/raw/twcs.csv
.venv/bin/python -c "import kaggle" 2>/dev/null || .venv/bin/pip install -q kaggle
.venv/bin/kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
[ -f data/raw/twcs.csv ] || { [ -f data/raw/twcs/twcs.csv ] && mv data/raw/twcs/twcs.csv data/raw/twcs.csv; }
echo "Done: $(wc -l < data/raw/twcs.csv) rows in data/raw/twcs.csv"
