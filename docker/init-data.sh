#!/bin/sh
# Init container: populate the PV with the HTS index.
# Downloads the raw USITC export and builds the index ONCE; skips if the
# index already exists on the volume (idempotent re-deploys).
set -eu

DATA_DIR="${DATA_DIR:-/data}"
INDEX="$DATA_DIR/hts_index.json"
RAW="$DATA_DIR/hts_raw.json"

if [ -s "$INDEX" ]; then
  echo "index already present: $INDEX ($(wc -c < "$INDEX") bytes), skipping"
  exit 0
fi

echo "downloading USITC HTS export to $RAW ..."
curl -fsSL --retry 3 --retry-delay 5 \
  -o "$RAW" \
  "https://hts.usitc.gov/reststop/exportList?from=0100&to=9999&format=JSON&styles=false"

echo "building index ..."
python /app/src/build_index.py --data-dir "$DATA_DIR" 2>/dev/null || \
  python "$(dirname "$0")/../src/build_index.py" --data-dir "$DATA_DIR"

echo "done: $(wc -c < "$INDEX") bytes"
