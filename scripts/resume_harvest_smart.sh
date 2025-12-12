#!/bin/bash
set -e

# Data Output Directory
DATA_DIR="data/raw"
KEYWORDS=("Yape" "Yapear")
PY=".venv/bin/python"

echo "=== RESUMING HARVEST SMARTLY ==="

# 1. REMAINING 2019 (From Dec 26)
echo "--- Resuming 2019 (Dec 26 - Dec 31) ---"
$PY -m src.news_harvester.cli harvest \
    --keyword "${KEYWORDS[@]}" \
    --from "2019-12-26" --to "2019-12-31" \
    --sources gdelt --format parquet \
    --output "${DATA_DIR}/yape_2019_part2.parquet"

# 2. SKIP 2020 (Done)
echo "--- Skipping 2020 (Already complete) ---"

# 3. REMAINING 2021 (From July 04)
echo "--- Resuming 2021 (July 04 - Dec 31) ---"
$PY -m src.news_harvester.cli harvest \
    --keyword "${KEYWORDS[@]}" \
    --from "2021-07-04" --to "2021-12-31" \
    --sources gdelt --format parquet \
    --output "${DATA_DIR}/yape_2021_part2.parquet"

# 4. FULL 2022-2025
for YEAR in 2022 2023 2024 2025; do
    echo "--- Harvesting Full Year: $YEAR ---"
    $PY -m src.news_harvester.cli harvest \
        --keyword "${KEYWORDS[@]}" \
        --from "${YEAR}-01-01" --to "${YEAR}-12-31" \
        --sources gdelt --format parquet \
        --output "${DATA_DIR}/yape_${YEAR}.parquet"
done

echo "=== HARVEST COMPLETED. INITIATING MERGE... ==="

# Python one-liner to merge parts
$PY -c "
import pandas as pd
from pathlib import Path

data = Path('${DATA_DIR}')

# Merge 2019
p1_19 = data / 'yape_2019_part1.parquet'
p2_19 = data / 'yape_2019_part2.parquet'
if p1_19.exists() and p2_19.exists():
    df1 = pd.read_parquet(p1_19)
    df2 = pd.read_parquet(p2_19)
    # Ensure columns match via concatenation
    full = pd.concat([df1, df2], ignore_index=True)
    full.to_parquet(data / 'yape_2019.parquet', index=False)
    print('Merged 2019')
    # p1_19.unlink(); p2_19.unlink() # Safer to keep for now

# Merge 2021
p1_21 = data / 'yape_2021_part1.parquet'
p2_21 = data / 'yape_2021_part2.parquet'
if p1_21.exists() and p2_21.exists():
    df1 = pd.read_parquet(p1_21)
    df2 = pd.read_parquet(p2_21)
    full = pd.concat([df1, df2], ignore_index=True)
    full.to_parquet(data / 'yape_2021.parquet', index=False)
    print('Merged 2021')
"

echo "=== RESUME PROCESS FINISHED ==="
