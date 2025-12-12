#!/bin/bash
set -e

# Data Output Directory
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

# Expanded Keyword List
KEYWORDS=(
  "Yape"
  "Yapear"
  "Yapeo"
  "Yapeando"
  "Yapeas"
  "Yapea"
  "Yapean"
  "Yapeé"
  "Yapeó"
  "Yapeado"
  "Yapeame"
  "Yapearte"
  "Yapearon"
  "Yapearía"
)

echo "=== STARTING FULL HARVEST (2019-2025) WITH EXPANDED KEYWORDS ==="
echo "Keywords: ${KEYWORDS[*]}"
echo "Sources: gdelt, google, rss"

for YEAR in 2019 2020 2021 2022 2023 2024 2025; do
    echo "------------------------------------------------"
    echo "Harvesting Year: $YEAR"
    echo "------------------------------------------------"
    
    # Calculate dates
    FROM_DATE="${YEAR}-01-01"
    TO_DATE="${YEAR}-12-31"
    
    # Output file
    OUTPUT_FILE="${DATA_DIR}/yape_${YEAR}.csv"
    
    # Run Harvest Command
    # Using .venv/bin/python explicitly
    # Sources: gdelt google rss
    # Format: csv (prevents RAM saturation vs parquet append)
    
    .venv/bin/python -m src.news_harvester.cli harvest \
        --keyword "${KEYWORDS[@]}" \
        --from "$FROM_DATE" \
        --to "$TO_DATE" \
        --sources gdelt google rss \
        --format csv \
        --output "$OUTPUT_FILE"
        
    echo "Completed $YEAR -> $OUTPUT_FILE"
done

echo "=== FULL HARVEST COMPLETED ==="
