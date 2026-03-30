#!/usr/bin/env python3
"""
Data download helper for DSA210 Tech Job Market Salary Analysis.

Usage (inside the container):
    python scripts/download_data.py

What this script does:
  1. Downloads the HuggingFace `lukebarousse/data_jobs` dataset and saves it
     as data/raw/jobs_raw.parquet for fast notebook loading.
  2. Prints step-by-step instructions for the BLS OEWS manual download.
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW  = REPO_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ─── 1. HuggingFace dataset (automated) ──────────────────────────────────────
print("=" * 60)
print("Step 1 / 2 — Downloading lukebarousse/data_jobs from HuggingFace")
print("=" * 60)

try:
    from datasets import load_dataset
    dataset = load_dataset("lukebarousse/data_jobs")
    df = dataset["train"].to_pandas()
    print(f"  ✓  Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    parquet_path = DATA_RAW / "jobs_raw.parquet"
    df.to_parquet(parquet_path, index=False)
    size_mb = parquet_path.stat().st_size / 1e6
    print(f"  ✓  Saved to: {parquet_path} ({size_mb:.1f} MB)")
except Exception as exc:
    print(f"  ✗  Failed to download HuggingFace dataset: {exc}", file=sys.stderr)
    sys.exit(1)

print()

# ─── 2. BLS OEWS (manual) ────────────────────────────────────────────────────
OEWS_PATH = DATA_RAW / "oews_national.xlsx"

print("=" * 60)
print("Step 2 / 2 — BLS OEWS data (manual download required)")
print("=" * 60)

if OEWS_PATH.exists():
    print(f"  ✓  File already present at: {OEWS_PATH}")
else:
    print("""
  The BLS Occupational Employment and Wage Statistics (OEWS) dataset
  requires a manual download because the BLS website does not provide
  a stable direct-download URL.

  Instructions:
  ─────────────
  1. Open this page in your browser:
       https://www.bls.gov/oes/tables.htm

  2. Under "National" → find the most recent May survey year.

  3. Download the file labelled:
       "All data (national)" — it is an .xlsx file.

  4. Rename it to  oews_national.xlsx  and place it at:
       data/raw/oews_national.xlsx

     (or /app/data/raw/oews_national.xlsx inside the container)

  5. Re-run this script to confirm the file is found.
""")

print("Done.")
