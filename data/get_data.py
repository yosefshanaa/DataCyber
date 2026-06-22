"""Deterministically fetch the datasets into ``data/raw/``.

Usage:
    python data/get_data.py

Downloads:
  * **NSL-KDD** — the canonical KDDTrain+ / KDDTest+ partitions (primary study).
  * **UNSW-NB15** — the official partitioned train/test CSVs (modern external
    validation, §7 of the analysis).

Both datasets are public and pulled from stable GitHub mirrors so the notebook is
fully reproducible from a clean clone.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"

# (mirror base URL, [filenames]) for each dataset.
SOURCES: list[tuple[str, list[str]]] = [
    ("https://raw.githubusercontent.com/defcom17/NSL_KDD/master",
     ["KDDTrain+.txt", "KDDTest+.txt", "KDDTrain+_20Percent.txt"]),
    ("https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification",
     ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]),
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for base, files in SOURCES:
        for fname in files:
            dest = RAW_DIR / fname
            if dest.exists():
                print(f"[skip] {fname} already present")
                continue
            url = f"{base}/{urllib.parse.quote(fname)}"
            print(f"[get ] {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"       -> {dest} ({dest.stat().st_size:,} bytes)")
    print("Done. Datasets are in", RAW_DIR)


if __name__ == "__main__":
    main()
