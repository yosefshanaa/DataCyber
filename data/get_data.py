"""Deterministically fetch the NSL-KDD dataset into ``data/raw/``.

Usage:
    python data/get_data.py

Downloads the canonical KDDTrain+ / KDDTest+ partitions so the notebook is fully
reproducible from a clean clone. The dataset is public (Canadian Institute for
Cybersecurity); we pull from a stable GitHub mirror.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
FILES = ["KDDTrain+.txt", "KDDTest+.txt", "KDDTrain+_20Percent.txt"]
RAW_DIR = Path(__file__).resolve().parent / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dest = RAW_DIR / fname
        if dest.exists():
            print(f"[skip] {fname} already present")
            continue
        url = f"{MIRROR}/{urllib.parse.quote(fname)}"
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"       -> {dest} ({dest.stat().st_size:,} bytes)")
    print("Done. NSL-KDD is in", RAW_DIR)


if __name__ == "__main__":
    main()
