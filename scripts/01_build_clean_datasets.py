"""Read every raw spreadsheet and write tidy CSVs to data/processed/.

Run this first: the other scripts load the raw files through the same loaders,
but the processed copies are what you want for inspection in Excel or for
sharing without the original workbook formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from durable_dominance import datasets as D

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, loader in D.LOADERS.items():
        df = loader()
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        summary.append({
            "dataset": name,
            "rows": len(df),
            "seasons": int(df["season"].nunique()),
            "first": df["season"].min(),
            "last": df["season"].max(),
            "columns": len(df.columns),
        })
        print(f"wrote {path.relative_to(OUT.parents[1])}  ({len(df):,} rows)")

    index = pd.DataFrame(summary)
    index.to_csv(OUT / "_index.csv", index=False)
    print("\n" + index.to_string(index=False))


if __name__ == "__main__":
    main()
