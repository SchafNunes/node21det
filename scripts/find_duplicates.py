"""Verificação de duplicatas aproximadas sobre o conjunto completo. Roda em CPU (~3 min).

  python scripts/find_duplicates.py --images data/images --metadata data/metadata.csv \
      --min-corr 0.99 --out-dir splits/

Grava dup_pairs.csv (pares acima do limiar), dup_top_pairs.csv (os 200 pares
mais correlacionados, para inspecionar o vão abaixo do limiar) e dup_groups.csv
(entrada de make_splits).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from node21det.data.dataset import read_image
from node21det.data.dedup import group_pairs, near_duplicate_pairs, thumbnail, top_pairs
from node21det.data.metadata import load_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--min-corr", type=float, default=0.99)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    names = sorted(load_metadata(args.metadata)["img_name"].unique())
    thumbs = []
    for i, name in enumerate(names):
        thumbs.append(thumbnail(read_image(Path(args.images) / name)))
        if i % 500 == 0:
            print(f"{i}/{len(names)}", flush=True)
    thumbs = np.stack(thumbs)

    pairs = near_duplicate_pairs(names, thumbs, args.min_corr)
    groups = group_pairs(names, pairs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["a", "b", "corr"]
    pd.DataFrame(pairs, columns=cols).to_csv(out / "dup_pairs.csv", index=False)
    pd.DataFrame(top_pairs(names, thumbs, 200), columns=cols).to_csv(out / "dup_top_pairs.csv", index=False)
    pd.DataFrame({"img_name": names, "group": [groups[n] for n in names]}).to_csv(out / "dup_groups.csv", index=False)
    sizes = pd.Series(groups).value_counts()
    print(f"{len(pairs)} pares com correlação >= {args.min_corr}; {int((sizes > 1).sum())} grupos com mais de uma imagem")
    for a, b, c in pairs:
        print(f"  {a} {b} {c:.4f}")


if __name__ == "__main__":
    main()
