"""Verificação de duplicatas aproximadas sobre o conjunto completo. Roda em CPU.

  python scripts/find_duplicates.py --images /data/images --metadata metadata.csv \
      --max-distance 6 --out-dir splits/

Grava dup_pairs.csv (para inspeção visual) e dup_groups.csv (entrada de make_splits).
"""

import argparse
from pathlib import Path

import pandas as pd

from node21det.data.dataset import read_image
from node21det.data.dedup import group_pairs, near_duplicate_pairs, phash
from node21det.data.metadata import load_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--max-distance", type=int, default=6)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    names = sorted(load_metadata(args.metadata)["img_name"].unique())
    hashes = []
    for i, name in enumerate(names):
        hashes.append(phash(read_image(Path(args.images) / name)))
        if i % 500 == 0:
            print(f"{i}/{len(names)}", flush=True)

    pairs = near_duplicate_pairs(names, hashes, args.max_distance)
    groups = group_pairs(names, pairs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs, columns=["a", "b", "distance"]).to_csv(out / "dup_pairs.csv", index=False)
    pd.DataFrame({"img_name": names, "group": [groups[n] for n in names]}).to_csv(out / "dup_groups.csv", index=False)
    sizes = pd.Series(groups).value_counts()
    print(f"{len(pairs)} pares; {int((sizes > 1).sum())} grupos com mais de uma imagem")


if __name__ == "__main__":
    main()
