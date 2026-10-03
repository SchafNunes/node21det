"""Gera o arquivo de partições (treino/validação/teste + folds) uma única vez.

  python scripts/make_splits.py --metadata metadata.csv --groups splits/dup_groups.csv \
      --out splits/splits.csv

--groups é a saída de scripts/find_duplicates.py. Sem ele, cada imagem é um grupo.
"""

import argparse
from pathlib import Path

import pandas as pd

from node21det.data.metadata import image_table, load_metadata
from node21det.data.splits import make_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--groups")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-folds", type=int, default=3)
    args = p.parse_args()

    images = image_table(load_metadata(args.metadata))
    groups = None
    if args.groups:
        g = pd.read_csv(args.groups)
        groups = dict(zip(g["img_name"], g["group"]))
    splits = make_splits(images, groups, k_folds=args.k_folds, seed=args.seed)

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} já existe; as partições são geradas uma única vez")
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)

    summary = splits.groupby("split").agg(images=("img_name", "size"), positive=("positive", "sum"))
    summary["positive_frac"] = summary["positive"] / summary["images"]
    print(summary.to_string())
    pool = splits[splits["split"] != "test"]
    print(pool.groupby("fold").agg(images=("img_name", "size"), positive=("positive", "sum")).to_string())


if __name__ == "__main__":
    main()
