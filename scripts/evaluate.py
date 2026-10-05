"""Avalia o checkpoint de uma execução num conjunto de radiografias.

  # avaliação única no teste (etapa 3, modelo final)
  python scripts/evaluate.py --run RUNS/s3_final --checkpoint last.pt --split test \
      --images DATA/images --metadata DATA/metadata.csv --splits splits/splits.csv

  # conferir uma execução de seleção na validação
  python scripts/evaluate.py --run RUNS/s2_frcnn_clahe --split val ...

Grava em <run>/eval_<conjunto>/: metrics.json, froc.csv e predictions.csv.
O conjunto de teste deve ser avaliado uma única vez, com o modelo final (D26).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from node21det.data.metadata import boxes_by_image, load_metadata
from node21det.data.splits import select
from node21det.inference import evaluate_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="diretório da execução (com config.yaml e o checkpoint)")
    p.add_argument("--checkpoint", default="best.pt", help="best.pt (seleção) ou last.pt (folds e modelo final)")
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--fold", type=int, help="avalia um fold do conjunto de seleção em vez de uma partição")
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--force", action="store_true", help="sobrescreve uma avaliação existente")
    args = p.parse_args()
    if (args.split is None) == (args.fold is None):
        p.error("informe --split ou --fold")

    splits = pd.read_csv(args.splits)
    names = select(splits, split=args.split) if args.split else select(splits, folds=[args.fold])
    tag = args.split or f"fold{args.fold}"
    out = Path(args.run) / f"eval_{tag}"
    if (out / "metrics.json").exists() and not args.force:
        raise SystemExit(f"{out} já existe. O teste é avaliado uma única vez; use --force só se souber o motivo.")
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes = boxes_by_image(load_metadata(args.metadata))
    metrics, curve, preds = evaluate_run(args.run, args.images, boxes, names, args.checkpoint, device)
    metrics["set"] = tag

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    curve.to_csv(out / "froc.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2, default=float))
    print(f"gravado em {out}")


if __name__ == "__main__":
    main()
