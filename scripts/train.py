"""Treina uma execução do protocolo.

Modos:
  select   treina em split=train, avalia em split=val a cada época (etapas 1 e 2)
  fold     treina no conjunto de seleção menos o fold --fold, por --epochs fixas,
           e avalia no fold deixado de fora ao final (etapa 3)
  final    treina no conjunto de seleção inteiro por --epochs fixas (etapa 3)

Exemplo:
  python scripts/train.py --config configs/frcnn_none.yaml --mode select \
      --images /data/images --metadata /data/metadata.csv \
      --splits splits/splits.csv --out runs/frcnn_none
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from node21det.config import load_config
from node21det.data.dataset import NoduleDataset, collate
from node21det.data.metadata import boxes_by_image, load_metadata
from node21det.data.splits import select
from node21det.data.transforms import build_transforms
from node21det.models import build_detector
from node21det.training import evaluate_loader, fit, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["select", "fold", "final"], default="select")
    p.add_argument("--fold", type=int)
    p.add_argument("--epochs", type=int, help="número fixo de épocas (modos fold e final)")
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, help="usa só as N primeiras imagens de cada conjunto (teste rápido)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg.enhancement != "none":
        raise NotImplementedError("realce de contraste ainda não implementado")
    if args.mode != "select" and not args.epochs:
        p.error("--epochs é obrigatório nos modos fold e final")
    if args.mode == "fold" and args.fold is None:
        p.error("--fold é obrigatório no modo fold")

    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    boxes = boxes_by_image(load_metadata(args.metadata))
    splits = pd.read_csv(args.splits)
    if args.mode == "select":
        train_names, val_names = select(splits, split="train"), select(splits, split="val")
    elif args.mode == "fold":
        train_names, val_names = select(splits, exclude_folds=[args.fold]), select(splits, folds=[args.fold])
    else:
        train_names, val_names = select(splits, exclude_folds=[]), []
    if args.limit:
        train_names, val_names = train_names[: args.limit], val_names[: args.limit]
    print(f"treino: {len(train_names)} imagens | avaliação: {len(val_names)} imagens")

    t = cfg.train
    train_ds = NoduleDataset(args.images, train_names, boxes, build_transforms(cfg.augmentation))
    train_loader = DataLoader(train_ds, batch_size=t.batch_size, shuffle=True, num_workers=t.num_workers,
                              collate_fn=collate, pin_memory=device.type == "cuda")
    eval_loader = None
    if val_names:
        eval_loader = DataLoader(NoduleDataset(args.images, val_names, boxes), batch_size=t.batch_size,
                                 shuffle=False, num_workers=t.num_workers, collate_fn=collate)

    m = cfg.model
    model = build_detector(m.arch, m.pretrained_backbone, m.trainable_backbone_layers, m.nms_threshold,
                           m.score_threshold, m.detections_per_image, m.min_size, m.max_size)

    if args.mode == "select":
        state = fit(cfg, model, train_loader, eval_loader, args.out, device)
        print(f"melhor época: {state['best_epoch']} (rank={state['best_metric']:.4f})")
    else:
        fit(cfg, model, train_loader, None, args.out, device, max_epochs=args.epochs)
        if args.mode == "fold":
            metrics = evaluate_loader(model, eval_loader, device)
            Path(args.out, "fold_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
