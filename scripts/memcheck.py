"""Mede o pico de memória da GPU em alguns passos de treino, antes da fila.

Usa as radiografias do treino com mais nódulos, para que os lotes medidos
tenham o máximo de caixas de referência. Sai com código 1 se a memória
reservada passar de --max-frac da memória total da GPU.

  python scripts/memcheck.py --config configs/frcnn_clahe.yaml \
      --images /data/images --metadata /data/metadata.csv --splits splits/splits.csv
"""

import argparse
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from node21det.config import load_config
from node21det.data.dataset import NoduleDataset, collate
from node21det.data.enhancement import build_enhancement
from node21det.data.metadata import boxes_by_image, load_metadata
from node21det.data.splits import select
from node21det.data.transforms import build_transforms
from node21det.models import build_detector
from node21det.training import seed_everything, train_one_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--max-frac", type=float, default=0.9)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("memcheck exige GPU")
    device = torch.device("cuda")
    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)

    boxes = boxes_by_image(load_metadata(args.metadata))
    names = select(pd.read_csv(args.splits), split="train")
    names = sorted(names, key=lambda n: len(boxes.get(n, [])), reverse=True)[: args.batches * cfg.train.batch_size]

    t = cfg.train
    enhance = build_enhancement(cfg.enhancement, cfg.enhancement_params)
    ds = NoduleDataset(args.images, names, boxes, build_transforms(cfg.augmentation), enhance)
    loader = DataLoader(ds, batch_size=t.batch_size, shuffle=False, num_workers=t.num_workers, collate_fn=collate)

    m = cfg.model
    model = build_detector(m.arch, m.pretrained_backbone, m.trainable_backbone_layers, m.nms_threshold,
                           m.score_threshold, m.detections_per_image, m.min_size, m.max_size,
                           m.retinanet_loss_normalization, m.retinanet_head_norm).to(device)
    o = cfg.optim
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=o.lr, momentum=o.momentum, weight_decay=o.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=t.amp)

    torch.cuda.reset_peak_memory_stats(device)
    train_one_epoch(model, optimizer, loader, device, scaler, log_every=0, grad_clip_norm=o.grad_clip_norm)
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    ok = reserved <= args.max_frac * total
    print(f"{cfg.name}: pico alocado {allocated:.2f} GB | reservado {reserved:.2f} GB | "
          f"GPU {total:.1f} GB | limite {args.max_frac * total:.1f} GB | {'ok' if ok else 'ACIMA DO LIMITE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
