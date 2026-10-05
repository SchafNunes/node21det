"""Avaliação de um checkpoint sobre um conjunto de radiografias.

Usa a configuração gravada no diretório da execução (config.yaml), para aplicar
o mesmo realce e o mesmo pós-processamento do treino. Devolve as métricas, a
curva FROC e uma tabela com todas as predições.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data.dataset import NoduleDataset, collate
from .data.enhancement import build_enhancement
from .evaluation import evaluate, froc, match_image
from .models import build_detector
from .training import predict_loader


def load_run_model(run_dir, checkpoint: str = "best.pt", device=torch.device("cpu")):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    m = cfg.model
    model = build_detector(m.arch, False, m.trainable_backbone_layers, m.nms_threshold, m.score_threshold,
                           m.detections_per_image, m.min_size, m.max_size,
                           m.retinanet_loss_normalization, m.retinanet_head_norm)
    ckpt = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return cfg, model.to(device).eval(), ckpt.get("epoch", ckpt.get("state", {}).get("epoch"))


def evaluate_run(run_dir, images_dir, boxes: dict, names: list[str], checkpoint: str = "best.pt",
                 device=torch.device("cpu"), batch_size: int = 12, num_workers: int = 2):
    cfg, model, epoch = load_run_model(run_dir, checkpoint, device)
    enhance = build_enhancement(cfg.enhancement, cfg.enhancement_params)
    ds = NoduleDataset(images_dir, names, boxes, enhance=enhance)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    preds = predict_loader(model, loader, device)
    gts = [boxes[n] for n in names]

    metrics = evaluate(preds, gts)
    metrics.update(run=cfg.name, checkpoint=checkpoint, epoch=epoch, enhancement=cfg.enhancement)

    curve = froc(preds, gts)
    curve_df = pd.DataFrame({"threshold": curve.thresholds, "fps_per_image": curve.fps_per_image,
                             "sensitivity": curve.sensitivity})

    rows = []
    for name, p, g in zip(names, preds, gts):
        tp = match_image(p, g)
        for (x1, y1, x2, y2), s, t in zip(p.boxes, p.scores, tp):
            rows.append((name, len(g) > 0, float(x1), float(y1), float(x2), float(y2), float(s), bool(t)))
    pred_df = pd.DataFrame(rows, columns=["img_name", "image_positive", "x1", "y1", "x2", "y2", "score", "is_tp"])
    return metrics, curve_df, pred_df
