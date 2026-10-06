"""Mapas D-RISE do modelo final sobre casos do teste selecionados por tipo (E19).

Casos escolhidos automaticamente de <run>/eval_test/predictions.csv:
  acerto        2 verdadeiros positivos de maior escore
  fp_positiva   2 falsos positivos de maior escore em radiografias positivas
  fp_negativa   2 falsos positivos de maior escore em radiografias negativas
  borda         1 predição degenerada (< 2 px) de maior escore
  falso_neg     1 nódulo sem predição correta acima do escore de corte de 1/2 FP
                por imagem; o alvo é a própria caixa de referência

Rodar com GPU (cada caso exige N inferências):
  python scripts/drise.py --run RUNS/s3_final --images /content/data/images \
      --metadata /content/data/metadata.csv --splits splits/splits.csv --masks 1000

Grava em <run>/drise/: cases.csv, um .npy por caso e drise.png (todos os casos).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from node21det.data.dataset import read_image
from node21det.data.enhancement import build_enhancement
from node21det.data.metadata import boxes_by_image, load_metadata
from node21det.drise import drise, model_detector
from node21det.evaluation import iou_matrix
from node21det.inference import load_run_model

OVERLAY = np.array([0xEB, 0x68, 0x34]) / 255  # slot 2 da paleta de referência
TARGET_COLOR, GT_COLOR = "#2a78d6", "#1baf7a"
TITLES = {"acerto": "Verdadeiro positivo", "fp_positiva": "Falso positivo (radiografia positiva)",
          "fp_negativa": "Falso positivo (radiografia negativa)", "borda": "Predição degenerada na borda",
          "falso_neg": "Falso negativo (alvo: referência)"}


def choose_cases(pred: pd.DataFrame, boxes: dict) -> pd.DataFrame:
    pred = pred.copy()
    pred["degenerate"] = ((pred.x2 - pred.x1) < 2) | ((pred.y2 - pred.y1) < 2)
    rows = []

    def take(df, kind, n):
        for _, r in df.sort_values("score", ascending=False).drop_duplicates("img_name").head(n).iterrows():
            rows.append(dict(kind=kind, img_name=r.img_name, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, score=r.score))

    take(pred[pred.is_tp], "acerto", 2)
    take(pred[~pred.is_tp & pred.image_positive & ~pred.degenerate], "fp_positiva", 2)
    take(pred[~pred.is_tp & ~pred.image_positive & ~pred.degenerate], "fp_negativa", 2)
    take(pred[pred.degenerate], "borda", 1)

    # falso negativo: referência sem predição com IoU > 0,2 acima do corte de 1/2 FP por imagem
    cut = 0.086  # E17: escore de corte a 1/2 FP por imagem no teste
    missed = []
    for name, g in pred[pred.image_positive & (pred.score >= cut)].groupby("img_name"):
        ious = iou_matrix(boxes[name], g[["x1", "y1", "x2", "y2"]].to_numpy())
        for i, gt in enumerate(boxes[name]):
            if ious.shape[1] == 0 or ious[i].max() <= 0.2:
                missed.append((name, *gt, (gt[2] - gt[0]) * (gt[3] - gt[1])))
    positives_with_no_pred = set(n for n, b in boxes.items() if len(b)) & set(pred.img_name) - set(
        pred[pred.score >= cut].img_name)
    for name in positives_with_no_pred:
        for gt in boxes[name]:
            missed.append((name, *gt, (gt[2] - gt[0]) * (gt[3] - gt[1])))
    if missed:
        name, x1, y1, x2, y2, _ = max(missed, key=lambda m: m[-1])  # o maior nódulo perdido
        rows.append(dict(kind="falso_neg", img_name=name, x1=x1, y1=y1, x2=x2, y2=y2, score=np.nan))
    return pd.DataFrame(rows)


def panel(ax, image, sal, case, gts, half=192):
    h, w = image.shape
    cx, cy = (case.x1 + case.x2) / 2, (case.y1 + case.y2) / 2
    x0 = int(np.clip(cx - half, 0, w - 2 * half)); y0 = int(np.clip(cy - half, 0, h - 2 * half))
    crop = image[y0:y0 + 2 * half, x0:x0 + 2 * half]
    s = sal[y0:y0 + 2 * half, x0:x0 + 2 * half]
    s = (s - sal.min()) / max(sal.max() - sal.min(), 1e-9)  # normalizado na imagem inteira
    rgba = np.zeros(s.shape + (4,)); rgba[..., :3] = OVERLAY; rgba[..., 3] = 0.75 * np.clip(s, 0, 1) ** 2
    ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
    ax.imshow(rgba)
    for gx1, gy1, gx2, gy2 in gts:
        ax.add_patch(plt.Rectangle((gx1 - x0, gy1 - y0), gx2 - gx1, gy2 - gy1, fill=False, ec=GT_COLOR, lw=1.5))
    ax.add_patch(plt.Rectangle((case.x1 - x0, case.y1 - y0), max(case.x2 - case.x1, 2), max(case.y2 - case.y1, 2),
                               fill=False, ec=TARGET_COLOR, lw=1.8, ls=(0, (3, 2))))
    score = "" if np.isnan(case.score) else f", escore {case.score:.2f}".replace(".", ",")
    ax.set_title(f"{TITLES[case.kind]}\n{case.img_name}{score}", fontsize=7, color="#0b0b0b")
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="last.pt")
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--masks", type=int, default=1000)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run = Path(args.run)
    out = run / "drise"
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes = boxes_by_image(load_metadata(args.metadata))
    pred = pd.read_csv(run / "eval_test" / "predictions.csv")
    cases = choose_cases(pred, boxes)
    cases.to_csv(out / "cases.csv", index=False)
    print(cases.to_string(index=False))

    cfg, model, _ = load_run_model(run, args.checkpoint, device)
    enhance = build_enhancement(cfg.enhancement, cfg.enhancement_params)
    detect = model_detector(model, device)

    maps = {}
    for name, group in cases.groupby("img_name", sort=False):
        image = read_image(Path(args.images) / name)
        if enhance is not None:
            image = enhance(image)
        targets = [g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) for _, g in group.iterrows()]
        sals = drise(detect, torch.from_numpy(image).unsqueeze(0), targets, n_masks=args.masks, grid=args.grid,
                     batch=args.batch, seed=args.seed, device=device)
        for (idx, _), sal in zip(group.iterrows(), sals):
            np.save(out / f"case{idx}_{name.replace('.mha', '')}.npy", sal)
            maps[idx] = (image, sal)
        print(f"{name}: {len(targets)} alvo(s) concluído(s)", flush=True)

    n = len(cases)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 2.6, int(np.ceil(n / cols)) * 2.9), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for i, (_, case) in enumerate(cases.iterrows()):
        image, sal = maps[i]
        panel(axes[i], image, sal, case, boxes[case.img_name])
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "drise.png")
    print(f"gravado em {out}")


if __name__ == "__main__":
    main()
