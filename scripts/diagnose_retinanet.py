"""Diagnóstico curto de treino da RetinaNet (E10): a perda cai? passos são descartados?

Roda N iterações com e sem precisão mista, a partir da mesma inicialização, e
imprime a cada 10 iterações: perda, escala do GradScaler, passos descartados
(a escala diminuiu) e a norma do gradiente da camada final de classificação.

  python scripts/diagnose_retinanet.py --images /content/data/images \
      --metadata /content/data/metadata.csv --splits splits/splits.csv
"""

import argparse
import copy

import pandas as pd
import torch
from torch.utils.data import DataLoader

from node21det.data.dataset import NoduleDataset, collate
from node21det.data.metadata import boxes_by_image, load_metadata
from node21det.data.splits import select
from node21det.models import build_detector


def run(model, loader, device, amp, iters, lr):
    model = copy.deepcopy(model).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cls_w = model.net.head.classification_head.cls_logits.weight
    skipped, it = 0, 0
    print(f"\n=== precisão mista: {amp} ===", flush=True)
    while it < iters:
        for images, targets in loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = cls_w.grad.norm().item() if cls_w.grad is not None else float("nan")
            before = scaler.get_scale() if amp else 1.0
            scaler.step(opt)
            scaler.update()
            if amp and scaler.get_scale() < before:
                skipped += 1
            if it % 10 == 0:
                parts = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items())
                print(f"  iter {it:3d} {parts} | escala={scaler.get_scale() if amp else '-'} "
                      f"descartados={skipped} | grad cls={gnorm:.2e}", flush=True)
            it += 1
            if it >= iters:
                break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-3)
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    boxes = boxes_by_image(load_metadata(args.metadata))
    names = select(pd.read_csv(args.splits), split="train")
    loader = DataLoader(NoduleDataset(args.images, names, boxes), batch_size=12, shuffle=True,
                        num_workers=2, collate_fn=collate)
    model = build_detector("retinanet")
    for amp in (True, False):
        torch.manual_seed(0)
        run(model, loader, device, amp, args.iters, args.lr)


if __name__ == "__main__":
    main()
