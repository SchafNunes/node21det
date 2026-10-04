"""Diagnóstico curto de treino da RetinaNet (E10).

Roda variantes a partir da mesma inicialização e imprime, a cada 50 iterações,
as perdas médias do trecho, a escala do GradScaler, os passos descartados e a
norma do gradiente da camada final de classificação. Aquecimento linear nas
primeiras 100 iterações, como no treino.

Variantes:
  A  taxa 5e-3, todas as radiografias (o treino atual)
  B  taxa 1e-2, todas as radiografias (a taxa é baixa?)
  C  taxa 5e-3, só radiografias positivas (as negativas abafam o sinal?)
  D  como A, com recorte da norma do gradiente em 10
  E  como A, com GroupNorm nas torres da cabeça

Ao fim de cada variante mede, num lote fixo de radiografias positivas da
validação: a fração de ativações zeradas na saída da torre de classificação e o
desvio-padrão dos logits entre âncoras. Torre morta = fração perto de 1 e
desvio perto de 0.

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

# nome, taxa, só positivas, recorte do gradiente, norma da cabeça
VARIANTS = [
    ("A", 5e-3, False, None, "none"),
    ("B", 1e-2, False, None, "none"),
    ("C", 5e-3, True, None, "none"),
    ("D", 5e-3, False, 10.0, "none"),
    ("E", 5e-3, False, None, "group"),
]


@torch.no_grad()
def tower_stats(model, images, device):
    """Fração de ativações zeradas na saída da torre de classificação e desvio dos logits."""
    model.eval()
    net = model.net
    out = {}
    h1 = net.head.classification_head.conv.register_forward_hook(lambda m, i, o: out.setdefault("tower", []).append(o))
    h2 = net.head.classification_head.cls_logits.register_forward_hook(lambda m, i, o: out.setdefault("logits", []).append(o))
    model([i.to(device) for i in images])
    h1.remove(); h2.remove()
    zero = sum((t == 0).sum().item() for t in out["tower"]) / sum(t.numel() for t in out["tower"])
    logits = torch.cat([l.flatten() for l in out["logits"]])
    return zero, logits.std().item(), torch.sigmoid(logits).max().item()


def run(model, loader, device, iters, lr, amp=True, warmup=100, clip=None, probe=None):
    model = copy.deepcopy(model).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, 0.001 + i / warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cls_w = model.net.head.classification_head.cls_logits.weight
    skipped, it, acc = 0, 0, {}
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
            gnorm = cls_w.grad.norm().item()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, clip)
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            sched.step()
            skipped += int(amp and scaler.get_scale() < before)
            for k, v in losses.items():
                acc[k] = acc.get(k, 0.0) + v.item()
            it += 1
            if it % 50 == 0:
                parts = " ".join(f"{k}={v / 50:.3f}" for k, v in acc.items())
                print(f"  iter {it:4d} média {parts} | lr={opt.param_groups[0]['lr']:.1e} "
                      f"descartados={skipped} | grad cls={gnorm:.2e}", flush=True)
                acc = {}
            if it >= iters:
                break
    if probe is not None:
        zero, std, pmax = tower_stats(model, probe, device)
        print(f"  torre de classificação: {zero:.1%} ativações zeradas | desvio dos logits {std:.3f} | "
              f"maior probabilidade {pmax:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--variants", default="ADE")
    args = p.parse_args()

    device = torch.device("cuda")
    boxes = boxes_by_image(load_metadata(args.metadata))
    splits = pd.read_csv(args.splits)
    all_names = select(splits, split="train")
    pos_names = [n for n in all_names if len(boxes[n])]
    val_pos = [n for n in select(splits, split="val") if len(boxes[n])][:8]
    probe_ds = NoduleDataset(args.images, val_pos, boxes)
    probe = [probe_ds[i][0] for i in range(len(probe_ds))]
    for name, lr, positives_only, clip, head_norm in VARIANTS:
        if name not in args.variants:
            continue
        names = pos_names if positives_only else all_names
        torch.manual_seed(0)
        model = build_detector("retinanet", retinanet_head_norm=head_norm)
        loader = DataLoader(NoduleDataset(args.images, names, boxes), batch_size=12, shuffle=True,
                            num_workers=2, collate_fn=collate)
        print(f"\n=== {name}: taxa {lr:g}, {'só positivas' if positives_only else 'todas'} ({len(names)} imagens), "
              f"recorte {clip}, cabeça {head_norm} ===", flush=True)
        zero, std, pmax = tower_stats(model.to(device), probe, device)
        print(f"  antes do treino: {zero:.1%} ativações zeradas | desvio dos logits {std:.3f} | "
              f"maior probabilidade {pmax:.3f}", flush=True)
        run(model, loader, device, args.iters, lr, clip=clip, probe=probe)


if __name__ == "__main__":
    main()
