"""Loop de treino com retomada, precisão mista e interrupção antecipada.

No diretório de saída de uma execução:
  last.pt      estado completo ao fim da última época (retomada)
  best.pt      pesos da época com melhor métrica de seleção na validação
  history.csv  uma linha por época
  config.yaml  configuração usada

Se last.pt existe, fit() continua da época seguinte.
"""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml

from .config import RunConfig
from .evaluation import Prediction, evaluate

# Sensibilidade média da FROC em 1/8, 1/4 e 1/2 FP por imagem. O rank do desafio
# continua calculado e reportado, mas não seleciona: a AUC por imagem, 75% dele,
# é afetada por um atalho de origem das imagens (Tcc/EXPERIMENTOS.md, E7).
SELECTION_METRIC = "mean_sens"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(obj, path: Path):
    """Grava em arquivo temporário e renomeia: uma queda no meio não corrompe o checkpoint anterior."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one_epoch(model, optimizer, loader, device, scaler, warmup=None, log_every=50):
    model.train()
    totals, n = {}, 0
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for it, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"perda não finita na iteração {it}: { {k: v.item() for k, v in losses.items()} }")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if warmup is not None:
            warmup.step()
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        totals["loss"] = totals.get("loss", 0.0) + loss.item()
        n += 1
        if log_every and it % log_every == 0:
            print(f"  iter {it}/{len(loader)} loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
    out = {f"train_{k}": v / max(n, 1) for k, v in totals.items()}
    out["train_seconds"] = time.time() - start
    if device.type == "cuda":
        out["gpu_peak_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    return out


@torch.no_grad()
def predict_loader(model, loader, device) -> list[Prediction]:
    model.eval()
    preds = []
    for images, _ in loader:
        for o in model([img.to(device) for img in images]):
            keep = o["labels"] == 1
            preds.append(Prediction(o["boxes"][keep].cpu().numpy(), o["scores"][keep].cpu().numpy()))
    return preds


def evaluate_loader(model, loader, device) -> dict:
    start = time.time()
    preds = predict_loader(model, loader, device)
    gts = [loader.dataset.boxes[name] for name in loader.dataset.img_names]
    metrics = evaluate(preds, gts)
    metrics["eval_seconds"] = time.time() - start
    return metrics


def _warmup(optimizer, iters: int, factor: float):
    def f(i):
        return 1.0 if i >= iters else factor * (1 - i / iters) + i / iters

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def fit(cfg: RunConfig, model, train_loader, val_loader, out_dir, device, max_epochs: int | None = None):
    """Treina até max_epochs (padrão: cfg.train.max_epochs).

    Com val_loader, avalia a cada época, guarda best.pt pela métrica de seleção
    e aplica a interrupção antecipada. Sem val_loader (folds e modelo final),
    treina o número fixo de épocas.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    max_epochs = max_epochs or cfg.train.max_epochs

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    o = cfg.optim
    optimizer = torch.optim.SGD(params, lr=o.lr, momentum=o.momentum, weight_decay=o.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=o.lr_step_epochs, gamma=o.lr_gamma)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.train.amp and device.type == "cuda")

    state = {"epoch": -1, "best_metric": -math.inf, "best_epoch": -1, "bad_epochs": 0, "stopped": False, "history": []}
    last = out_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"])
        state = ckpt["state"]
        print(f"retomando após a época {state['epoch']}", flush=True)

    patience = cfg.train.early_stopping_patience
    for epoch in range(state["epoch"] + 1, max_epochs):
        if state["stopped"]:
            break
        warmup = None
        if epoch == 0:
            warmup = _warmup(optimizer, max(1, min(1000, len(train_loader) - 1)), o.warmup_factor)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update(train_one_epoch(model, optimizer, train_loader, device, scaler, warmup))
        lr_scheduler.step()

        if val_loader is not None:
            metrics = evaluate_loader(model, val_loader, device)
            row.update({f"val_{k}": v for k, v in metrics.items()})
            value = metrics[SELECTION_METRIC]
            if not math.isfinite(value):
                raise ValueError(f"métrica de seleção não finita na época {epoch}: {metrics}")
            if value > state["best_metric"]:
                state.update(best_metric=value, best_epoch=epoch, bad_epochs=0)
                _atomic_save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "best.pt")
            else:
                state["bad_epochs"] += 1
            state["stopped"] = state["bad_epochs"] >= patience

        state["epoch"] = epoch
        state["history"].append(row)
        pd.DataFrame(state["history"]).to_csv(out_dir / "history.csv", index=False)
        _atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "state": state,
                "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
            },
            last,
        )
        msg = f"[época {epoch}] loss={row['train_loss']:.4f} ({row['train_seconds']/60:.1f} min)"
        if val_loader is not None:
            msg += (f" val mean_sens={row['val_mean_sens']:.4f} sens@0.25={row['val_sens@0.25']:.4f}"
                    f" rank={row['val_rank']:.4f} auc={row['val_auc']:.4f} auc_fora={row['val_auc_outside_nodules']:.4f}")
        print(msg, flush=True)
    return state
