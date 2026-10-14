"""As configs das verificações de hiperparâmetros (D44, D46) só podem diferir da
configuração final na taxa, no decaimento de pesos e na semente. Com modelo,
lote, resolução e realce iguais, o uso de memória da GPU é o mesmo da
frcnn_clahe.yaml, medido nas execuções anteriores."""

from dataclasses import asdict
from pathlib import Path

import pytest

from node21det.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
VARIANTS = sorted(CONFIGS.glob("frcnn_clahe_lr*.yaml"))
ALLOWED = {("optim", "lr"), ("optim", "weight_decay"), ("train", "seed")}


def _flat(cfg) -> dict:
    d = asdict(cfg)
    d.pop("name")
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update({(k, kk): vv for kk, vv in v.items()})
        else:
            out[(k,)] = v
    return out


def test_variants_exist():
    assert len(VARIANTS) == 13


@pytest.mark.parametrize("path", VARIANTS, ids=lambda p: p.stem)
def test_variant_differs_only_in_allowed_fields(path):
    base, cfg = _flat(load_config(CONFIGS / "frcnn_clahe.yaml")), _flat(load_config(path))
    changed = {k for k in base if base[k] != cfg[k]}
    assert changed <= ALLOWED, changed
    assert load_config(path).name == path.stem
