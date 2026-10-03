"""Leitura do metadata.csv do NODE21.

Uma linha por caixa de referência. Imagens negativas aparecem com uma única
linha de label 0 e caixa (0, 0, 0, 0), que não é uma caixa.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["img_name", "x", "y", "width", "height", "label"]


def load_metadata(path: str | Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed")]
    missing = set(COLUMNS) - set(meta.columns)
    if missing:
        raise ValueError(f"metadata sem colunas {sorted(missing)}")
    meta = meta[COLUMNS].copy()

    mixed = meta.groupby("img_name")["label"].nunique()
    if (mixed > 1).any():
        raise ValueError(f"imagens com label 0 e 1 ao mesmo tempo: {list(mixed[mixed > 1].index)}")
    return meta


def image_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por imagem: img_name, n_nodules, positive."""
    positives = meta[meta["label"] == 1].groupby("img_name").size()
    names = sorted(meta["img_name"].unique())
    n = np.array([int(positives.get(name, 0)) for name in names])
    return pd.DataFrame({"img_name": names, "n_nodules": n, "positive": n > 0})


def boxes_by_image(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Caixas de referência por imagem, em (x1, y1, x2, y2) no espaço da imagem padronizada.

    Imagens negativas recebem um array (0, 4).
    """
    out = {name: np.zeros((0, 4), dtype=np.float32) for name in meta["img_name"].unique()}
    pos = meta[meta["label"] == 1]
    for name, rows in pos.groupby("img_name"):
        x1 = rows["x"].to_numpy(dtype=np.float32)
        y1 = rows["y"].to_numpy(dtype=np.float32)
        x2 = x1 + rows["width"].to_numpy(dtype=np.float32)
        y2 = y1 + rows["height"].to_numpy(dtype=np.float32)
        out[name] = np.stack([x1, y1, x2, y2], axis=1)
    return out
