"""Partições treino/validação/teste e folds do k-fold.

As unidades de alocação são grupos de imagens: cada grupo de duplicatas
aproximadas vai inteiro para uma partição. Imagens sem duplicata formam um
grupo sozinhas. A estratificação é pela presença de nódulo no grupo.

O arquivo de partições tem uma linha por imagem com as colunas
img_name, positive, group, split (train/val/test) e fold (0..k-1, ou -1 no teste).
Os folds dividem o conjunto de seleção, isto é, a união de train e val.
"""

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def _assign(unit_sizes: np.ndarray, targets: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Aloca unidades, na ordem dada, ao destino mais distante da sua cota."""
    filled = np.zeros(len(targets), dtype=float)
    dest = np.empty(len(unit_sizes), dtype=int)
    for u in order:
        k = int(np.argmin(filled / targets))
        dest[u] = k
        filled[k] += unit_sizes[u]
    return dest


def _stratified_assign(units: pd.DataFrame, fractions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dest = np.empty(len(units), dtype=int)
    for positive in (True, False):
        idx = np.flatnonzero(units["positive"].to_numpy() == positive)
        if len(idx) == 0:
            continue
        sizes = units["size"].to_numpy()[idx]
        targets = fractions * sizes.sum()
        order = rng.permutation(len(idx))
        dest[idx] = _assign(sizes, targets, order)
    return dest


def make_splits(
    images: pd.DataFrame,
    groups: dict[str, str] | None = None,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    k_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """images: colunas img_name e positive. groups: img_name -> id do grupo de duplicatas."""
    df = images[["img_name", "positive"]].copy()
    groups = groups or {}
    df["group"] = [groups.get(n, n) for n in df["img_name"]]

    units = df.groupby("group").agg(positive=("positive", "any"), size=("img_name", "size")).reset_index()
    rng = np.random.default_rng(seed)

    split_idx = _stratified_assign(units, np.asarray(fractions, dtype=float), rng)
    units["split"] = np.asarray(SPLITS)[split_idx]

    units["fold"] = -1
    pool = units["split"] != "test"
    pool_units = units[pool].reset_index(drop=True)
    fold_idx = _stratified_assign(pool_units, np.full(k_folds, 1.0 / k_folds), rng)
    units.loc[pool, "fold"] = fold_idx

    out = df.merge(units[["group", "split", "fold"]], on="group", how="left")
    return out.sort_values("img_name").reset_index(drop=True)


def select(splits: pd.DataFrame, *, split: str | None = None, folds=None, exclude_folds=None) -> list[str]:
    """Nomes das imagens de uma partição ou de um conjunto de folds."""
    mask = np.ones(len(splits), dtype=bool)
    if split is not None:
        mask &= splits["split"].to_numpy() == split
    if folds is not None:
        mask &= splits["fold"].isin(list(folds)).to_numpy()
    if exclude_folds is not None:
        mask &= splits["split"].ne("test").to_numpy() & ~splits["fold"].isin(list(exclude_folds)).to_numpy()
    return splits.loc[mask, "img_name"].tolist()
