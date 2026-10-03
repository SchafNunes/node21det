"""Verificação de duplicatas aproximadas por correlação entre miniaturas.

Cada radiografia vira uma miniatura 64 x 64, padronizada para média 0 e
desvio 1. A correlação de Pearson é calculada entre todos os pares. Pares acima
do limiar são unidos em grupos (fecho transitivo), que make_splits mantém numa
partição só.

A pHash de 64 bits foi descartada: em radiografias de tórax, que têm todas a
mesma silhueta, ela aproximava milhares de pares de pacientes diferentes.
"""

import numpy as np
from scipy.ndimage import zoom

THUMB_SIZE = 64


def thumbnail(image: np.ndarray, size: int = THUMB_SIZE) -> np.ndarray:
    return zoom(image.astype(np.float32), (size / image.shape[0], size / image.shape[1]), order=1)


def correlation_matrix(thumbs: np.ndarray) -> np.ndarray:
    v = thumbs.reshape(len(thumbs), -1).astype(np.float64)
    v = (v - v.mean(1, keepdims=True)) / v.std(1, keepdims=True)
    return (v @ v.T) / v.shape[1]


def near_duplicate_pairs(names: list[str], thumbs: np.ndarray, min_corr: float) -> list[tuple[str, str, float]]:
    corr = correlation_matrix(thumbs)
    i, j = np.triu_indices(len(names), 1)
    hit = corr[i, j] >= min_corr
    return [(names[a], names[b], float(corr[a, b])) for a, b in zip(i[hit], j[hit])]


def top_pairs(names: list[str], thumbs: np.ndarray, n: int) -> list[tuple[str, str, float]]:
    """Os n pares mais correlacionados, para inspeção visual do limiar."""
    corr = correlation_matrix(thumbs)
    i, j = np.triu_indices(len(names), 1)
    order = np.argsort(-corr[i, j])[:n]
    return [(names[i[k]], names[j[k]], float(corr[i[k], j[k]])) for k in order]


def group_pairs(names: list[str], pairs: list[tuple[str, str, float]]) -> dict[str, str]:
    """Union-find: img_name -> id do grupo (o menor nome do grupo)."""
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {n: find(n) for n in names}
