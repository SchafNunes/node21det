"""Realce de contraste: a variável da etapa 2 (cap. 3, `Realce de contraste`).

Todas as técnicas recebem a radiografia já normalizada em [0, 1] (saída de
read_image) e devolvem float32 em [0, 1], com a mesma forma. A intensidade é
quantizada em 4.096 níveis, a profundidade das imagens do NODE21 (12 bits).
Nenhuma técnica altera a geometria: as caixas de referência continuam válidas.

  none   sem realce
  he     equalização de histograma global
  clahe  CLAHE (OpenCV), clip limit e grade de tiles fixados na configuração
  rmshe  equalização recursiva separada pela média (Chen e Ramli, 2003)
"""

import cv2
import numpy as np

LEVELS = 4096


def _quantize(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * (LEVELS - 1)), 0, LEVELS - 1).astype(np.int64)


def _equalized_lut(hist: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    """Tabela de 4.096 níveis: cada intervalo [lo, hi] é equalizado dentro de si mesmo."""
    lut = np.arange(LEVELS, dtype=np.float64)
    for lo, hi in bounds:
        seg = hist[lo : hi + 1]
        total = seg.sum()
        if total:
            lut[lo : hi + 1] = lo + np.cumsum(seg) / total * (hi - lo)
    return lut / (LEVELS - 1)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    q = _quantize(image)
    hist = np.bincount(q.ravel(), minlength=LEVELS)
    return _equalized_lut(hist, [(0, LEVELS - 1)]).astype(np.float32)[q]


def rmshe(image: np.ndarray, recursion: int = 2) -> np.ndarray:
    """Divide o histograma recursivamente pela média de cada parte (2^r partes) e
    equaliza cada parte dentro do seu próprio intervalo, preservando o brilho médio."""
    q = _quantize(image)
    hist = np.bincount(q.ravel(), minlength=LEVELS)
    levels = np.arange(LEVELS)
    bounds = [(0, LEVELS - 1)]
    for _ in range(recursion):
        nxt = []
        for lo, hi in bounds:
            seg = hist[lo : hi + 1]
            if seg.sum() == 0 or hi - lo < 2:
                nxt.append((lo, hi))
                continue
            m = int(np.floor((levels[lo : hi + 1] * seg).sum() / seg.sum()))
            m = min(max(m, lo), hi - 1)
            nxt += [(lo, m), (m + 1, hi)]
        bounds = nxt
    return _equalized_lut(hist, bounds).astype(np.float32)[q]


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    q16 = (_quantize(image) * 16).astype(np.uint16)  # 12 bits na faixa de 16 bits do OpenCV
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return (op.apply(q16).astype(np.float32) / 65535.0).clip(0, 1)


def build_enhancement(name: str, params: dict | None = None):
    """Função imagem -> imagem, ou None para 'none'."""
    params = params or {}
    if name == "none":
        return None
    if name == "he":
        return histogram_equalization
    if name == "clahe":
        return lambda img: clahe(img, params.get("clip_limit", 2.0), params.get("tile_grid", 8))
    if name == "rmshe":
        return lambda img: rmshe(img, params.get("recursion", 2))
    raise ValueError(f"realce desconhecido: {name!r}; opções: none, he, clahe, rmshe")
