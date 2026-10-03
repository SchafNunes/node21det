"""Métricas de avaliação no protocolo do NODE21.

Correspondência por nódulo: as predições de uma imagem são percorridas em ordem
decrescente de escore, e cada uma é verdadeiro positivo se tiver IoU > 0,2 com
uma caixa de referência ainda não atribuída (escolhe-se a de maior IoU). As
demais predições são falsos positivos, inclusive as duplicatas sobre um nódulo
já atribuído.

A leitura da FROC nas taxas de falso positivo por imagem usa interpolação
linear, como em node21-submit/src/utils/custom_metrics.py (solução MTEC), que
segue o código de avaliação do CAMELYON16.
"""

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLD = 0.2
FP_RATES = (0.125, 0.25, 0.5)
RANK_FP_RATE = 0.25


@dataclass
class Prediction:
    boxes: np.ndarray  # (n, 4) x1, y1, x2, y2
    scores: np.ndarray  # (n,)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_image(pred: Prediction, gt_boxes: np.ndarray, iou_threshold: float = IOU_THRESHOLD) -> np.ndarray:
    """Para cada predição (na ordem original), True se for verdadeiro positivo."""
    is_tp = np.zeros(len(pred.scores), dtype=bool)
    if len(pred.scores) == 0 or len(gt_boxes) == 0:
        return is_tp
    ious = iou_matrix(pred.boxes, gt_boxes)
    assigned = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-pred.scores, kind="stable"):
        candidates = np.where(~assigned & (ious[i] > iou_threshold), ious[i], -1.0)
        j = int(np.argmax(candidates))
        if candidates[j] > 0:
            assigned[j] = True
            is_tp[i] = True
    return is_tp


@dataclass
class FrocCurve:
    thresholds: np.ndarray  # escores, decrescentes
    fps_per_image: np.ndarray
    sensitivity: np.ndarray

    def sensitivity_at(self, fp_rate: float) -> float:
        x = np.concatenate([[0.0], self.fps_per_image])
        y = np.concatenate([[0.0], self.sensitivity])
        # várias sensibilidades com o mesmo FP/imagem: vale a maior (np.interp exige x crescente)
        x, last = np.unique(x[::-1], return_index=True)
        y = y[::-1][last]
        return float(np.interp(fp_rate, x, y))


def _pooled(preds: list[Prediction], gts: list[np.ndarray]):
    scores, tps = [], []
    for pred, gt in zip(preds, gts, strict=True):
        scores.append(np.asarray(pred.scores, dtype=float))
        tps.append(match_image(pred, gt))
    scores = np.concatenate(scores) if scores else np.zeros(0)
    tps = np.concatenate(tps) if tps else np.zeros(0, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    scores, tps = scores[order], tps[order]
    # um ponto de operação por escore distinto: o último índice de cada escore
    last = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1] if len(scores) else np.zeros(0, int)
    return scores[last], np.cumsum(tps)[last], np.cumsum(~tps)[last]


def froc(preds: list[Prediction], gts: list[np.ndarray]) -> FrocCurve:
    n_lesions = sum(len(g) for g in gts)
    thresholds, tp, fp = _pooled(preds, gts)
    sens = tp / n_lesions if n_lesions else np.full(len(tp), np.nan)
    return FrocCurve(thresholds, fp / len(preds), sens)


def average_precision(preds: list[Prediction], gts: list[np.ndarray]) -> float:
    """Precisão média sob IoU > 0,2, com envelope de precisão (todos os pontos)."""
    n_lesions = sum(len(g) for g in gts)
    _, tp, fp = _pooled(preds, gts)
    if n_lesions == 0 or len(tp) == 0:
        return 0.0
    recall = np.r_[0.0, tp / n_lesions]
    precision = np.r_[1.0, tp / (tp + fp)]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))


def image_scores(preds: list[Prediction]) -> np.ndarray:
    """Escore da imagem: o maior escore entre as predições, ou 0 se não houver."""
    return np.array([float(p.scores.max()) if len(p.scores) else 0.0 for p in preds])


def image_scores_outside_nodules(preds: list[Prediction], gts: list[np.ndarray]) -> np.ndarray:
    """Escore da imagem contando só predições que não tocam nenhuma caixa de referência.

    Diagnóstico de atalho: se a AUC com esse escore ficar bem acima de 0,5, o
    modelo separa positivas de negativas por algo fora dos nódulos.
    """
    out = []
    for p, g in zip(preds, gts, strict=True):
        keep = np.ones(len(p.scores), dtype=bool)
        if len(g) and len(p.scores):
            keep = iou_matrix(p.boxes, g).max(axis=1) == 0
        out.append(float(p.scores[keep].max()) if keep.any() else 0.0)
    return np.array(out)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC por Mann-Whitney, empates contam meio."""
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def evaluate(preds: list[Prediction], gts: list[np.ndarray]) -> dict:
    curve = froc(preds, gts)
    positive = np.array([len(g) > 0 for g in gts])
    auc = roc_auc(image_scores(preds), positive)
    sens = {f"sens@{r:g}": curve.sensitivity_at(r) for r in FP_RATES}
    s = curve.sensitivity_at(RANK_FP_RATE)
    return {
        "mean_sens": float(np.mean(list(sens.values()))),
        "rank": 0.75 * auc + 0.25 * s,
        "auc": auc,
        **sens,
        "ap@0.2": average_precision(preds, gts),
        "auc_outside_nodules": roc_auc(image_scores_outside_nodules(preds, gts), positive),
        "n_images": len(preds),
        "n_lesions": sum(len(g) for g in gts),
    }
