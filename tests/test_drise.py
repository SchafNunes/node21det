import numpy as np
import torch

from node21det.drise import box_iou, drise, generate_masks, similarity


def test_masks_shape_range_and_keep_rate():
    g = torch.Generator().manual_seed(0)
    m = generate_masks(200, grid=8, p=0.5, height=64, width=80, generator=g)
    assert m.shape == (200, 1, 64, 80)
    assert m.min() >= 0 and m.max() <= 1
    assert abs(m.mean().item() - 0.5) < 0.05


def test_similarity_is_iou_times_best_score():
    t = np.array([0, 0, 10, 10.0])
    boxes = np.array([[0, 0, 10, 10.0], [0, 0, 5, 10.0]])
    assert similarity(t, boxes, np.array([0.2, 0.9])) == np.float64(max(1 * 0.2, 0.5 * 0.9))
    assert similarity(t, np.zeros((0, 4)), np.zeros(0)) == 0.0
    assert box_iou(t, boxes).tolist() == [1.0, 0.5]


def test_saliency_concentrates_on_the_evidence_region():
    """Detector de brinquedo: devolve a caixa-alvo com escore = brilho médio de uma região à
    direita. O mapa deve ser maior nessa região que no resto da imagem."""
    h, w = 64, 64
    image = torch.ones(1, h, w)
    evidence = (slice(20, 44), slice(40, 60))
    target = np.array([5.0, 5.0, 20.0, 20.0])  # a caixa fica longe da evidência, de propósito

    def detect(images):
        return [(target[None], np.array([float(im[0][evidence].mean())])) for im in images]

    sal = drise(detect, image, [target], n_masks=400, grid=8, p=0.5, batch=50, seed=1)[0]
    # o mapa é uma diferença de médias (contraste baixo, normalizado para exibir):
    # a região de evidência deve ser a mais saliente entre blocos do mesmo tamanho
    block = lambda y, x: sal[y:y + 24, x:x + 20].mean()
    scores = {(y, x): block(y, x) for y in range(0, h - 24 + 1, 4) for x in range(0, w - 20 + 1, 4)}
    best = max(scores, key=scores.get)
    assert abs(best[0] - 20) <= 4 and abs(best[1] - 40) <= 4
    assert sal[evidence].mean() > sal[:, :20].mean()
