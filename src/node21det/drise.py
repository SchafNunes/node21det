"""D-RISE: mapas de saliência para detectores por mascaramento aleatório.

Petsiuk et al., "Black-box explanation of object detectors via saliency maps",
CVPR 2021. Para uma predição-alvo (caixa), a imagem é multiplicada por N máscaras
aleatórias suaves; o detector roda em cada versão; cada máscara recebe o peso
da semelhança entre o alvo e a melhor detecção que sobreviveu. O mapa é a média
das máscaras ponderada por esses pesos.

Semelhança: IoU(alvo, detecção) x escore da detecção, máximo sobre as detecções.
O artigo multiplica ainda a similaridade de cosseno entre os vetores de classe;
com uma classe só, esse termo é 1.

Todas as predições-alvo de uma mesma radiografia reaproveitam as mesmas
inferências mascaradas.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def generate_masks(n: int, grid: int, p: float, height: int, width: int, generator: torch.Generator,
                   device=torch.device("cpu")) -> torch.Tensor:
    """n máscaras (n, 1, H, W) em [0, 1]: grade binária grid x grid com probabilidade p de
    manter cada célula, ampliada bilinearmente e deslocada ao acaso dentro de uma célula."""
    cell_h, cell_w = math.ceil(height / grid), math.ceil(width / grid)
    up_h, up_w = (grid + 1) * cell_h, (grid + 1) * cell_w
    cells = (torch.rand(n, 1, grid, grid, generator=generator) < p).float()
    big = F.interpolate(cells.to(device), size=(up_h, up_w), mode="bilinear", align_corners=False)
    dy = torch.randint(0, cell_h, (n,), generator=generator)
    dx = torch.randint(0, cell_w, (n,), generator=generator)
    return torch.stack([big[i, :, dy[i]: dy[i] + height, dx[i]: dx[i] + width] for i in range(n)])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre uma caixa a (4,) e caixas b (k, 4)."""
    if len(b) == 0:
        return np.zeros(0)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def similarity(target: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> float:
    if len(boxes) == 0:
        return 0.0
    return float((box_iou(target, boxes) * scores).max())


@torch.no_grad()
def drise(detect, image: torch.Tensor, targets: list[np.ndarray], n_masks: int = 1000, grid: int = 16,
          p: float = 0.5, batch: int = 16, seed: int = 0, device=torch.device("cpu")) -> list[np.ndarray]:
    """detect: lista de imagens (1, H, W) -> lista de (boxes (k, 4), scores (k,)) em numpy.
    image: (1, H, W) em [0, 1]. Devolve um mapa (H, W) por alvo."""
    _, h, w = image.shape
    g = torch.Generator().manual_seed(seed)
    image = image.to(device)
    sal = [torch.zeros(h, w, device=device) for _ in targets]
    done = 0
    while done < n_masks:
        k = min(batch, n_masks - done)
        masks = generate_masks(k, grid, p, h, w, g, device)
        outs = detect([image * masks[i] for i in range(k)])
        for t, target in enumerate(targets):
            weights = torch.tensor([similarity(target, b, s) for b, s in outs], device=device, dtype=torch.float32)
            sal[t] += (weights.view(k, 1, 1) * masks[:, 0]).sum(0)
        done += k
    return [(s / (n_masks * p)).cpu().numpy() for s in sal]


def model_detector(model, device):
    """Adapta o Detector do projeto à interface de drise()."""
    def detect(images):
        out = model([i.to(device) for i in images])
        res = []
        for o in out:
            keep = o["labels"] == 1
            res.append((o["boxes"][keep].cpu().numpy(), o["scores"][keep].cpu().numpy()))
        return res
    return detect
