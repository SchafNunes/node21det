"""Dataset de radiografias padronizadas do NODE21.

Adaptado de node21_detection_baseline/training_utils/dataset.py. Mantém a
normalização do baseline (divisão pelo máximo da imagem, faixa [0, 1]) e a
entrada com um canal.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


def read_image(path: str | Path) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(f"{path}: esperada uma radiografia, veio pilha de {img.shape[0]}")
        img = img[0]
    peak = img.max()
    return img / peak if peak > 0 else img


class NoduleDataset(Dataset):
    def __init__(self, image_dir, img_names: list[str], boxes: dict[str, np.ndarray], transforms=None,
                 enhance=None):
        """enhance: realce de contraste aplicado à radiografia antes do aumento de dados (treino e avaliação)."""
        self.image_dir = Path(image_dir)
        self.img_names = list(img_names)
        self.boxes = boxes
        self.transforms = transforms
        self.enhance = enhance

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        image = read_image(self.image_dir / name)
        if self.enhance is not None:
            image = self.enhance(image)
        image = torch.from_numpy(image).unsqueeze(0)
        boxes = torch.as_tensor(self.boxes[name], dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(idx),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def balanced_sampler(img_names: list[str], boxes: dict[str, np.ndarray]):
    """Sorteio com reposição e peso inverso à contagem de cada classe (positiva/negativa):
    em média metade de cada lote é de radiografias com nódulo. O número de amostras por
    época é o tamanho do conjunto, para manter a duração da época. Usa o gerador global
    do torch, cujo estado é salvo no checkpoint."""
    from torch.utils.data import WeightedRandomSampler

    positive = np.array([len(boxes[n]) > 0 for n in img_names])
    n_pos, n_neg = positive.sum(), (~positive).sum()
    weights = np.where(positive, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(img_names),
                                 replacement=True)
