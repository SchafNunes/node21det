"""Aumento de dados compatível com caixas delimitadoras.

Inversão horizontal (propagação exata para a caixa) e transformações
fotométricas, que não alteram a geometria. Adaptado de
node21_detection_baseline/training_utils/transforms.py.
"""

import random

import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            width = image.shape[-1]
            image = image.flip(-1)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - target["boxes"][:, [2, 0]]
            target = {**target, "boxes": boxes}
        return image, target


class RandomBrightnessContrast:
    """img' = (img - média) * c + média + b, com c e b sorteados nos intervalos dados."""

    def __init__(self, p: float, brightness: float, contrast: float):
        self.p, self.brightness, self.contrast = p, brightness, contrast

    def __call__(self, image, target):
        if random.random() < self.p:
            b = random.uniform(-self.brightness, self.brightness)
            c = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = image.mean()
            image = ((image - mean) * c + mean + b).clamp(0, 1)
        return image, target


class RandomGaussianNoise:
    def __init__(self, p: float, std: float):
        self.p, self.std = p, std

    def __call__(self, image, target):
        if random.random() < self.p:
            image = (image + torch.randn_like(image) * self.std).clamp(0, 1)
        return image, target


def build_transforms(aug: dict | None):
    """aug: seção augmentation da configuração. None ou vazio = sem aumento (avaliação)."""
    if not aug:
        return None
    ts = []
    if aug.get("hflip_p", 0) > 0:
        ts.append(RandomHorizontalFlip(aug["hflip_p"]))
    if aug.get("brightness_contrast_p", 0) > 0:
        ts.append(RandomBrightnessContrast(aug["brightness_contrast_p"], aug["brightness"], aug["contrast"]))
    if aug.get("noise_p", 0) > 0:
        ts.append(RandomGaussianNoise(aug["noise_p"], aug["noise_std"]))
    return Compose(ts) if ts else None
