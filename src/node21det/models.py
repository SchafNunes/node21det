"""Construção dos detectores comparados: Faster R-CNN e RetinaNet, ResNet-50 + FPN.

Os dois usam o mesmo pós-processamento: nenhum limiar mínimo de escore, NMS
interna a 0,3 (o limiar do baseline do NODE21) e um teto de detecções por imagem.

A classe nódulo é o rótulo 1 nos alvos e nas saídas dos dois. A Faster R-CNN
tem a classe fundo explícita (num_classes=2). A RetinaNet classifica por sigmoide
sem classe fundo, então é construída com uma classe só e o Detector desloca os
rótulos na entrada e na saída.
"""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn

ARCHS = ("faster_rcnn", "retinanet")


class Detector(nn.Module):
    def __init__(self, arch: str, net: nn.Module, label_offset: int):
        super().__init__()
        self.arch, self.net, self.label_offset = arch, net, label_offset

    def forward(self, images, targets=None):
        if targets is not None and self.label_offset:
            targets = [{**t, "labels": t["labels"] - self.label_offset} for t in targets]
        out = self.net(images, targets)
        if not self.training and self.label_offset:
            out = [{**o, "labels": o["labels"] + self.label_offset} for o in out]
        return out


def build_detector(
    arch: str,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    nms_threshold: float = 0.3,
    score_threshold: float = 0.0,
    detections_per_image: int = 100,
    min_size: int = 800,
    max_size: int = 1333,
) -> Detector:
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    common = dict(
        weights=None,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )
    if arch == "faster_rcnn":
        net = fasterrcnn_resnet50_fpn(
            num_classes=2,
            box_score_thresh=score_threshold,
            box_nms_thresh=nms_threshold,
            box_detections_per_img=detections_per_image,
            **common,
        )
        return Detector(arch, net, label_offset=0)
    if arch == "retinanet":
        net = retinanet_resnet50_fpn(
            num_classes=1,
            score_thresh=score_threshold,
            nms_thresh=nms_threshold,
            detections_per_img=detections_per_image,
            **common,
        )
        return Detector(arch, net, label_offset=1)
    raise ValueError(f"arquitetura desconhecida: {arch!r}; opções: {ARCHS}")


@torch.no_grad()
def predict(model: Detector, images: list[torch.Tensor]) -> list[dict]:
    model.eval()
    return model(images)
