"""Construção dos detectores comparados: Faster R-CNN e RetinaNet, ResNet-50 + FPN.

Os dois usam o mesmo pós-processamento: nenhum limiar mínimo de escore, NMS
interna a 0,3 (o limiar do baseline do NODE21) e um teto de detecções por imagem.

A classe nódulo é o rótulo 1 nos alvos e nas saídas dos dois. A Faster R-CNN
tem a classe fundo explícita (num_classes=2). A RetinaNet classifica por sigmoide
sem classe fundo, então é construída com uma classe só e o Detector desloca os
rótulos na entrada e na saída.

Normalização das perdas da RetinaNet. O torchvision divide a perda de cada
imagem pelo número de âncoras positivas daquela imagem e tira a média sobre o
lote. Numa radiografia negativa não há âncora positiva: a perda focal de todas
as âncoras de fundo é dividida por 1, e as negativas (77% do treino) dominam o
gradiente. Com essa normalização a RetinaNet não aprendeu a detectar
(Tcc/EXPERIMENTOS.md, E9). A receita de treino do torchvision para o COCO
descarta imagens sem anotação, e por isso a normalização por imagem não causa
problema lá. Aqui as perdas são somadas sobre o lote e divididas pelo total de
âncoras positivas do lote, como no Detectron2 (sem a média móvel do
denominador que ele usa).
"""

import types
from functools import partial

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection._utils import _box_loss
from torchvision.ops import sigmoid_focal_loss

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


def _classification_loss_batch(self, targets, head_outputs, matched_idxs):
    total, num_foreground = 0.0, 0
    for t, logits, matched in zip(targets, head_outputs["cls_logits"], matched_idxs):
        fg = matched >= 0
        num_foreground += int(fg.sum())
        gt = torch.zeros_like(logits)
        gt[fg, t["labels"][matched[fg]]] = 1.0
        valid = matched != self.BETWEEN_THRESHOLDS
        total = total + sigmoid_focal_loss(logits[valid], gt[valid], reduction="sum")
    return total / max(1, num_foreground)


def _regression_loss_batch(self, targets, head_outputs, anchors, matched_idxs):
    total, num_foreground = 0.0, 0
    for t, reg, anc, matched in zip(targets, head_outputs["bbox_regression"], anchors, matched_idxs):
        fg = torch.where(matched >= 0)[0]
        num_foreground += fg.numel()
        total = total + _box_loss(self._loss_type, self.box_coder, anc[fg], t["boxes"][matched[fg]], reg[fg])
    return total / max(1, num_foreground)


def build_detector(
    arch: str,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    nms_threshold: float = 0.3,
    score_threshold: float = 0.0,
    detections_per_image: int = 100,
    min_size: int = 800,
    max_size: int = 1333,
    retinanet_loss_normalization: str = "batch",
    retinanet_head_norm: str = "none",
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
        if retinanet_head_norm == "group":
            # GroupNorm nas torres da cabeça, como na retinanet_resnet50_fpn_v2 do torchvision
            num_anchors = net.anchor_generator.num_anchors_per_location()[0]
            net.head = RetinaNetHead(net.backbone.out_channels, num_anchors, 1, norm_layer=partial(nn.GroupNorm, 32))
        elif retinanet_head_norm != "none":
            raise ValueError(f"retinanet_head_norm: 'none' ou 'group', veio {retinanet_head_norm!r}")
        if retinanet_loss_normalization == "batch":
            head = net.head
            head.classification_head.compute_loss = types.MethodType(_classification_loss_batch, head.classification_head)
            head.regression_head.compute_loss = types.MethodType(_regression_loss_batch, head.regression_head)
        elif retinanet_loss_normalization != "image":
            raise ValueError(f"retinanet_loss_normalization: 'batch' ou 'image', veio {retinanet_loss_normalization!r}")
        return Detector(arch, net, label_offset=1)
    raise ValueError(f"arquitetura desconhecida: {arch!r}; opções: {ARCHS}")


@torch.no_grad()
def predict(model: Detector, images: list[torch.Tensor]) -> list[dict]:
    model.eval()
    return model(images)
