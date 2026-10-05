"""Testes de integração em CPU, com imagens sintéticas pequenas e backbone sem pesos."""

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from node21det.config import RunConfig
from node21det.data.dataset import NoduleDataset, collate
from node21det.models import build_detector
from node21det.training import fit

SMALL = dict(pretrained_backbone=False, min_size=128, max_size=128)


@pytest.mark.parametrize("arch", ["faster_rcnn", "retinanet"])
def test_detector_settings_and_labels(arch):
    model = build_detector(arch, **SMALL)
    net = model.net
    if arch == "faster_rcnn":
        assert (net.roi_heads.score_thresh, net.roi_heads.nms_thresh, net.roi_heads.detections_per_img) == (0.0, 0.3, 100)
    else:
        assert (net.score_thresh, net.nms_thresh, net.detections_per_img) == (0.0, 0.3, 100)

    images = [torch.rand(1, 128, 128), torch.rand(1, 128, 128)]
    targets = [
        {"boxes": torch.tensor([[10.0, 10, 40, 40]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)},
    ]
    model.train()
    losses = model(images, targets)
    assert all(torch.isfinite(v) for v in losses.values())

    model.eval()
    with torch.no_grad():
        out = model(images)
    assert len(out) == 2
    assert all((o["labels"] == 1).all() for o in out)


def test_pretrained_backbone_uses_frozen_batchnorm():
    from torchvision.ops import FrozenBatchNorm2d
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    # sem baixar pesos: a escolha de normalização depende só de haver pesos pedidos
    net = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    assert not isinstance(net.backbone.body.layer1[0].bn1, FrozenBatchNorm2d)


@pytest.fixture
def tiny_data(tmp_path):
    rng = np.random.default_rng(0)
    boxes = {}
    for i in range(4):
        img = (rng.random((128, 128)) * 1000).astype(np.uint16)
        name = f"i{i}.mha"
        if i % 2 == 0:
            img[30:60, 30:60] = 4000
            boxes[name] = np.array([[30, 30, 60, 60]], np.float32)
        else:
            boxes[name] = np.zeros((0, 4), np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(img), str(tmp_path / name))
    ds = NoduleDataset(tmp_path, list(boxes), boxes)
    return DataLoader(ds, batch_size=2, collate_fn=collate)


def _cfg():
    cfg = RunConfig()
    cfg.model.pretrained_backbone = False
    cfg.train.amp = False
    return cfg


def _model():
    return build_detector("faster_rcnn", **SMALL)


def test_fit_writes_history_and_resumes(tiny_data, tmp_path):
    out = tmp_path / "run"
    cpu = torch.device("cpu")
    state = fit(_cfg(), _model(), tiny_data, tiny_data, out, cpu, max_epochs=1)
    assert state["epoch"] == 0
    assert (out / "last.pt").exists() and (out / "best.pt").exists()

    state = fit(_cfg(), _model(), tiny_data, tiny_data, out, cpu, max_epochs=2)
    assert state["epoch"] == 1
    hist = pd.read_csv(out / "history.csv")
    assert hist["epoch"].tolist() == [0, 1]
    assert {"val_mean_sens", "val_rank", "val_auc", "val_auc_outside_nodules", "val_sens@0.25", "train_loss"} <= set(hist.columns)


def test_fit_early_stopping(tiny_data, tmp_path):
    cfg = _cfg()
    cfg.train.early_stopping_patience = 0  # para logo depois da primeira época
    state = fit(cfg, _model(), tiny_data, tiny_data, tmp_path / "run", torch.device("cpu"), max_epochs=5)
    assert state["epoch"] == 0


def test_fit_without_validation_runs_fixed_epochs(tiny_data, tmp_path):
    state = fit(_cfg(), _model(), tiny_data, None, tmp_path / "run", torch.device("cpu"), max_epochs=2)
    assert state["epoch"] == 1 and state["best_epoch"] == -1


def test_retinanet_batch_normalization_weights_negatives_by_batch_foreground():
    """Uma positiva e uma negativa: a perda em lote é (soma das duas) / âncoras positivas do lote."""
    torch.manual_seed(0)
    images = [torch.rand(1, 128, 128), torch.rand(1, 128, 128)]
    targets = [
        {"boxes": torch.tensor([[10.0, 10, 60, 60]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)},
    ]
    by_image = build_detector("retinanet", **SMALL, retinanet_loss_normalization="image")
    by_batch = build_detector("retinanet", **SMALL, retinanet_loss_normalization="batch")
    by_batch.load_state_dict(by_image.state_dict())
    for m in (by_image, by_batch):
        m.train()
        for mod in m.modules():  # sem pesos a normalização é treinável; fixa para o lote não mudar as ativações
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
    li, lb = by_image(images, targets), by_batch(images, targets)
    # na imagem positiva só, as duas normalizações coincidem
    pi, pb = by_image(images[:1], targets[:1]), by_batch(images[:1], targets[:1])
    assert pb["classification"].item() == pytest.approx(pi["classification"].item(), rel=1e-4)
    assert pb["bbox_regression"].item() == pytest.approx(pi["bbox_regression"].item(), rel=1e-4)
    # com a negativa, o padrão do torchvision dilui a regressão pela metade; a versão em lote não
    assert lb["bbox_regression"].item() == pytest.approx(pi["bbox_regression"].item(), rel=1e-4)
    assert li["bbox_regression"].item() == pytest.approx(pi["bbox_regression"].item() / 2, rel=1e-4)
    # classificação: a negativa sozinha tem perda ni (soma das âncoras de fundo / 1)
    ni = by_image(images[1:], targets[1:])["classification"].item()
    assert li["classification"].item() == pytest.approx((pi["classification"].item() + ni) / 2, rel=1e-4)
    # em lote, a parcela da negativa é dividida pelas âncoras positivas do lote (> 1)
    assert 0 < lb["classification"].item() - pb["classification"].item() < ni


def test_evaluate_run_uses_saved_config_and_writes_predictions(tiny_data, tmp_path):
    from node21det.inference import evaluate_run

    cfg = _cfg()
    cfg.model.min_size = cfg.model.max_size = 128
    cfg.enhancement, cfg.enhancement_params = "clahe", {"clip_limit": 2.0, "tile_grid": 8}
    out = tmp_path / "run"
    fit(cfg, _model(), tiny_data, None, out, torch.device("cpu"), max_epochs=1)
    ds = tiny_data.dataset
    metrics, curve, preds = evaluate_run(out, ds.image_dir, ds.boxes, ds.img_names, checkpoint="last.pt",
                                         num_workers=0)
    assert metrics["enhancement"] == "clahe" and metrics["epoch"] == 0
    assert {"mean_sens", "rank", "auc", "auc_outside_nodules"} <= set(metrics)
    assert list(preds.columns) == ["img_name", "image_positive", "x1", "y1", "x2", "y2", "score", "is_tp"]
    assert set(preds["img_name"]) <= set(ds.img_names)
    assert (curve["fps_per_image"].diff().dropna() >= 0).all()
