import numpy as np
import pytest

from node21det.config import load_config
from node21det.data.enhancement import LEVELS, build_enhancement, clahe, histogram_equalization, rmshe


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    # radiografia sintética: gradiente com ruído, concentrada no meio da faixa
    x = np.linspace(0.3, 0.7, 256)[None, :] * np.ones((256, 1))
    return np.clip(x + rng.normal(0, 0.02, x.shape), 0, 1).astype(np.float32)


@pytest.mark.parametrize("fn", [histogram_equalization, clahe, rmshe])
def test_shape_range_dtype(fn, image):
    out = fn(image)
    assert out.shape == image.shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


def test_he_spreads_histogram_and_is_monotonic(image):
    out = histogram_equalization(image)
    assert out.max() - out.min() > (image.max() - image.min()) * 1.5
    order = np.argsort(image.ravel(), kind="stable")
    assert np.all(np.diff(out.ravel()[order]) >= -1e-6)


def test_rmshe_preserves_mean_brightness_better_than_he(image):
    he, rm = histogram_equalization(image), rmshe(image, recursion=2)
    assert abs(rm.mean() - image.mean()) < abs(he.mean() - image.mean())
    assert abs(rm.mean() - image.mean()) < 0.02


def test_rmshe_keeps_pixels_on_their_side_of_the_mean(image):
    # recursão 1: pixels abaixo da média ficam abaixo, acima ficam acima
    q = np.rint(image * (LEVELS - 1))
    m = np.floor(q.mean()) / (LEVELS - 1)
    out = rmshe(image, recursion=1)
    assert np.all(out[image <= m] <= m + 1e-6)
    assert np.all(out[image > m] >= m)


def test_build_enhancement_and_configs():
    assert build_enhancement("none") is None
    for name in ("he", "clahe", "rmshe"):
        cfg = load_config(f"configs/frcnn_{name}.yaml")
        assert cfg.enhancement == name
        assert build_enhancement(cfg.enhancement, cfg.enhancement_params) is not None
    assert load_config("configs/frcnn_clahe.yaml").enhancement_params == {"clip_limit": 2.0, "tile_grid": 8}
    with pytest.raises(ValueError):
        build_enhancement("gamma")
