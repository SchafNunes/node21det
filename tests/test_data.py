import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import torch

from node21det.data.dataset import NoduleDataset, read_image
from node21det.data.dedup import group_pairs, near_duplicate_pairs, thumbnail
from node21det.data.metadata import boxes_by_image, image_table, load_metadata
from node21det.data.splits import make_splits, select
from node21det.data.transforms import RandomBrightnessContrast, RandomHorizontalFlip

CSV = """,height,img_name,label,width,x,y
0,10,a.mha,1,20,100,200
1,30,a.mha,1,40,300,400
2,0,b.mha,0,0,0,0
3,5,c.mha,1,5,1,2
"""


@pytest.fixture
def meta(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text(CSV)
    return load_metadata(path)


def test_metadata_drops_index_and_converts_boxes(meta):
    assert list(meta.columns) == ["img_name", "x", "y", "width", "height", "label"]
    boxes = boxes_by_image(meta)
    assert boxes["a.mha"].tolist() == [[100, 200, 120, 210], [300, 400, 340, 430]]
    assert boxes["b.mha"].shape == (0, 4)


def test_image_table(meta):
    t = image_table(meta).set_index("img_name")
    assert t["n_nodules"].to_dict() == {"a.mha": 2, "b.mha": 0, "c.mha": 1}
    assert t["positive"].to_dict() == {"a.mha": True, "b.mha": False, "c.mha": True}


def _images(n_pos=300, n_neg=1000):
    names = [f"p{i:04d}" for i in range(n_pos)] + [f"n{i:04d}" for i in range(n_neg)]
    return pd.DataFrame({"img_name": names, "positive": [True] * n_pos + [False] * n_neg})


def test_splits_proportions_and_stratification():
    s = make_splits(_images())
    frac = s["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.01)
    assert frac["val"] == pytest.approx(0.15, abs=0.01)
    pos_frac = s.groupby("split")["positive"].mean()
    assert (pos_frac - 300 / 1300).abs().max() < 0.01


def test_splits_keep_groups_together():
    images = _images()
    groups = {f"n{i:04d}": "dup" for i in range(10)}
    s = make_splits(images, groups)
    assert s[s["group"] == "dup"]["split"].nunique() == 1
    assert s[s["group"] == "dup"]["fold"].nunique() == 1


def test_splits_deterministic_and_seed_dependent():
    a, b = make_splits(_images(), seed=1), make_splits(_images(), seed=1)
    assert a.equals(b)
    assert not a["split"].equals(make_splits(_images(), seed=2)["split"])


def test_folds_cover_selection_pool_only():
    s = make_splits(_images(), k_folds=3)
    assert (s.loc[s["split"] == "test", "fold"] == -1).all()
    pool = s[s["split"] != "test"]
    counts = pool["fold"].value_counts()
    assert sorted(counts.index) == [0, 1, 2]
    assert counts.max() - counts.min() <= 2
    train_f0 = set(select(s, exclude_folds=[0]))
    assert train_f0.isdisjoint(select(s, folds=[0]))
    assert train_f0.isdisjoint(select(s, split="test"))
    assert len(select(s, exclude_folds=[])) == len(pool)


def test_correlation_groups_near_duplicates():
    rng = np.random.default_rng(0)
    base = rng.random((256, 256))
    near = base * 0.9 + 0.05 + rng.normal(0, 0.005, base.shape)  # mesma imagem, intensidade reescalada
    other = rng.random((256, 256))
    names = ["a", "b", "c"]
    thumbs = np.stack([thumbnail(x) for x in (base, near, other)])
    pairs = near_duplicate_pairs(names, thumbs, min_corr=0.99)
    assert [(a, b) for a, b, _ in pairs] == [("a", "b")]
    assert group_pairs(names, pairs) == {"a": "a", "b": "a", "c": "c"}


def test_group_pairs_is_transitive():
    g = group_pairs(["a", "b", "c", "d"], [("c", "d", 1), ("b", "c", 1)])
    assert g == {"a": "a", "b": "b", "c": "b", "d": "b"}


def test_hflip_mirrors_boxes():
    img = torch.zeros(1, 10, 100)
    target = {"boxes": torch.tensor([[10.0, 0, 30, 5]])}
    _, t = RandomHorizontalFlip(1.0)(img, target)
    assert t["boxes"].tolist() == [[70, 0, 90, 5]]
    assert target["boxes"].tolist() == [[10, 0, 30, 5]]  # não altera o alvo original


def test_photometric_keeps_range_and_boxes():
    img = torch.rand(1, 32, 32)
    target = {"boxes": torch.tensor([[1.0, 2, 3, 4]])}
    out, t = RandomBrightnessContrast(1.0, 0.5, 0.5)(img, target)
    assert out.min() >= 0 and out.max() <= 1
    assert t["boxes"].tolist() == [[1, 2, 3, 4]]


def test_dataset_reads_mha_and_builds_target(tmp_path, meta):
    arr = (np.arange(64 * 64).reshape(64, 64) % 1000).astype(np.uint16)
    for name in ["a.mha", "b.mha"]:
        sitk.WriteImage(sitk.GetImageFromArray(arr), str(tmp_path / name))
    ds = NoduleDataset(tmp_path, ["a.mha", "b.mha"], boxes_by_image(meta))
    img, t = ds[0]
    assert img.shape == (1, 64, 64) and img.max() == pytest.approx(1.0)
    assert t["boxes"].shape == (2, 4) and t["labels"].tolist() == [1, 1]
    _, t = ds[1]
    assert t["boxes"].shape == (0, 4) and t["labels"].shape == (0,)


def test_read_image_rejects_stack(tmp_path):
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((3, 8, 8), np.uint16)), str(tmp_path / "s.mha"))
    with pytest.raises(ValueError):
        read_image(tmp_path / "s.mha")


def test_balanced_sampler_draws_half_positives():
    from node21det.data.dataset import balanced_sampler

    names = [f"p{i}" for i in range(23)] + [f"n{i}" for i in range(77)]
    boxes = {n: (np.array([[0, 0, 5, 5]], np.float32) if n.startswith("p") else np.zeros((0, 4), np.float32))
             for n in names}
    torch.manual_seed(0)
    sampler = balanced_sampler(names, boxes)
    assert len(sampler) == len(names)  # a época mantém o tamanho do conjunto
    draws = [names[i] for _ in range(50) for i in sampler]
    frac = np.mean([d.startswith("p") for d in draws])
    assert 0.46 < frac < 0.54


def test_balanced_configs_differ_only_in_sampling():
    from node21det.config import load_config

    for arch in ("frcnn", "retinanet"):
        base, bal = load_config(f"configs/{arch}_none.yaml"), load_config(f"configs/{arch}_balanced.yaml")
        assert bal.train.balanced_sampling and not base.train.balanced_sampling
        bal.train.balanced_sampling, bal.name = False, base.name
        assert bal.to_dict() == base.to_dict()


def test_clip_configs_differ_only_in_gradient_clipping():
    from node21det.config import load_config

    for arch in ("frcnn", "retinanet"):
        bal, clip = load_config(f"configs/{arch}_balanced.yaml"), load_config(f"configs/{arch}_balanced_clip.yaml")
        assert clip.optim.grad_clip_norm == 3.0 and bal.optim.grad_clip_norm is None
        clip.optim.grad_clip_norm, clip.name = None, bal.name
        assert clip.to_dict() == bal.to_dict()
