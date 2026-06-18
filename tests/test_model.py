import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import WINDOW_SIZE, N_FEATURES, N_CLASSES
from src.deeplob_model import build_model


def test_forward_shape():
    model = build_model()
    x = torch.randn(8, WINDOW_SIZE, N_FEATURES)
    out = model(x)
    assert out.shape == (8, N_CLASSES)


def test_predict_proba_is_a_distribution():
    model = build_model()
    x = torch.randn(4, WINDOW_SIZE, N_FEATURES)
    p = model.predict_proba(x)
    assert p.shape == (4, N_CLASSES)
    assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (p >= 0).all()


def test_save_and_load_roundtrip(tmp_path):
    from src.deeplob_model import DeepLOB
    model = build_model().eval()   # eval: deterministic (no dropout, BN running stats)
    ckpt = tmp_path / "m.pt"
    model.save(ckpt)
    reloaded = DeepLOB.load(ckpt)  # .load() already calls .eval()
    x = torch.randn(2, WINDOW_SIZE, N_FEATURES)
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-5)
