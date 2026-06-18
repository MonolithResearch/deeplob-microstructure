import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.signal_logic import direction_from_prob_ofi


def test_contrarian_fades_agreed_up_pressure():
    assert direction_from_prob_ofi(0.2, 0.7, 1.4, 0.38, 1.0) == -1


def test_contrarian_fades_agreed_down_pressure():
    assert direction_from_prob_ofi(0.7, 0.2, -1.4, 0.38, 1.0) == +1


def test_trend_mode_takes_agreed_direction():
    assert direction_from_prob_ofi(0.2, 0.7, 1.4, 0.38, 1.0, mode="trend") == +1


def test_blocks_weak_or_disagreeing_or_nonfinite():
    assert direction_from_prob_ofi(0.37, 0.36, 2.0, 0.38, 1.0) == 0   # below confidence gate
    assert direction_from_prob_ofi(0.1, 0.8, 0.9, 0.38, 1.0) == 0     # below OFI gate
    assert direction_from_prob_ofi(0.1, 0.8, -2.0, 0.38, 1.0) == 0    # disagree (UP prob, sell flow)
    assert direction_from_prob_ofi(math.nan, 0.8, 2.0, 0.38, 1.0) == 0
