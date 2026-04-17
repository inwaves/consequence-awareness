"""Tests for activation layer ranking helpers."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rank_activation_layers import cosine_distance, l2_distance, token_region_mask  # noqa: E402


def test_token_region_mask_supports_all_and_args():
    metadata = {
        "tokens": [
            {"assistant_token_index": 0, "regions": ["planning"]},
            {"assistant_token_index": 1, "regions": ["tool_name"]},
            {"assistant_token_index": 2, "regions": ["arg.path"]},
            {"assistant_token_index": 3, "regions": ["arg.content", "tool_arguments"]},
        ]
    }
    assert token_region_mask(metadata, "all") == [0, 1, 2, 3]
    assert token_region_mask(metadata, "planning") == [0]
    assert token_region_mask(metadata, "tool_name") == [1]
    assert token_region_mask(metadata, "tool_arguments") == [3]
    assert token_region_mask(metadata, "args") == [2, 3]


def test_distance_helpers_upcast_low_precision_vectors():
    np = pytest.importorskip("numpy")

    a = np.array([60000.0, 60000.0], dtype=np.float16)
    b = np.array([60000.0, -60000.0], dtype=np.float16)

    assert np.isfinite(cosine_distance(a, b))
    assert np.isfinite(l2_distance(a, b))
