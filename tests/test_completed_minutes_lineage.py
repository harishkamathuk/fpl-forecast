"""Optional completed-history lineage flags retain Series semantics."""

import pandas as pd
import pytest

from fpl_forecast.operations.model_chain import _completed_minutes_training_frame


@pytest.mark.parametrize("with_lineage", [False, True])
def test_completed_minutes_optional_lineage_is_index_aligned(with_lineage):
    frame = pd.DataFrame({
        "season": ["2026-27"] * 3,
        "fixture_id": [7] * 3,
        "player_uid": ["player_a", "player_b", "player_c"],
        "team_uid": ["villa"] * 3,
        "opponent_uid": ["brighton"] * 3,
        "kickoff_time": ["2026-08-23T13:00:00Z"] * 3,
        "source_available_time": ["2026-11-04T15:44:22Z"] * 3,
        "minutes": [0, 90, 45],
        "starts": [0, 1, 1],
    }, index=[4, 8, 12])
    if with_lineage:
        frame["lineage_note"] = ["transferred_player", "position_change", ""]
    result = _completed_minutes_training_frame(frame)
    assert result.index.tolist() == [4, 8, 12]
    assert result["transferred_player"].tolist() == [with_lineage, False, False]
    assert result["position_change"].tolist() == [False, with_lineage, False]
    assert result["actual_minutes"].tolist() == [0, 90, 45]
    assert not result["prior_seen_before"].any()
    assert "lineage_note" in frame if with_lineage else "lineage_note" not in frame
