"""Integration contracts for P1 provisioning on develop's fixture-level history."""

from copy import deepcopy
from datetime import timedelta

import pandas as pd
import pytest

from test_prior_event_snapshots import OfficialCase
from test_p1_compatibility import reconstruct, recontext


def test_b2_partial_postponement_then_offline_reconstruction(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    live = case.live(1)
    # The postponed fixture has no completed player results, not inferred DNPs.
    live["elements"] = [r for r in live["elements"] if r["id"] not in (1, 2)]
    case.bootstrap["events"][0].update(finished=False, data_checked=False)
    case.fixtures[0].update(
        started=False, finished=False, finished_provisional=False,
        team_h_score=None, team_a_score=None,
        kickoff_time=case.fixtures[20]["kickoff_time"],
    )
    recontext(case)
    case.write("event_live_1", live)
    records = case.run()
    assert len(records) == 1 and not case.calls
    result = reconstruct(case, monkeypatch)
    players = pd.read_parquet(result.player_history_path)
    assert result.player_rows == 18 and result.team_rows == 9
    assert set(players["fixture_id"]) == set(range(2, 11))
    assert not {1, 2}.intersection(players["player_id"])


def test_b2_wholly_deferred_event_needs_no_live_snapshot(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    case.bootstrap["events"][0].update(finished=False, data_checked=False)
    for fixture in case.fixtures[:10]:
        fixture.update(started=False, finished=False, finished_provisional=False,
                       team_h_score=None, team_a_score=None,
                       kickoff_time=case.fixtures[20]["kickoff_time"])
    recontext(case)
    assert case.run() == []
    assert not case.calls and case.path("event_live_1") is None
    result = reconstruct(case, monkeypatch)
    assert result.player_rows == result.team_rows == 0


def test_b2_archived_identity_survives_later_roster_removal(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    original = deepcopy(case.bootstrap)
    case.write("bootstrap_static", original, at=case.retrieved - timedelta(days=2))
    live = case.live(1)
    case.bootstrap["elements"].pop(0)
    recontext(case)
    case.write("event_live_1", live)
    assert len(case.run()) == 1
    result = reconstruct(case, monkeypatch)
    rows = pd.read_parquet(result.player_history_path)
    assert result.player_rows == 20
    assert rows.loc[rows["player_id"].eq(1), "player_uid"].tolist() == ["player_code_1001"]


@pytest.mark.parametrize("field", ["finished", "data_checked"])
def test_completed_fixture_evidence_does_not_depend_on_event_flag(tmp_path, monkeypatch, field):
    case = OfficialCase(tmp_path, monkeypatch)
    case.bootstrap["events"][0][field] = False
    recontext(case)
    assert len(case.run()) == 1
    assert reconstruct(case, monkeypatch).player_rows == 20


@pytest.mark.parametrize("offset", [0, 1])
def test_b2_never_hides_finalized_fixture_at_or_after_cutoff(tmp_path, monkeypatch, offset):
    case = OfficialCase(tmp_path, monkeypatch)
    deadline = pd.Timestamp(case.bootstrap["events"][1]["deadline_time"])
    case.fixtures[0]["kickoff_time"] = (deadline + pd.Timedelta(seconds=offset)).isoformat()
    recontext(case)
    with pytest.raises(ValueError, match="finalized at or after"):
        case.run()
    assert not case.calls
