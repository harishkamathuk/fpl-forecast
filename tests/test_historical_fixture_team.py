"""Player 28's recorded GW1 DNP belongs to Aston Villa, not his current Chelsea club."""

from __future__ import annotations

from datetime import UTC, datetime
import json

import pandas as pd
import pytest

from fpl_forecast.ingest.snapshots import write_raw_snapshot
from fpl_forecast.operations.current_panel import build_current_player_fixture_history
from fpl_forecast.operations.live_results import normalize_event_live, audit_event_live_scoring


def case(tmp_path, *, current_team=6, old_team=2, old_code=98980, at=None):
    # Minimal unit projection of the recorded fixture/player facts, not a claim
    # that this synthetic bootstrap is a complete captured production response.
    teams = [
        {"id": 2, "name": "Aston Villa"},
        {"id": 5, "name": "Brighton"},
        {"id": 6, "name": "Chelsea"},
    ]
    bootstrap = {
        "events": [],
        "teams": teams,
        "elements": [
            {
                "id": 28,
                "code": 98980,
                "web_name": "Martinez",
                "element_type": 1,
                "team": current_team,
                "team_join_date": "2026-08-29",
            }
        ],
        "game_settings": {},
        "phases": [],
        "total_players": 1,
        "element_stats": [],
        "element_types": [],
    }
    historical = {
        **bootstrap,
        "elements": [
            {
                **bootstrap["elements"][0],
                "team": old_team,
                "code": old_code,
                "team_join_date": "2020-09-16",
            }
        ],
    }
    record = write_raw_snapshot(
        tmp_path,
        season="2026-27",
        endpoint_name="bootstrap_static",
        content=json.dumps(historical).encode(),
        source="fpl_api",
        source_url="https://fantasy.premierleague.com/api/bootstrap-static/",
        http_status=200,
        retrieved_at=at or datetime(2026, 8, 21, 18, 38, 42, tzinfo=UTC),
    )
    fixture = {
        "id": 7,
        "event": 1,
        "team_h": 5,
        "team_a": 2,
        "kickoff_time": "2026-08-23T13:00:00Z",
        "finished": True,
        "finished_provisional": True,
        "team_h_score": 4,
        "team_a_score": 0,
    }
    payload = {
        "elements": [
            {
                "id": 28,
                "stats": {"minutes": 0, "total_points": 0},
                "explain": [
                    {"fixture": 7, "stats": [{"identifier": "minutes", "value": 0, "points": 0}]}
                ],
            }
        ]
    }
    kwargs = dict(
        season="2026-27",
        gameweek=1,
        payload=payload,
        retrieved_at="2026-11-04T12:04:15Z",
        raw_snapshot_path="recorded-live",
        bootstrap_payload=bootstrap,
        fixtures_payload=[fixture],
        raw_fpl_dir=tmp_path,
    )
    return kwargs, record, fixture


@pytest.mark.parametrize("current_team", [6, 5])
def test_recorded_transfer_and_opponent_transfer_use_historical_away_side(tmp_path, current_team):
    kwargs, record, fixture = case(tmp_path, current_team=current_team)
    live = normalize_event_live(**kwargs)
    assert live["fixture_team_id"].tolist() == [2]
    assert live["current_team_id"].tolist() == [current_team]
    assert live["historical_club_evidence_raw_snapshot_path"].tolist() == [str(record.raw_path)]
    assert live["historical_club_evidence_sha256"].tolist() == [record.checksum_sha256]
    assert live["historical_club_resolution_method"].tolist() == ["archived_bootstrap_fixture_compatible"]
    assert live["was_home"].tolist() == [False]
    assert live["total_points"].tolist() == [0]
    assert set(audit_event_live_scoring(live)["player_event_reconciliation"]["audit_status"]) == {
        "exact_match"
    }
    source = live.attrs["historical_bootstrap_snapshots"][0]
    assert source["metadata"]["sha256"] == record.checksum_sha256
    assert source["metadata"]["retrieved_at"] == record.retrieved_at
    players = pd.DataFrame(
        [
            {
                "player_id": 28,
                "player_code": 98980,
                "web_name": "Martinez",
                "team_id": current_team,
                "position": "GKP",
                "price_tenths": 50,
                "entity_type": "player",
            }
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                **fixture,
                "fixture_id": 7,
                "gameweek": 1,
                "home_team_id": 5,
                "away_team_id": 2,
                "raw_snapshot_path": "official-fixtures",
            }
        ]
    )
    identities = pd.DataFrame({"team_id": [2, 5, 6], "team_uid": ["villa", "brighton", "chelsea"]})
    history = build_current_player_fixture_history(
        fixtures=fixtures, live_results=live, players=players, team_identities=identities
    )
    assert history["team_id"].tolist() == [2]
    assert history["player_team_uid"].tolist() == ["villa"]
    assert history["opponent_team_uid"].tolist() == ["brighton"]


@pytest.mark.parametrize("defect", ["missing", "corrupt", "code", "wrong_side", "at_kickoff"])
def test_unavailable_or_contradictory_historical_team_fails_closed(tmp_path, defect):
    kwargs, record, _ = case(
        tmp_path,
        old_team=6 if defect == "wrong_side" else 2,
        old_code=99999 if defect == "code" else 98980,
        at=datetime(2026, 8, 23, 13, tzinfo=UTC) if defect == "at_kickoff" else None,
    )
    if defect == "missing":
        record.raw_path.unlink()
        record.metadata_path.unlink()
    elif defect == "corrupt":
        record.raw_path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="historical team"):
        normalize_event_live(**kwargs)


def test_absent_fixture_explanations_do_not_create_dnp_assignments(tmp_path):
    kwargs, _, _ = case(tmp_path)
    kwargs["payload"]["elements"][0]["explain"] = []
    assert normalize_event_live(**kwargs).empty



def test_completed_player_history_keeps_one_authoritative_kickoff_column():
    live = pd.DataFrame([{
        "season": "2026-27", "gameweek": 1, "fixture_id": 7, "player_id": 28,
        "player_uid": "player_code_98980", "fixture_team_id": 2,
        "kickoff_time": "2026-08-23T13:00:00Z", "minutes": 0, "total_points": 0,
        "source_available_time": "2026-11-04T13:58:00Z", "raw_snapshot_path": "official-live",
    }])
    players = pd.DataFrame([{
        "player_id": 28, "player_code": 98980, "web_name": "Martinez", "team_id": 6,
        "position": "GKP", "price_tenths": 50, "entity_type": "player",
    }])
    fixtures = pd.DataFrame([{
        "fixture_id": 7, "gameweek": 1, "kickoff_time": "2026-08-23T13:00:00Z",
        "finished": True, "finished_provisional": True, "home_team_id": 5,
        "away_team_id": 2, "team_h_score": 4, "team_a_score": 0,
        "raw_snapshot_path": "official-fixtures",
    }])
    history = build_current_player_fixture_history(
        fixtures=fixtures, live_results=live, players=players,
    )
    assert len(history) == 1
    assert history["kickoff_time"].tolist() == pd.to_datetime(fixtures["kickoff_time"], utc=True).tolist()
    assert "kickoff_time_x" not in history and "kickoff_time_y" not in history
    assert history["source_available_time"].tolist() == live["source_available_time"].tolist()
    assert history["raw_snapshot_path"].tolist() == ["official-live"]
    assert history["fixtures_raw_snapshot_path"].tolist() == ["official-fixtures"]
    assert history["team_id"].tolist() == [2]
    assert history["total_points"].tolist() == [0]
