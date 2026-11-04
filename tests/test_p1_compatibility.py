"""Contract-shaped edge cases; all evidence and reconstruction writes are temporary.

Local archived bootstrap snapshots confirm changing rosters and team_join_date.
The fixture archive's data dictionary permits unassigned events. These fixtures
exercise those shapes, not a claim to reproduce a captured event-live response.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pandas as pd
import pytest

from fpl_forecast.ingest.season import SeasonIdentityError
from fpl_forecast.operations.current_panel import reconstruct_completed_current_season
from fpl_forecast.operations.live_results import VALUE_COMPONENTS, normalize_event_live
from test_prior_event_snapshots import OfficialCase, SEASON


def recontext(case):
    case.retrieved += timedelta(seconds=1)
    case.context()


def reconstruct(case, monkeypatch):
    monkeypatch.setattr(
        "fpl_forecast.operations.current_panel._current_team_identities",
        lambda teams, **kw: teams.assign(team_uid=teams["team_id"].map(lambda i: f"team_{i}")),
    )
    return reconstruct_completed_current_season(
        season=SEASON,
        target_gameweek=case.target,
        information_cutoff=case.bootstrap["events"][case.target - 1]["deadline_time"],
        raw_fpl_dir=case.raw,
        normalized_dir=case.normalized,
        run_id="compatibility",
        requested_gameweek=case.target,
        git_commit="test",
        clean_source=True,
        refresh=False,
    )


@pytest.mark.parametrize("event,kickoff", [(None, None), (None, "known"), (38, None)])
def test_future_unscheduled_fixture_does_not_block_prior_results(
    tmp_path, monkeypatch, event, kickoff
):
    case = OfficialCase(tmp_path, monkeypatch)
    future = case.fixtures[-1]
    future["event"] = event
    if kickoff is None:
        future["kickoff_time"] = None
    recontext(case)
    # This is also the phase-A season-identity gate, with network prohibited.
    assert len(case.client.snapshot_current(season=SEASON, offline=True)) == 2
    normalized = pd.read_parquet(case.normalized / SEASON / "current_fixtures.parquet")
    assert len(normalized) == 380
    assert pd.isna(normalized.iloc[-1]["gameweek"]) == (event is None)
    assert pd.isna(normalized.iloc[-1]["kickoff_time"]) == (kickoff is None)
    assert [r.endpoint_name for r in case.run()] == ["event_live_1"]
    result = reconstruct(case, monkeypatch)
    assert result.player_rows == 20
    assert result.team_rows == 10
    assert set(pd.read_parquet(result.player_history_path)["fixture_id"]) == set(range(1, 11))


@pytest.mark.parametrize("index", [0, 10])
def test_missing_prior_or_target_kickoff_still_fails(tmp_path, monkeypatch, index):
    case = OfficialCase(tmp_path, monkeypatch)
    case.fixtures[index]["kickoff_time"] = None
    try:
        recontext(case)
    except (ValueError, SeasonIdentityError):
        pass
    with pytest.raises((ValueError, SeasonIdentityError)):
        case.run()
    assert not case.calls


@pytest.mark.parametrize(
    "field,value",
    [("started", True), ("finished", True), ("finished_provisional", True), ("team_h_score", 0)],
)
def test_unscheduled_fixture_cannot_hide_result_evidence(tmp_path, monkeypatch, field, value):
    case = OfficialCase(tmp_path, monkeypatch)
    case.fixtures[-1].update(event=None, kickoff_time=None)
    case.fixtures[-1][field] = value
    try:
        recontext(case)
    except (ValueError, SeasonIdentityError):
        pass
    with pytest.raises((ValueError, SeasonIdentityError)):
        case.run()
    assert not case.calls


@pytest.mark.parametrize("field,value", [("event", "invalid"), ("kickoff_time", "invalid")])
def test_null_support_does_not_turn_malformed_values_into_missing(
    tmp_path, monkeypatch, field, value
):
    case = OfficialCase(tmp_path, monkeypatch)
    case.fixtures[-1][field] = value
    with pytest.raises((ValueError, SeasonIdentityError)):
        recontext(case)


def test_later_roster_addition_does_not_invent_historical_results(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    original = case.live(1)
    case.write("event_live_1", original)
    new_player = {
        **case.bootstrap["elements"][0],
        "id": 21,
        "code": 1021,
        "web_name": "New registrant",
    }
    case.bootstrap["elements"].append(new_player)
    recontext(case)
    case.run()
    assert not case.calls  # no refresh or older-cache fallback
    result = reconstruct(case, monkeypatch)
    rows = pd.read_parquet(result.player_history_path)
    assert result.player_rows == 20
    assert 21 not in set(rows["player_id"])
    assert len(original["elements"]) == 20


def test_addition_with_explicit_event_evidence_is_supported(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    case.bootstrap["elements"].append({**case.bootstrap["elements"][0], "id": 21, "code": 1021})
    recontext(case)
    case.run()
    result = reconstruct(case, monkeypatch)
    assert result.player_rows == 21
    assert 21 in set(pd.read_parquet(result.player_history_path)["player_id"])


def test_removed_flag_does_not_erase_official_historical_results(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    historical = case.live(1)
    case.write("event_live_1", historical)
    case.bootstrap["elements"][0].update(removed=True, can_select=False, can_transact=False)
    recontext(case)
    case.run()
    result = reconstruct(case, monkeypatch)
    rows = pd.read_parquet(result.player_history_path)
    assert 1 in set(rows["player_id"])
    assert rows.loc[rows["player_id"].eq(1), "total_points"].tolist() == [2]


def test_physically_missing_identity_is_not_silently_dropped(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    historical = case.live(1)
    case.write("event_live_1", historical)
    # Remove the only archived identity in this negative case; develop can
    # legitimately recover disappeared players when that evidence is retained.
    prior_bootstrap = case.path("bootstrap_static")
    prior_bootstrap.unlink()
    prior_bootstrap.with_suffix(".json.metadata.json").unlink()
    case.bootstrap["elements"].pop(0)
    recontext(case)
    with pytest.raises(ValueError, match="element-summary|player coverage"):
        case.run()
    with pytest.raises(ValueError, match="element-summary|identity|absent from bootstrap-static"):
        reconstruct(case, monkeypatch)
    assert not list(case.normalized.rglob("*history*"))


@pytest.mark.parametrize("new_team,joined", [(3, None), (2, "2026-08-20")])
def test_transferred_player_uses_archive_or_requires_historical_team_evidence(
    tmp_path, monkeypatch, new_team, joined
):
    case = OfficialCase(tmp_path, monkeypatch)
    original = case.live(1)
    case.write("event_live_1", original)
    case.bootstrap["elements"][0].update(team=new_team, team_join_date=joined)
    recontext(case)
    if new_team == 3:
        # The original bootstrap is an authentic retained identity/club source.
        case.run()
        result = reconstruct(case, monkeypatch)
        rows = pd.read_parquet(result.player_history_path)
        player = rows.loc[rows["player_id"].eq(1)].iloc[0]
        assert player["historical_team_id"] == 1
        assert player["current_team_id"] == 3
        assert player["historical_club_resolution_method"] == "archived_bootstrap_fixture_compatible"
        return
    with pytest.raises(ValueError, match="reconciled|historical team|element-summary"):
        case.run()
    with pytest.raises(ValueError, match="reconciled|historical team|element-summary"):
        reconstruct(case, monkeypatch)
    assert not list(case.normalized.rglob("*history*"))


def test_missing_fixture_summary_cannot_manufacture_a_dnp(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    original = case.live(1)
    original["elements"][0].update(
        stats={**dict.fromkeys(VALUE_COMPONENTS, 0), "starts": 0, "total_points": 0}, explain=[]
    )
    case.write("event_live_1", original)
    case.bootstrap["elements"][0]["team_join_date"] = "2026-08-20"
    recontext(case)
    # Develop requires an official fixture-specific summary when live explain
    # is absent. Missing evidence must not become a guessed historical DNP.
    with pytest.raises(ValueError, match="element-summary"):
        case.run()
    with pytest.raises(ValueError, match="element-summary"):
        reconstruct(case, monkeypatch)
    assert not list(case.normalized.rglob("*history*"))



def test_known_membership_before_fixture_is_unchanged(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    case.bootstrap["elements"][0]["team_join_date"] = "2026-07-01"
    recontext(case)
    case.run()
    result = reconstruct(case, monkeypatch)
    assert result.player_rows == 20


def test_latest_bootstrap_alone_cannot_resolve_opponent_transfer(tmp_path, monkeypatch):
    """Reproduce the unsafe pre-fix interpretation; require rejection with known join date."""
    case = OfficialCase(tmp_path, monkeypatch)
    payload = case.live(1)
    changed = deepcopy(case.bootstrap)
    changed["elements"][0].update(team=2, team_join_date="2026-08-20")
    with pytest.raises(ValueError, match="historical team"):
        normalize_event_live(
            season=SEASON,
            gameweek=1,
            payload=payload,
            retrieved_at=case.retrieved.isoformat(),
            raw_snapshot_path="test",
            bootstrap_payload=changed,
            fixtures_payload=case.fixtures,
        )


def test_all_kickoffs_missing_cannot_establish_season(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch, target=1)
    for fixture in case.fixtures:
        fixture["kickoff_time"] = None
    with pytest.raises(SeasonIdentityError, match="no parseable"):
        recontext(case)


@pytest.mark.parametrize("kickoff", ["known", None])
def test_unassigned_scored_fixture_never_becomes_future_evidence(tmp_path, monkeypatch, kickoff):
    case = OfficialCase(tmp_path, monkeypatch)
    case.fixtures[-1]["event"] = None
    case.fixtures[-1]["team_h_score"] = 0
    if kickoff is None:
        case.fixtures[-1]["kickoff_time"] = None
    with pytest.raises(ValueError, match="Scored fixture|Result-bearing"):
        recontext(case)


def test_model_target_validator_still_requires_known_kickoff(tmp_path, monkeypatch):
    # Invoke only the table validator/transform, never the forecasting chain.
    from fpl_forecast.operations.model_chain import _official_target_fixtures

    case = OfficialCase(tmp_path, monkeypatch)
    fixtures = pd.read_parquet(case.normalized / SEASON / "current_fixtures.parquet")
    teams = pd.read_parquet(case.normalized / SEASON / "current_teams.parquet")
    teams["team_uid"] = teams["team_id"].map(lambda i: f"team_{i}")
    target = fixtures.loc[fixtures["gameweek"].eq(2)].copy()
    cutoff = pd.Timestamp(case.bootstrap["events"][1]["deadline_time"])
    context = {"fixtures": target, "team_identity": teams}
    assert len(_official_target_fixtures(context, as_of=cutoff)) == 10
    context["fixtures"].loc[target.index[0], "kickoff_time"] = pd.NA
    with pytest.raises(ValueError, match="invalid kickoff_time"):
        _official_target_fixtures(context, as_of=cutoff)
