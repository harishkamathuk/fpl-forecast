from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from pathlib import Path

from fpl_forecast.ingest.fpl_api import FPLApiError, load_bootstrap_before_fixture

import numpy as np
import pandas as pd


VALUE_COMPONENTS = (
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "bps",
    "defensive_contribution",
)

POINT_COMPONENTS = (
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "defensive_contribution",
)

EVENT_STAT_COLUMNS = (
    *VALUE_COMPONENTS,
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "total_points",
    "played",
    "in_dreamteam",
)

POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD", 5: "AM"}


def normalize_event_live(
    *,
    season: str,
    gameweek: int,
    payload: dict[str, Any],
    retrieved_at: str,
    raw_snapshot_path: str,
    bootstrap_payload: dict[str, Any] | None = None,
    fixtures_payload: Iterable[dict[str, Any]] | None = None,
    eligible_fixture_ids: set[int] | None = None,
    resolved_fixture_blocks: Mapping[int, Iterable[dict[str, Any]]] | None = None,
    player_identity_evidence: Mapping[int, Any] | None = None,
    historical_club_assignments: Mapping[tuple[int, int], Any] | None = None,
    raw_fpl_dir: Path | None = None,
) -> pd.DataFrame:
    players = _player_lookup(bootstrap_payload)
    fixtures = _fixture_lookup(fixtures_payload)
    historical_cache = {}
    historical_sources = {}
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Official event-live payload must contain an elements list.")
    element_ids = [element.get("id") for element in elements if isinstance(element, dict)]
    if len(element_ids) != len(elements) or any(value is None for value in element_ids):
        raise ValueError("Official event-live payload contains an invalid player element.")
    if len(set(map(int, element_ids))) != len(element_ids):
        raise ValueError("Official event-live payload contains duplicate player elements.")
    rows: list[dict[str, Any]] = []
    for element in elements:
        player_id = int(element.get("id"))
        identity_evidence = (player_identity_evidence or {}).get(player_id)
        if bootstrap_payload is not None and player_id not in players and identity_evidence is None:
            raise ValueError(
                f"Official event-live player {player_id} is absent from bootstrap-static and lacks "
                "archived stable identity evidence."
            )
        player = {**players.get(player_id, {}), **_as_mapping(identity_evidence)}
        event_stats = element.get("stats") if isinstance(element.get("stats"), dict) else {}
        if resolved_fixture_blocks is not None:
            explain = list(resolved_fixture_blocks.get(player_id, ()))
        else:
            explain = list(element.get("explain") or [])
        if eligible_fixture_ids is not None:
            explain = [
                block
                for block in explain
                if block.get("fixture") is not None and int(block["fixture"]) in eligible_fixture_ids
            ]
        if not explain and eligible_fixture_ids is None and any(
            _to_number(event_stats.get(k), default=0) != 0
            for k in (*VALUE_COMPONENTS, "total_points", "starts")
        ):
            raise ValueError("Official historical outcomes lack fixture explanations.")
        for fixture_block in explain:
            fixture_id = fixture_block.get("fixture")
            if fixture_id is None or int(fixture_id) not in fixtures:
                raise ValueError(
                    f"Official event-live player {player_id} references unknown fixture {fixture_id}."
                )
            fixture = fixtures[int(fixture_id)]
            if int(fixture.get("event") or 0) != int(gameweek):
                raise ValueError(
                    f"Official event-live fixture {fixture_id} belongs to gameweek "
                    f"{fixture.get('event')}, not {gameweek}."
                )
            fixture_player = player
            assignment = (historical_club_assignments or {}).get((player_id, int(fixture_id)))
            historical_team_id = _evidence_value(assignment, "historical_team_id")
            if _evidence_value(assignment, "resolution_method") == "current_bootstrap_fixture_compatible":
                try:
                    _require_historical_team_membership(player, fixture)
                except ValueError:
                    assignment = None
            if assignment is None:
                sides = {fixture.get("team_h"), fixture.get("team_a")}
                try:
                    if player.get("team_id") not in sides:
                        raise ValueError(
                            f"Official player {player_id} cannot be reconciled to fixture {fixture_id} "
                            "from the current bootstrap snapshot."
                        )
                    _require_historical_team_membership(player, fixture)
                except ValueError:
                    if raw_fpl_dir is None:
                        raise
                    kickoff = fixture["kickoff_time"]
                    if kickoff not in historical_cache:
                        try:
                            historical_cache[kickoff] = load_bootstrap_before_fixture(
                                raw_dir=raw_fpl_dir, season=season, kickoff=kickoff,
                            )
                        except FPLApiError as exc:
                            raise ValueError(str(exc)) from exc
                    historical = historical_cache[kickoff]
                    historical_player = _player_lookup(historical.payload).get(player_id)
                    if (not historical_player or type(player.get("player_code")) is not int
                            or historical_player.get("player_code") != player["player_code"]
                            or historical_player.get("team_id") not in sides):
                        raise ValueError(
                            f"Authoritative historical team evidence cannot resolve player "
                            f"{player_id} for fixture {fixture_id}."
                        )
                    _require_historical_team_membership(historical_player, fixture)
                    fixture_player = {**player,
                                      "team_id": historical_player["team_id"],
                                      "team_uid": historical_player["team_uid"]}
                    historical_sources[historical.raw_path] = {
                        "raw_path": historical.raw_path, "metadata": historical.metadata,
                    }
                    assignment = {
                        "historical_team_id": historical_player["team_id"],
                        "current_team_id": player.get("team_id"),
                        "resolution_method": "archived_bootstrap_fixture_compatible",
                        "evidence_endpoint": "bootstrap_static",
                        "evidence_raw_snapshot_path": historical.raw_path,
                        "evidence_sha256": historical.metadata["sha256"],
                        "evidence_retrieved_at": historical.metadata["retrieved_at"],
                    }
                historical_team_id = fixture_player.get("team_id")
            fixture_sides = {fixture.get("team_h"), fixture.get("team_a")}
            if historical_team_id not in fixture_sides:
                raise ValueError("Fixture-specific historical club evidence is required.")
            rows.append(
                _normalised_fixture_row(
                    season=season,
                    gameweek=gameweek,
                    element=element,
                    fixture_block=fixture_block,
                    player=player,
                    fixture=fixture,
                    event_stats=event_stats,
                    retrieved_at=retrieved_at,
                    raw_snapshot_path=raw_snapshot_path,
                    fixture_count=len(explain),
                    historical_team_id=int(historical_team_id),
                    club_assignment=assignment,
                )
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=_output_columns())
    frame = frame[_output_columns()]
    frame = _mark_event_component_mismatches(frame)
    frame.attrs["historical_bootstrap_snapshots"] = list(historical_sources.values())
    key = ["season", "fixture_id", "player_uid"]
    if frame.duplicated(key).any():
        raise ValueError("Official event-live normalization produced duplicate player-fixture keys.")
    return frame


def audit_event_live_scoring(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    players = frame.loc[frame["entity_type"].eq("player")].copy()
    grouped = _event_totals(players)
    grouped["difference"] = grouped["reconstructed_points"] - grouped["official_event_total_points"]
    grouped["audit_status"] = np.select(
        [
            grouped["unresolved_source_limitation"].astype(bool),
            grouped["incomplete_fixture"].astype(bool),
            grouped["difference"].eq(0),
        ],
        ["unresolved_source_limitation", "incomplete_fixture", "exact_match"],
        default="genuine_reconstruction_error",
    )
    diagnostics = grouped.loc[grouped["audit_status"].ne("exact_match")].copy()
    return {
        "player_event_reconciliation": grouped.sort_values(["audit_status", "player_id"]),
        "mismatch_diagnostics": diagnostics,
        "difference_counts": grouped["difference"].value_counts().rename_axis("difference").reset_index(name="rows").sort_values("difference"),
        "status_counts": grouped["audit_status"].value_counts().rename_axis("audit_status").reset_index(name="rows"),
        "grouped_mismatches": _group_mismatches(grouped),
    }


def validate_event_live_for_forecast(
    frame: pd.DataFrame,
    *,
    information_cutoff: str | pd.Timestamp,
) -> list[str]:
    issues: list[str] = []
    if frame.duplicated(["season", "fixture_id", "player_uid"]).any():
        issues.append("duplicate player-fixture keys")
    if frame["fixture_id"].isna().any():
        issues.append("fixture IDs cannot be resolved")
    source_available = pd.to_datetime(frame["source_available_time"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(information_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if source_available.isna().any():
        issues.append("timestamps are invalid")
    if source_available.ge(cutoff).any():
        issues.append("source availability is after the forecast cutoff")
    if frame["fixture_completed"].eq(False).any():
        issues.append("incomplete fixture is being treated as final")
    audit = audit_event_live_scoring(frame)
    bad = audit["player_event_reconciliation"].loc[
        audit["player_event_reconciliation"]["audit_status"].eq("genuine_reconstruction_error")
    ]
    if not bad.empty:
        issues.append("unexplained scoring mismatches remain in eligible completed player rows")
    repeated = frame.loc[frame["fixture_count_for_player_event"].gt(1)]
    if not repeated.empty and repeated["total_points"].eq(repeated["official_event_total_points"]).any():
        issues.append("event totals are repeated across multiple fixtures")
    forbidden = {"event_points", "xP", "official_event_total_points"}.intersection(frame.columns)
    if forbidden:
        feature_cols = set(frame.attrs.get("pre_deadline_feature_columns", []))
        if forbidden.intersection(feature_cols):
            issues.append("forbidden outcome columns appear in pre-deadline model features")
    unresolved_training = frame.loc[frame["entity_type"].eq("player") & frame["unresolved_source_limitation"].astype(bool)]
    if not unresolved_training.empty:
        issues.append("unresolved source limitation rows cannot enter model training")
    return issues


def finalized_fixture_ids(fixtures: Iterable[dict[str, Any]]) -> set[int]:
    return {
        int(fixture["id"])
        for fixture in fixtures
        if fixture.get("finished") and fixture.get("finished_provisional") and fixture.get("id") is not None
    }


def _stats(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _normalised_fixture_row(
    *,
    season: str,
    gameweek: int,
    element: dict[str, Any],
    fixture_block: dict[str, Any],
    player: dict[str, Any],
    fixture: dict[str, Any] | None,
    event_stats: dict[str, Any],
    retrieved_at: str,
    raw_snapshot_path: str,
    fixture_count: int,
    historical_team_id: int,
    club_assignment: Any | None,
) -> dict[str, Any]:
    player_id = int(element["id"])
    fixture_id = fixture_block.get("fixture")
    stats = _stats(fixture_block.get("stats"))
    event_totals_assignable = fixture_count <= 1
    current_team_id = _evidence_value(club_assignment, "current_team_id")
    if current_team_id is None:
        current_team_id = player.get("current_team_id", player.get("team_id"))
    row: dict[str, Any] = {
        "season": season,
        "gameweek": gameweek,
        "fixture_id": fixture_id,
        "player_id": player_id,
        "fixture_team_id": historical_team_id,
        "player_uid": player.get("player_uid", f"official_player_id_{player_id}"),
        "player_code": player.get("player_code"),
        "player_name": player.get("player_name"),
        "entity_type": player.get("entity_type", "player"),
        "fpl_position": player.get("fpl_position"),
        "historical_team_id": historical_team_id,
        "current_team_id": current_team_id,
        "team_uid": f"official_team_{historical_team_id}",
        "opponent_uid": _opponent_uid(fixture, historical_team_id),
        "was_home": _was_home(fixture, historical_team_id),
        "historical_club_resolution_method": _evidence_value(
            club_assignment,
            "resolution_method",
            "current_bootstrap_fixture_compatible",
        ),
        "historical_club_evidence_endpoint": _evidence_value(
            club_assignment,
            "evidence_endpoint",
            "bootstrap_static",
        ),
        "historical_club_evidence_raw_snapshot_path": _evidence_value(
            club_assignment,
            "evidence_raw_snapshot_path",
            player.get("identity_raw_snapshot_path", ""),
        ),
        "historical_club_evidence_sha256": _evidence_value(
            club_assignment,
            "evidence_sha256",
            player.get("identity_sha256", ""),
        ),
        "historical_club_evidence_retrieved_at": _evidence_value(
            club_assignment,
            "evidence_retrieved_at",
            player.get("identity_retrieved_at", retrieved_at),
        ),
        "kickoff_time": fixture.get("kickoff_time") if fixture else pd.NA,
        "fixture_completed": bool(fixture and fixture.get("finished") and fixture.get("finished_provisional")),
        "exact_start": pd.NA,
        "source_available_time": retrieved_at,
        "source_available_method": "official_event_live_retrieved_after_fixture_final",
        "retrieved_at": retrieved_at,
        "raw_snapshot_path": raw_snapshot_path,
        "source": "fpl_api",
        "source_version": "event_live",
        "event_totals_assignable_to_fixture": event_totals_assignable,
        "fixture_count_for_player_event": fixture_count,
        "official_event_total_points": _to_number(event_stats.get("total_points"), default=0),
        "reconstructed_points": 0,
        "total_points": 0,
        "unresolved_source_limitation": False,
        "unresolved_reason": "",
    }
    for column in VALUE_COMPONENTS:
        row[column] = 0
        row[f"event_{column}"] = event_stats.get(column, 0)
    for column in EVENT_STAT_COLUMNS:
        row[f"event_{column}"] = event_stats.get(column, row.get(f"event_{column}", 0))
    for column in POINT_COMPONENTS:
        row[f"points_{column}"] = 0
        row[f"points_modification_{column}"] = 0

    seen_identifiers: set[str] = set()
    duplicate_identifiers: set[str] = set()
    for stat in stats:
        identifier = stat.get("identifier")
        if identifier in seen_identifiers:
            duplicate_identifiers.add(str(identifier))
        seen_identifiers.add(str(identifier))
        if identifier in VALUE_COMPONENTS:
            row[identifier] = _to_number(stat.get("value"), default=0)
        if identifier in POINT_COMPONENTS:
            row[f"points_{identifier}"] += _to_number(stat.get("points"), default=0)
            row[f"points_modification_{identifier}"] += _to_number(stat.get("points_modification"), default=0)
    if duplicate_identifiers:
        row["unresolved_source_limitation"] = True
        row["unresolved_reason"] = "duplicate_component_entries:" + ",".join(sorted(duplicate_identifiers))
    if event_totals_assignable:
        for column in VALUE_COMPONENTS:
            row[column] = _to_number(event_stats.get(column, row[column]), default=0)
        row["starts"] = _to_number(event_stats.get("starts"), default=0)
        row["exact_start"] = bool(row["starts"])
    else:
        row["starts"] = pd.NA
    row["reconstructed_points"] = sum(
        _to_number(row[f"points_{column}"], default=0) + _to_number(row[f"points_modification_{column}"], default=0)
        for column in POINT_COMPONENTS
    )
    row["total_points"] = row["reconstructed_points"]
    return row


def _mark_event_component_mismatches(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for _, group in output.groupby(["season", "gameweek", "player_id"], sort=False):
        if int(group["fixture_count_for_player_event"].max()) <= 1:
            continue
        mismatched = [
            column
            for column in VALUE_COMPONENTS
            if _to_number(group[f"event_{column}"].iloc[0], default=0)
            != _to_number(pd.to_numeric(group[column], errors="coerce").fillna(0).sum(), default=0)
        ]
        if mismatched:
            output.loc[group.index, "unresolved_source_limitation"] = True
            reason = "event_totals_not_fixture_reconciled:" + ",".join(sorted(mismatched))
            existing = output.loc[group.index, "unresolved_reason"].fillna("").astype(str)
            output.loc[group.index, "unresolved_reason"] = existing.map(
                lambda value: ";".join(part for part in (value, reason) if part)
            )
    return output


def _event_totals(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(["season", "gameweek", "player_id", "player_uid"], dropna=False)
    for key, group in grouped:
        fixture_ids = ",".join(str(value) for value in group["fixture_id"].dropna().astype(int).sort_values())
        rows.append(
            {
                "season": key[0],
                "gameweek": key[1],
                "player_id": key[2],
                "player_uid": key[3],
                "player_name": group["player_name"].iloc[0],
                "entity_type": group["entity_type"].iloc[0],
                "fpl_position": group["fpl_position"].iloc[0],
                "team_uid": group["team_uid"].iloc[0],
                "fixture_ids": fixture_ids,
                "fixture_count": int(group["fixture_id"].nunique()),
                "double_gameweek": int(group["fixture_id"].nunique()) > 1,
                "official_event_total_points": int(group["official_event_total_points"].iloc[0]),
                "reconstructed_points": int(group["reconstructed_points"].sum()),
                "minutes": int(pd.to_numeric(group["minutes"], errors="coerce").fillna(0).sum()),
                "starts": int(pd.to_numeric(group["starts"], errors="coerce").fillna(0).sum()),
                "appearance_status": _appearance_status(group),
                "unresolved_source_limitation": bool(group["unresolved_source_limitation"].any()),
                "incomplete_fixture": not bool(group["fixture_completed"].all()),
                "unresolved_reason": ";".join(sorted(set(group["unresolved_reason"].dropna().astype(str)) - {""})),
                **{
                    column: int(pd.to_numeric(group[column], errors="coerce").fillna(0).sum())
                    for column in VALUE_COMPONENTS
                    if column not in {"minutes"}
                },
                **{
                    f"points_{column}": int(pd.to_numeric(group[f"points_{column}"], errors="coerce").fillna(0).sum())
                    for column in POINT_COMPONENTS
                },
            }
        )
    return pd.DataFrame(rows)


def _group_mismatches(frame: pd.DataFrame) -> pd.DataFrame:
    mismatches = frame.loc[frame["difference"].ne(0)].copy()
    if mismatches.empty:
        return pd.DataFrame(
            columns=[
                "fpl_position",
                "entity_type",
                "appearance_status",
                "fixture_count",
                "double_gameweek",
                "difference",
                "rows",
            ]
        )
    return (
        mismatches.groupby(
            ["fpl_position", "entity_type", "appearance_status", "fixture_count", "double_gameweek", "difference"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["fpl_position", "difference"])
    )


def _player_lookup(bootstrap_payload: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    if not bootstrap_payload:
        return lookup
    team_by_id = {
        int(team["id"]): f"team_{str(team.get('name') or team.get('short_name') or team['id']).lower().replace(' ', '_')}"
        for team in bootstrap_payload.get("teams", [])
    }
    elements = bootstrap_payload.get("elements", [])
    ids = [int(element["id"]) for element in elements if element.get("id") is not None]
    if len(ids) != len(elements) or len(set(ids)) != len(ids):
        raise ValueError("bootstrap-static contains missing or duplicate player IDs.")
    for element in elements:
        player_id = int(element["id"])
        element_type = int(element.get("element_type", 0) or 0)
        player_code = element.get("code")
        lookup[player_id] = {
            "player_uid": f"player_code_{player_code}" if player_code else f"official_player_id_{player_id}",
            "player_code": player_code,
            "player_name": element.get("web_name") or element.get("second_name") or element.get("first_name"),
            "entity_type": "assistant_manager" if element_type == 5 else "player",
            "fpl_position": POSITION_BY_ELEMENT_TYPE.get(element_type),
            "team_id": element.get("team"),
            "team_join_date": element.get("team_join_date"),
            "team_uid": team_by_id.get(int(element["team"])) if element.get("team") is not None else None,
        }
    return lookup


def _fixture_lookup(fixtures_payload: Iterable[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if fixtures_payload is None:
        return {}
    fixtures = list(fixtures_payload)
    ids = [int(fixture["id"]) for fixture in fixtures if fixture.get("id") is not None]
    if len(ids) != len(fixtures) or len(set(ids)) != len(ids):
        raise ValueError("Official fixtures contain missing or duplicate fixture IDs.")
    return {int(fixture["id"]): fixture for fixture in fixtures}


def _require_historical_team_membership(player: dict[str, Any], fixture: dict[str, Any]) -> None:
    """Reject a provably wrong current-club assignment; never infer the old club.

    team_join_date is a current-club date, not an FPL registration timestamp.
    Missing dates cannot prove historical membership; this check only rejects
    known contradictions, including when the new club was the old opponent.
    """
    if player.get("team_join_date") is None:
        return
    joined = pd.Timestamp(player["team_join_date"])
    kickoff = pd.Timestamp(fixture.get("kickoff_time"))
    if pd.isna(joined) or pd.isna(kickoff):
        raise ValueError("Cannot validate historical team membership without valid dates.")
    joined = joined.tz_localize("UTC") if joined.tzinfo is None else joined.tz_convert("UTC")
    kickoff = kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
    if joined > kickoff:
        raise ValueError(
            f"Authoritative historical team evidence required: current club join date "
            f"{joined.isoformat()} is after fixture {fixture.get('id')} kickoff."
        )


def _opponent_uid(fixture: dict[str, Any] | None, player_team_id: int | None) -> str | None:
    if not fixture or player_team_id is None:
        return None
    if fixture.get("team_h") == player_team_id:
        return f"official_team_{fixture.get('team_a')}"
    if fixture.get("team_a") == player_team_id:
        return f"official_team_{fixture.get('team_h')}"
    return None


def _was_home(fixture: dict[str, Any] | None, player_team_id: int | None) -> bool | None:
    if not fixture or player_team_id is None:
        return None
    if fixture.get("team_h") == player_team_id:
        return True
    if fixture.get("team_a") == player_team_id:
        return False
    return None


def _appearance_status(group: pd.DataFrame) -> str:
    minutes = pd.to_numeric(group["minutes"], errors="coerce").fillna(0)
    starts = pd.to_numeric(group["starts"], errors="coerce").fillna(0)
    if minutes.le(0).all():
        return "did_not_play"
    if starts.gt(0).any():
        return "starter"
    return "substitute"


def _to_number(value: Any, *, default: int | float = 0) -> int | float:
    if value is None or value is pd.NA:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number.is_integer():
        return int(number)
    return number


def _output_columns() -> list[str]:
    return [
        "season",
        "gameweek",
        "fixture_id",
        "player_id",
        "fixture_team_id",
        "player_uid",
        "player_code",
        "player_name",
        "entity_type",
        "fpl_position",
        "historical_team_id",
        "current_team_id",
        "team_uid",
        "opponent_uid",
        "was_home",
        "historical_club_resolution_method",
        "historical_club_evidence_endpoint",
        "historical_club_evidence_raw_snapshot_path",
        "historical_club_evidence_sha256",
        "historical_club_evidence_retrieved_at",
        "kickoff_time",
        "fixture_completed",
        "exact_start",
        "starts",
        "source_available_time",
        "source_available_method",
        "retrieved_at",
        "raw_snapshot_path",
        "source",
        "source_version",
        "event_totals_assignable_to_fixture",
        "fixture_count_for_player_event",
        "official_event_total_points",
        "reconstructed_points",
        "total_points",
        "unresolved_source_limitation",
        "unresolved_reason",
        *VALUE_COMPONENTS,
        *[f"event_{column}" for column in EVENT_STAT_COLUMNS],
        *[f"points_{column}" for column in POINT_COMPONENTS],
        *[f"points_modification_{column}" for column in POINT_COMPONENTS],
    ]


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        return {name: getattr(value, name) for name in fields}
    return {}


def _evidence_value(value: Any, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)
