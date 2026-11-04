from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.ingest.fpl_api import (
    BOOTSTRAP_STATIC,
    EVENT_LIVE,
    FIXTURES,
    FPLApiClient,
    load_latest_fpl_snapshot,
)
from fpl_forecast.ingest.snapshots import read_json_snapshot, read_metadata
from fpl_forecast.operations.live_results import (
    audit_event_live_scoring,
    normalize_event_live,
    validate_event_live_for_forecast,
)
from fpl_forecast.operations.config import load_operational_config
from fpl_forecast.operations.temporal_clubs import HistoricalClubResolver
from fpl_forecast.xpoints.config import load_xpoints_config


@dataclass(frozen=True)
class CurrentSeasonReconstruction:
    player_history_path: Path | None
    team_history_path: Path | None
    manifest_path: Path
    player_rows: int
    team_rows: int
    event_count: int
    blank_events: tuple[int, ...]
    source_hashes: dict[str, str]


def reconstruct_completed_current_season(
    *,
    season: str,
    target_gameweek: int,
    information_cutoff: str | pd.Timestamp,
    raw_fpl_dir: Path,
    normalized_dir: Path,
    run_id: str,
    requested_gameweek: int | None,
    git_commit: str | None,
    clean_source: bool,
    refresh: bool = True,
    client: FPLApiClient | None = None,
) -> CurrentSeasonReconstruction:
    """Reconstruct every completed event before a publication target.

    The official event-live payload remains the player-result authority. Fixture
    completion, sides, scores, and event assignment remain authoritative in the
    matching fixtures snapshot. Availability is the latest retrieval time among
    every official snapshot needed to construct a row; it is never backdated.
    """

    season_dir = normalized_dir / season
    events = pd.read_parquet(season_dir / "current_events.parquet")
    fixtures = pd.read_parquet(season_dir / "current_fixtures.parquet")
    players = pd.read_parquet(season_dir / "current_players.parquet")
    teams = pd.read_parquet(season_dir / "current_teams.parquet")
    bootstrap = load_latest_fpl_snapshot(
        raw_dir=raw_fpl_dir,
        season=season,
        endpoint_name=BOOTSTRAP_STATIC,
    )
    fixture_snapshot = load_latest_fpl_snapshot(
        raw_dir=raw_fpl_dir,
        season=season,
        endpoint_name=FIXTURES,
    )
    cutoff = _utc_timestamp(information_cutoff)
    base_available_at = max(
        _utc_timestamp(bootstrap.metadata["retrieved_at"]),
        _utc_timestamp(fixture_snapshot.metadata["retrieved_at"]),
    )
    if base_available_at >= cutoff:
        raise ValueError(
            f"Official bootstrap/fixture sources were retrieved at {base_available_at.isoformat()}, "
            f"not before target cutoff {cutoff.isoformat()}."
        )
    required_events = events.loc[
        pd.to_numeric(events["gameweek"], errors="coerce").lt(target_gameweek)
    ].sort_values("gameweek")
    prior_gameweeks = pd.to_numeric(required_events["gameweek"], errors="coerce")
    if (
        prior_gameweeks.duplicated().any()
        or set(prior_gameweeks) != set(range(1, target_gameweek))
    ):
        raise ValueError("Prior official event metadata is missing or duplicated.")
    prior_fixtures = fixtures.loc[
        pd.to_numeric(fixtures["gameweek"], errors="coerce").lt(target_gameweek)
    ].copy()
    fixture_kickoff = pd.to_datetime(prior_fixtures["kickoff_time"], utc=True, errors="coerce")
    if fixture_kickoff.isna().any():
        fixture_ids = prior_fixtures.loc[fixture_kickoff.isna(), "fixture_id"].astype(int).tolist()
        raise ValueError(f"Prior official fixtures have invalid kickoff times: {fixture_ids}.")
    finalized = (
        prior_fixtures["finished"].fillna(False).astype(bool)
        & prior_fixtures["finished_provisional"].fillna(False).astype(bool)
    )
    finalized_after_cutoff = finalized & fixture_kickoff.ge(cutoff)
    if finalized_after_cutoff.any():
        fixture_ids = prior_fixtures.loc[finalized_after_cutoff, "fixture_id"].astype(int).tolist()
        raise ValueError(
            f"Prior official fixtures are marked finalized at or after the forecast cutoff: {fixture_ids}."
        )
    eligible_fixtures = prior_fixtures.loc[finalized & fixture_kickoff.lt(cutoff)].copy()
    missing_scores = eligible_fixtures[["team_h_score", "team_a_score"]].isna().any(axis=1)
    if missing_scores.any():
        fixture_ids = eligible_fixtures.loc[missing_scores, "fixture_id"].astype(int).tolist()
        raise ValueError(f"Finalized prior fixtures lack completed scores: {fixture_ids}.")
    deferred_fixtures = prior_fixtures.loc[~finalized].copy()
    source_hashes = {
        BOOTSTRAP_STATIC: str(bootstrap.metadata["sha256"]),
        FIXTURES: str(fixture_snapshot.metadata["sha256"]),
    }
    snapshot_entries: dict[str, dict[str, Any]] = {
        BOOTSTRAP_STATIC: _snapshot_entry(bootstrap.metadata, bootstrap.raw_path),
        FIXTURES: _snapshot_entry(fixture_snapshot.metadata, fixture_snapshot.raw_path),
    }
    live_frames: list[pd.DataFrame] = []
    event_summaries: list[dict[str, Any]] = []
    blank_events: list[int] = []
    api_client = client or FPLApiClient(raw_dir=raw_fpl_dir)
    resolver = HistoricalClubResolver(
        season=season,
        bootstrap_payload=bootstrap.payload,
        bootstrap_metadata=bootstrap.metadata,
        bootstrap_raw_path=bootstrap.raw_path,
        fixtures_payload=fixture_snapshot.payload,
        raw_fpl_dir=raw_fpl_dir,
        normalized_dir=normalized_dir,
        client=api_client,
        information_cutoff=cutoff,
        allow_network=refresh,
        source_context={
            "requested_target_gameweek": requested_gameweek,
            "resolved_target_gameweek": target_gameweek,
            "publication_run_id": run_id,
            "git_commit": git_commit,
            "clean_source": clean_source,
            "source_mode": "official_current_season",
        },
    )

    for event in required_events.itertuples(index=False):
        gameweek = int(event.gameweek)
        event_fixtures = fixtures.loc[
            pd.to_numeric(fixtures["gameweek"], errors="coerce").eq(gameweek)
        ].copy()
        included_event_fixtures = eligible_fixtures.loc[
            pd.to_numeric(eligible_fixtures["gameweek"], errors="coerce").eq(gameweek)
        ].copy()
        deferred_event_fixtures = deferred_fixtures.loc[
            pd.to_numeric(deferred_fixtures["gameweek"], errors="coerce").eq(gameweek)
        ].copy()
        included_fixture_ids = set(included_event_fixtures["fixture_id"].astype(int))
        deferred_fixture_ids = sorted(deferred_event_fixtures["fixture_id"].astype(int).tolist())
        if event_fixtures.empty:
            blank_events.append(gameweek)
        if not included_fixture_ids and not event_fixtures.empty:
            event_summaries.append(
                {
                    "gameweek": gameweek,
                    "fixture_count": int(len(event_fixtures)),
                    "included_fixture_count": 0,
                    "included_fixture_ids": [],
                    "deferred_fixture_count": int(len(deferred_fixture_ids)),
                    "deferred_fixture_ids": deferred_fixture_ids,
                    "player_fixture_rows": 0,
                    "blank_event": False,
                    "event_finished": bool(event.finished),
                    "event_data_checked": bool(event.data_checked),
                    "event_live_requested": False,
                    "source_available_time": None,
                    "element_summary_player_ids": [],
                }
            )
            continue
        summaries_before = set(resolver.summary_player_ids)
        record = api_client.fetch_event_live(
            season=season,
            gameweek=gameweek,
            refresh=refresh,
            offline=not refresh,
            extra_metadata={
                "requested_target_gameweek": requested_gameweek,
                "resolved_target_gameweek": target_gameweek,
                "publication_run_id": run_id,
                "git_commit": git_commit,
                "clean_source": clean_source,
                "source_mode": "official_current_season",
            },
        )
        live_metadata = read_metadata(record.raw_path)
        endpoint = f"{EVENT_LIVE}_{gameweek}"
        source_hashes[endpoint] = record.checksum_sha256
        snapshot_entries[endpoint] = _snapshot_entry(live_metadata, str(record.raw_path))
        available_at = max(base_available_at, _utc_timestamp(record.retrieved_at))
        if available_at >= cutoff:
            raise ValueError(
                f"Official GW{gameweek} sources were retrieved at {available_at.isoformat()}, "
                f"not before target cutoff {cutoff.isoformat()}."
            )
        resolved = resolver.resolve_event(
            gameweek=gameweek,
            payload=read_json_snapshot(record.raw_path),
            eligible_fixture_ids=included_fixture_ids,
        )
        live = normalize_event_live(
            season=season,
            gameweek=gameweek,
            payload=read_json_snapshot(record.raw_path),
            retrieved_at=record.retrieved_at,
            raw_snapshot_path=str(record.raw_path),
            bootstrap_payload=bootstrap.payload,
            fixtures_payload=fixture_snapshot.payload,
            eligible_fixture_ids=included_fixture_ids,
            resolved_fixture_blocks=resolved.fixture_blocks,
            player_identity_evidence=resolved.identities,
            historical_club_assignments=resolved.club_assignments,
            raw_fpl_dir=raw_fpl_dir,
        )
        for source in live.attrs.get("historical_bootstrap_snapshots", []):
            endpoint = "historical_bootstrap_" + source["metadata"]["sha256"]
            source_hashes[endpoint] = source["metadata"]["sha256"]
            snapshot_entries[endpoint] = _snapshot_entry(source["metadata"], source["raw_path"])
        if event_fixtures.empty:
            if not live.empty:
                raise ValueError(f"Official blank GW{gameweek} unexpectedly produced player-fixture rows.")
        elif included_fixture_ids and live.empty:
            raise ValueError(f"Official event-live GW{gameweek} contains no player-fixture rows.")
        if not live.empty:
            club_evidence_time = pd.to_datetime(
                live["historical_club_evidence_retrieved_at"],
                utc=True,
                errors="coerce",
                format="mixed",
            )
            if club_evidence_time.isna().any():
                bad = live.loc[
                    club_evidence_time.isna(),
                    [
                        "player_id",
                        "fixture_id",
                        "historical_club_resolution_method",
                        "historical_club_evidence_endpoint",
                        "historical_club_evidence_retrieved_at",
                    ],
                ].head(10).to_dict("records")
                raise ValueError(
                    f"Official event-live GW{gameweek} lacks club-evidence timestamps: {bad}."
                )
            base_available = pd.Series(available_at, index=live.index)
            live["source_available_time"] = pd.concat(
                [base_available.rename("base"), club_evidence_time.rename("club")],
                axis=1,
            ).max(axis=1)
            live["source_available_method"] = live[
                "historical_club_resolution_method"
            ].map(
                lambda method: (
                    "latest_of_bootstrap_fixtures_event_live_and_historical_club_evidence"
                    if method != "current_bootstrap_fixture_compatible"
                    else "latest_of_bootstrap_fixtures_and_event_live_retrieval"
                )
            )
            issues = validate_event_live_for_forecast(live, information_cutoff=cutoff)
            if issues:
                raise ValueError(f"Official event-live GW{gameweek} failed safety checks: {', '.join(issues)}")
            audit = audit_event_live_scoring(live)["player_event_reconciliation"]
            unacceptable = audit.loc[audit["audit_status"].ne("exact_match")]
            if not unacceptable.empty:
                raise ValueError(
                    f"Official event-live GW{gameweek} has {len(unacceptable)} unresolved player-event totals."
                )
            live_frames.append(live)
        event_summaries.append(
            {
                "gameweek": gameweek,
                "fixture_count": int(len(event_fixtures)),
                "included_fixture_count": int(len(included_fixture_ids)),
                "included_fixture_ids": sorted(included_fixture_ids),
                "deferred_fixture_count": int(len(deferred_fixture_ids)),
                "deferred_fixture_ids": deferred_fixture_ids,
                "player_fixture_rows": int(len(live)),
                "blank_event": bool(event_fixtures.empty),
                "event_finished": bool(event.finished),
                "event_data_checked": bool(event.data_checked),
                "event_live_requested": True,
                "source_available_time": available_at.isoformat(),
                "event_live_sha256": record.checksum_sha256,
                "event_live_bytes": record.content_length,
                "event_live_raw_snapshot": str(record.raw_path),
                "element_summary_player_ids": sorted(
                    resolver.summary_player_ids.difference(summaries_before)
                ),
            }
        )

    source_hashes.update(resolver.source_hashes)
    snapshot_entries.update(resolver.snapshot_entries)

    live_results = pd.concat(live_frames, ignore_index=True) if live_frames else pd.DataFrame()
    team_identities = _current_team_identities(teams, normalized_dir=normalized_dir)
    player_history = build_current_player_fixture_history(
        fixtures=fixtures,
        live_results=live_results,
        players=players,
        team_identities=team_identities,
        events=events,
    )
    team_history = build_current_team_fixture_history(
        fixtures=fixtures,
        team_identities=team_identities,
        events=events,
        event_summaries=event_summaries,
    )
    if not player_history.empty:
        _require_rows_before_target(player_history, target_gameweek=target_gameweek, cutoff=cutoff)
    if not team_history.empty:
        _require_rows_before_target(team_history, target_gameweek=target_gameweek, cutoff=cutoff)

    expected_fixtures = set(eligible_fixtures["fixture_id"])
    for name, frame, fixture_column in (
        ("player", player_history, "fixture_id"),
        ("team", team_history, "source_fixture_id"),
    ):
        actual_fixtures = set(frame[fixture_column]) if not frame.empty else set()
        if actual_fixtures != expected_fixtures:
            raise ValueError(f"Reconstructed {name} history does not cover prior official fixtures.")

    player_path = season_dir / "current_player_fixture_history.parquet"
    team_path = season_dir / "current_team_fixture_history.parquet"
    if target_gameweek > 1:
        player_history.to_parquet(player_path, index=False)
        team_history.to_parquet(team_path, index=False)
    else:
        player_path = None
        team_path = None
    manifest_path = season_dir / "current_season_reconstruction.json"
    operational_config = load_operational_config()
    xpoints_config = load_xpoints_config()
    manifest = {
        "schema_version": "current_season_reconstruction_v2",
        "season": season,
        "requested_target_gameweek": requested_gameweek,
        "resolved_target_gameweek": target_gameweek,
        "information_cutoff": cutoff.isoformat(),
        "source_mode": "official_current_season",
        "publication_run_id": run_id,
        "git_commit": git_commit,
        "clean_source": clean_source,
        "events": event_summaries,
        "blank_events": blank_events,
        "deferred_fixture_ids": sorted(deferred_fixtures["fixture_id"].astype(int).tolist()),
        "historical_club_resolutions": resolver.resolution_records,
        "excluded_cross_season_bootstrap_snapshots": resolver.excluded_archive_snapshots,
        "element_summary_player_ids": sorted(resolver.summary_player_ids),
        "player_fixture_rows": int(len(player_history)),
        "team_fixture_rows": int(len(team_history)),
        "player_history_path": str(player_path) if player_path else None,
        "team_history_path": str(team_path) if team_path else None,
        "source_hashes": source_hashes,
        "official_snapshots": snapshot_entries,
        "temporal_policy": "source_available_time < information_cutoff",
        "source_available_policy": (
            "latest retrieval among all official inputs required for each player-fixture row"
        ),
        "fixture_eligibility_policy": (
            "finished and finished_provisional and kickoff_time < information_cutoff; "
            "unfinished or postponed earlier-event fixtures are deferred without blocking"
        ),
        "configuration": {
            "operational_schema_version": operational_config.schema_version,
            "frontend_schema_version": operational_config.frontend_schema_version,
            "default_models": operational_config.default_models,
            "xpoints_simulation_version": xpoints_config.simulation_version,
            "xpoints_contract_version": xpoints_config.model_contract_version,
            "xpoints_draw_count": xpoints_config.draw_count,
            "xpoints_master_seed": xpoints_config.random_seed,
            "xpoints_seed_derivation_policy": xpoints_config.seed_derivation_policy,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return CurrentSeasonReconstruction(
        player_history_path=player_path,
        team_history_path=team_path,
        manifest_path=manifest_path,
        player_rows=int(len(player_history)),
        team_rows=int(len(team_history)),
        event_count=int(len(required_events)),
        blank_events=tuple(blank_events),
        source_hashes=source_hashes,
    )


def build_current_player_fixture_history(
    *,
    fixtures: pd.DataFrame,
    live_results: pd.DataFrame,
    players: pd.DataFrame,
    team_identities: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if live_results.empty:
        return pd.DataFrame()
    key = ["season", "player_id", "fixture_id"]
    if live_results.duplicated(key).any():
        duplicates = live_results.loc[live_results.duplicated(key, keep=False)].copy()
        conflicting = (
            duplicates.astype("string")
            .groupby(key, dropna=False)
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicting.any():
            raise ValueError("Current event-live rows contain conflicting duplicate player-fixture keys.")
        live_results = live_results.drop_duplicates(key).copy()
    else:
        live_results = live_results.copy()
    for column in (
        "player_uid",
        "player_code",
        "player_name",
        "fpl_position",
        "entity_type",
        "historical_team_id",
        "current_team_id",
    ):
        if column not in live_results:
            live_results[column] = pd.NA
    players = players.copy()
    if "entity_type" not in players:
        players["entity_type"] = players["position"].map(
            lambda value: "assistant_manager" if str(value) == "AM" else "player"
        )
    for column in ("player_code", "web_name", "team_id", "position", "price_tenths", "entity_type"):
        if column not in players:
            players[column] = pd.NA
    player_meta = players[
        ["player_id", "player_code", "web_name", "team_id", "position", "price_tenths", "entity_type"]
    ].rename(
        columns={
            "player_code": "current_player_code",
            "web_name": "current_player_name",
            "team_id": "latest_current_team_id",
            "position": "current_position",
            "entity_type": "current_entity_type",
        }
    )
    fixtures = fixtures.copy()
    for column in (
        "home_team_id",
        "away_team_id",
        "team_h_score",
        "team_a_score",
        "raw_snapshot_path",
    ):
        if column not in fixtures:
            fixtures[column] = pd.NA
    fixture_columns = [
        "fixture_id",
        "gameweek",
        "kickoff_time",
        "finished",
        "finished_provisional",
        "home_team_id",
        "away_team_id",
        "team_h_score",
        "team_a_score",
        "raw_snapshot_path",
    ]
    fixture_meta = fixtures[fixture_columns].rename(
        columns={
            "kickoff_time": "fixture_kickoff_time",
            "raw_snapshot_path": "fixtures_raw_snapshot_path",
        }
    )
    frame = live_results.merge(player_meta, on="player_id", how="left", validate="many_to_one")
    frame = frame.merge(fixture_meta, on=["fixture_id", "gameweek"], how="left", validate="many_to_one")
    live_kickoff = (
        pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
        if "kickoff_time" in frame
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    )
    fixture_kickoff = pd.to_datetime(
        frame["fixture_kickoff_time"], utc=True, errors="coerce"
    )
    if fixture_kickoff.isna().any():
        bad = frame.loc[fixture_kickoff.isna(), ["player_id", "fixture_id"]].head(5).to_dict("records")
        raise ValueError(f"Current player-fixture history has invalid fixture kickoff evidence: {bad}")
    kickoff_conflicts = live_kickoff.notna() & live_kickoff.ne(fixture_kickoff)
    if kickoff_conflicts.any():
        bad = frame.loc[
            kickoff_conflicts,
            ["player_id", "fixture_id", "kickoff_time", "fixture_kickoff_time"],
        ].head(5).to_dict("records")
        raise ValueError(f"Event-live and fixture kickoff evidence conflict: {bad}")
    frame["kickoff_time"] = fixture_kickoff
    frame = frame.drop(columns="fixture_kickoff_time")
    frame["player_code"] = frame["player_code"].fillna(frame["current_player_code"])
    derived_uid = pd.Series(pd.NA, index=frame.index, dtype="string")
    coded = frame["player_code"].notna()
    derived_uid.loc[coded] = (
        "player_code_" + frame.loc[coded, "player_code"].astype("Int64").astype(str)
    )
    frame["player_uid"] = frame["player_uid"].fillna(derived_uid)
    frame["player_name"] = frame["player_name"].fillna(frame["current_player_name"])
    frame["fpl_position"] = frame["fpl_position"].fillna(frame["current_position"])
    frame["entity_type"] = frame["entity_type"].fillna(frame["current_entity_type"])
    frame["current_team_id"] = frame["current_team_id"].fillna(frame["latest_current_team_id"])
    if "fixture_team_id" in frame:
        if frame["fixture_team_id"].isna().any():
            raise ValueError("Current player-fixture history lacks resolved historical team IDs.")
        conflict = frame["historical_team_id"].notna() & frame["historical_team_id"].astype("Int64").ne(frame["fixture_team_id"].astype("Int64"))
        if conflict.any():
            raise ValueError("Historical team assignments contradict fixture-side evidence.")
        frame["historical_team_id"] = frame["historical_team_id"].fillna(frame["fixture_team_id"])
    frame["historical_team_id"] = frame["historical_team_id"].fillna(frame["latest_current_team_id"])
    frame["team_id"] = frame["historical_team_id"]
    both_codes = frame["player_code"].notna() & frame["current_player_code"].notna()
    conflicting_codes = both_codes & frame["player_code"].astype("Int64").ne(
        frame["current_player_code"].astype("Int64")
    )
    if conflicting_codes.any():
        bad = frame.loc[conflicting_codes, ["player_id", "fixture_id"]].head(5).to_dict("records")
        raise ValueError(f"Current and archived stable player codes conflict: {bad}")
    current_rows = frame["latest_current_team_id"].notna()
    conflicting_current_team = current_rows & frame["current_team_id"].notna() & frame[
        "current_team_id"
    ].astype("Int64").ne(frame["latest_current_team_id"].astype("Int64"))
    if conflicting_current_team.any():
        bad = frame.loc[
            conflicting_current_team,
            ["player_id", "fixture_id", "current_team_id", "latest_current_team_id"],
        ].head(5).to_dict("records")
        raise ValueError(f"Current player team evidence changed during reconstruction: {bad}")
    if frame[["player_uid", "player_name", "fpl_position", "historical_team_id"]].isna().any().any():
        raise ValueError("Current player-fixture history has unresolved historical identities.")
    if team_identities is not None and frame[["home_team_id", "away_team_id"]].isna().any().any():
        raise ValueError("Current player-fixture history has unresolved fixture identities.")
    frame = frame.loc[frame["entity_type"].eq("player")].copy()
    frame["entity_type"] = "player"
    sides_known = frame[["home_team_id", "away_team_id"]].notna().all(axis=1)
    participates = frame["historical_team_id"].eq(frame["home_team_id"].fillna(-1)) | frame[
        "historical_team_id"
    ].eq(
        frame["away_team_id"].fillna(-1)
    )
    if (sides_known & ~participates).any():
        bad = frame.loc[
            sides_known & ~participates,
            ["player_id", "fixture_id", "historical_team_id", "current_team_id"],
        ].head(5).to_dict("records")
        raise ValueError(f"Historical player clubs conflict with completed fixtures: {bad}")
    if team_identities is not None:
        identity = team_identities[["team_id", "team_uid"]].drop_duplicates("team_id")
        historical_team = identity.rename(
            columns={"team_id": "historical_team_id", "team_uid": "historical_team_uid"}
        )
        current_team = identity.rename(
            columns={"team_id": "current_team_id", "team_uid": "current_team_uid"}
        )
        home = identity.rename(columns={"team_id": "home_team_id", "team_uid": "home_team_uid"})
        away = identity.rename(columns={"team_id": "away_team_id", "team_uid": "away_team_uid"})
        frame = frame.merge(historical_team, on="historical_team_id", how="left", validate="many_to_one")
        frame = frame.merge(current_team, on="current_team_id", how="left", validate="many_to_one")
        frame = frame.merge(home, on="home_team_id", how="left", validate="many_to_one")
        frame = frame.merge(away, on="away_team_id", how="left", validate="many_to_one")
        if frame[["historical_team_uid", "home_team_uid", "away_team_uid"]].isna().any().any():
            raise ValueError("Current player-fixture history lacks stable historical team identities.")
        frame["opponent_team_uid"] = frame["away_team_uid"].where(
            frame["historical_team_id"].eq(frame["home_team_id"]), frame["home_team_uid"]
        )
        frame["player_team_uid"] = frame["historical_team_uid"]
        frame["team_uid"] = frame["historical_team_uid"]
        frame["opponent_uid"] = frame["opponent_team_uid"]
    frame["stable_fixture_uid"] = (
        frame["season"].astype(str) + ":official_fixture_" + frame["fixture_id"].astype("Int64").astype(str)
    )
    frame["fixture_key"] = frame["stable_fixture_uid"]
    if events is not None:
        deadlines = events[["gameweek", "deadline_time"]].copy()
        deadlines["information_cutoff"] = pd.to_datetime(deadlines["deadline_time"], utc=True)
        frame = frame.merge(
            deadlines[["gameweek", "information_cutoff"]],
            on="gameweek",
            how="left",
            validate="many_to_one",
        )
    if frame.duplicated(key).any():
        raise ValueError("Current player-fixture history contains duplicate keys after reconciliation.")
    return frame.sort_values(["gameweek", "fixture_id", "player_id"]).reset_index(drop=True)


def build_current_team_fixture_history(
    *,
    fixtures: pd.DataFrame,
    team_identities: pd.DataFrame,
    events: pd.DataFrame,
    event_summaries: list[dict[str, Any]],
) -> pd.DataFrame:
    if not event_summaries:
        return pd.DataFrame()
    included_fixture_ids = {
        int(fixture_id)
        for item in event_summaries
        for fixture_id in item.get("included_fixture_ids", [])
    }
    frame = fixtures.loc[
        pd.to_numeric(fixtures["fixture_id"], errors="coerce").isin(included_fixture_ids)
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    home = team_identities[["team_id", "team_uid", "team_name"]].rename(
        columns={"team_id": "home_team_id", "team_uid": "home_team_uid", "team_name": "home_team_name"}
    )
    away = team_identities[["team_id", "team_uid", "team_name"]].rename(
        columns={"team_id": "away_team_id", "team_uid": "away_team_uid", "team_name": "away_team_name"}
    )
    frame = frame.merge(home, on="home_team_id", how="left", validate="many_to_one")
    frame = frame.merge(away, on="away_team_id", how="left", validate="many_to_one")
    if frame[["home_team_uid", "away_team_uid"]].isna().any().any():
        raise ValueError("Current team fixtures lack stable team identities.")
    availability = {
        int(item["gameweek"]): item["source_available_time"]
        for item in event_summaries
        if item.get("source_available_time")
    }
    deadlines = events.set_index("gameweek")["deadline_time"].to_dict()
    frame["stable_fixture_uid"] = (
        frame["season"].astype(str) + ":official_fixture_" + frame["fixture_id"].astype("Int64").astype(str)
    )
    frame["source_fixture_id"] = frame["fixture_id"]
    frame["source_home_team_id"] = frame["home_team_id"]
    frame["source_away_team_id"] = frame["away_team_id"]
    frame["home_goals"] = frame["team_h_score"]
    frame["away_goals"] = frame["team_a_score"]
    frame["source_available_time"] = frame["gameweek"].map(availability)
    frame["source_available_method"] = "latest_of_bootstrap_fixtures_and_event_live_retrieval"
    frame["information_cutoff"] = pd.to_datetime(frame["gameweek"].map(deadlines), utc=True)
    frame["fixture_completed"] = frame["finished"].astype(bool) & frame["finished_provisional"].astype(bool)
    frame["result_valid"] = frame["fixture_completed"] & frame[["home_goals", "away_goals"]].notna().all(axis=1)
    if frame.duplicated(["season", "stable_fixture_uid"]).any():
        raise ValueError("Current team fixture history contains duplicate fixture observations.")
    if not frame["result_valid"].all():
        raise ValueError("Current team fixture history contains incomplete scores or finality flags.")
    return frame[
        [
            "season",
            "gameweek",
            "stable_fixture_uid",
            "source_fixture_id",
            "home_team_uid",
            "away_team_uid",
            "source_home_team_id",
            "source_away_team_id",
            "home_team_name",
            "away_team_name",
            "kickoff_time",
            "information_cutoff",
            "source_available_time",
            "source_available_method",
            "finished",
            "fixture_completed",
            "result_valid",
            "home_goals",
            "away_goals",
            "source_version",
            "raw_snapshot_path",
        ]
    ].sort_values(["gameweek", "source_fixture_id"])


def _current_team_identities(teams: pd.DataFrame, *, normalized_dir: Path) -> pd.DataFrame:
    # Reuse the operational identity bridge so target and completed fixtures share
    # exactly the same promoted-team fallback and alias policy.
    from fpl_forecast.operations.model_chain import _current_team_identity

    season = str(teams["season"].dropna().iloc[0])
    return _current_team_identity(teams, normalized_dir=normalized_dir, season=season)


def _snapshot_entry(metadata: dict[str, Any], raw_path: str) -> dict[str, Any]:
    return {
        "endpoint": metadata.get("source_url"),
        "retrieved_at": metadata.get("retrieved_at"),
        "sha256": metadata.get("sha256"),
        "bytes": metadata.get("content_length"),
        "source_mode": "official_current_season",
        "raw_snapshot_path": raw_path,
    }


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _require_rows_before_target(
    frame: pd.DataFrame,
    *,
    target_gameweek: int,
    cutoff: pd.Timestamp,
) -> None:
    gameweeks = pd.to_numeric(frame["gameweek"], errors="coerce")
    available = pd.to_datetime(frame["source_available_time"], utc=True, errors="coerce")
    if gameweeks.isna().any() or gameweeks.ge(target_gameweek).any():
        raise ValueError("Current-season training history contains target or future gameweeks.")
    if available.isna().any() or available.ge(cutoff).any():
        raise ValueError("Current-season training history violates the target information cutoff.")
