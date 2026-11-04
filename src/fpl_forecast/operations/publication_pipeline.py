from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, RAW_FPL_API_DIR, RAW_VAASTAV_DIR
from fpl_forecast.features.leakage import audit_leakage
from fpl_forecast.ingest.fpl_api import FPLApiClient
from fpl_forecast.ingest.vaastav import VaastavIngestor
from fpl_forecast.normalize.current import normalize_current
from fpl_forecast.normalize.historical import normalize_historical
from fpl_forecast.operations.config import load_operational_config
from fpl_forecast.operations.launch import check_season_launch
from fpl_forecast.operations.target_gameweek import (
    PublicationError,
    TargetGameweekResolution,
    resolve_target_gameweek,
)
from fpl_forecast.operations.current_panel import reconstruct_completed_current_season
from fpl_forecast.operations.orchestrator import RefreshResult, _source_state, refresh_operational
from fpl_forecast.panel.build import build_identities, build_panel
from fpl_forecast.validation.data_quality import validate_all


PUBLICATION_HISTORICAL_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
PUBLICATION_VAASTAV_REVISION = "f2090d378ebd1b0c3d14884770dde95f38c50a0d"
PUBLIC_ARTIFACTS = (
    "operational_status.json",
    "player_gameweek_projections.csv",
    "optimized_squad.csv",
    "optimized_lineup.csv",
    "model_comparison.csv",
    "data_freshness.json",
    "run_manifest.json",
)
LOCAL_PATH_PATTERN = re.compile(r"(?:^|[\"',\s])/(?:Users|home|private|var|tmp)/")
PLACEHOLDER_PATTERN = re.compile(r"\b(?:New Player|Player \d+|Team \d+)\b", re.IGNORECASE)
MOCK_PATTERN = re.compile(r"\b(?:mock|demo data|synthetic)\b", re.IGNORECASE)
SECRET_PATTERN = re.compile(r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Bearer\s+[A-Za-z0-9._-]+)")
PUBLICATION_MAX_AGE_HOURS = 6


@dataclass(frozen=True)
class PublicationPreparation:
    season: str
    historical_seasons: tuple[str, ...]
    source_revision: str
    target: TargetGameweekResolution
    historical_rows: dict[str, int]
    stage_seconds: dict[str, float]
    launch_state: str
    source_hashes: dict[str, str]
    run_id: str = "unassigned"
    current_season_player_rows: int = 0
    current_season_team_rows: int = 0
    reconstructed_events: int = 0
    blank_events: tuple[int, ...] = ()
    player_history_path: str | None = None
    team_history_path: str | None = None
    reconstruction_manifest_path: str | None = None
    source_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicationValidation:
    run_id: str
    season: str
    gameweek: int
    gates: dict[str, str]
    public_inventory: dict[str, dict[str, Any]]
    audit_path: Path


def prepare_publication_data(
    *,
    season: str,
    requested_gameweek: int | None = None,
    historical_seasons: tuple[str, ...] = PUBLICATION_HISTORICAL_SEASONS,
    revision: str = PUBLICATION_VAASTAV_REVISION,
    raw_fpl_dir: Path = RAW_FPL_API_DIR,
    raw_vaastav_dir: Path = RAW_VAASTAV_DIR,
    normalized_dir: Path = NORMALIZED_DIR,
    refresh: bool = True,
    now: datetime | None = None,
    run_id: str = "publication_preparation",
) -> PublicationPreparation:
    if not historical_seasons:
        raise PublicationError("Publication preparation requires at least one historical season.")
    timings: dict[str, float] = {}
    historical_rows: dict[str, int] = {}

    started = time.perf_counter()
    ingestor = VaastavIngestor(raw_dir=raw_vaastav_dir)
    for historical_season in historical_seasons:
        ingestor.ingest_season(
            season=historical_season,
            revision=revision,
            refresh=refresh,
        )
        normalize_historical(
            season=historical_season,
            raw_dir=raw_vaastav_dir,
            normalized_dir=normalized_dir,
        )
        history_path = normalized_dir / historical_season / "historical_player_fixtures.parquet"
        if not history_path.exists():
            raise PublicationError(f"Historical reconstruction did not create {history_path}.")
        history = pd.read_parquet(history_path)
        _validate_historical_publication_frame(history, season=historical_season)
        historical_rows[historical_season] = len(history)
    timings["historical_reconstruction"] = time.perf_counter() - started

    started = time.perf_counter()
    build_identities(seasons=list(historical_seasons), normalized_dir=normalized_dir)
    build_panel(seasons=list(historical_seasons), normalized_dir=normalized_dir)
    _validate_phase2_publication_artifacts(
        normalized_dir=normalized_dir,
        historical_seasons=historical_seasons,
    )
    data_validation = validate_all(
        normalized_dir=normalized_dir,
        raw_vaastav_dir=raw_vaastav_dir,
        seasons=historical_seasons,
    )
    if data_validation.errors:
        messages = "; ".join(issue.message for issue in data_validation.errors)
        raise PublicationError(f"Historical data validation failed: {messages}")
    leakage = audit_leakage(seasons=list(historical_seasons), normalized_dir=normalized_dir)
    if leakage.errors:
        messages = "; ".join(issue.message for issue in leakage.errors)
        raise PublicationError(f"Historical leakage audit failed: {messages}")
    timings["phase2_and_validation"] = time.perf_counter() - started

    started = time.perf_counter()
    source_state = _source_state()
    FPLApiClient(raw_dir=raw_fpl_dir).snapshot_current(
        season=season,
        refresh=refresh,
        offline=False,
        extra_metadata={
            "requested_target_gameweek": requested_gameweek,
            "publication_run_id": run_id,
            "git_commit": source_state.get("commit"),
            "clean_source": not bool(source_state.get("dirty")),
            "source_mode": "official_current_season",
        },
    )
    normalize_current(
        season=season,
        raw_dir=raw_fpl_dir,
        normalized_dir=normalized_dir,
    )
    launch = check_season_launch(
        season=season,
        raw_dir=raw_fpl_dir,
        offline=True,
    )
    if launch.status.state.value != "READY_TO_REFRESH":
        raise PublicationError(
            f"Official launch gate failed with {launch.status.state.value}: {launch.status.reason}"
        )
    timings["official_current_refresh"] = time.perf_counter() - started

    events = pd.read_parquet(normalized_dir / season / "current_events.parquet")
    fixtures = pd.read_parquet(normalized_dir / season / "current_fixtures.parquet")
    target = resolve_target_gameweek(
        season=season,
        events=events,
        fixtures=fixtures,
        requested_gameweek=requested_gameweek,
        now=now,
    )
    started = time.perf_counter()
    if target.gameweek > 1:
        reconstruction = reconstruct_completed_current_season(
            season=season,
            target_gameweek=target.gameweek,
            information_cutoff=target.deadline_time,
            raw_fpl_dir=raw_fpl_dir,
            normalized_dir=normalized_dir,
            run_id=run_id,
            requested_gameweek=requested_gameweek,
            git_commit=source_state.get("commit"),
            clean_source=not bool(source_state.get("dirty")),
            refresh=refresh,
        )
        source_hashes = reconstruction.source_hashes
    else:
        reconstruction = None
        source_hashes = _snapshot_source_hashes(raw_fpl_dir, season=season)
    timings["current_season_reconstruction"] = time.perf_counter() - started
    return PublicationPreparation(
        season=season,
        historical_seasons=historical_seasons,
        source_revision=revision,
        target=target,
        historical_rows=historical_rows,
        stage_seconds={key: round(value, 3) for key, value in timings.items()},
        launch_state=launch.status.state.value,
        source_hashes=source_hashes,
        run_id=run_id,
        current_season_player_rows=reconstruction.player_rows if reconstruction else 0,
        current_season_team_rows=reconstruction.team_rows if reconstruction else 0,
        reconstructed_events=reconstruction.event_count if reconstruction else 0,
        blank_events=reconstruction.blank_events if reconstruction else (),
        player_history_path=(
            str(reconstruction.player_history_path)
            if reconstruction and reconstruction.player_history_path
            else None
        ),
        team_history_path=(
            str(reconstruction.team_history_path)
            if reconstruction and reconstruction.team_history_path
            else None
        ),
        reconstruction_manifest_path=str(reconstruction.manifest_path) if reconstruction else None,
        source_state=source_state,
    )


def run_official_publication_forecast(
    *,
    preparation: PublicationPreparation,
    run_id: str,
    raw_fpl_dir: Path = RAW_FPL_API_DIR,
    normalized_dir: Path = NORMALIZED_DIR,
) -> RefreshResult:
    if preparation.run_id not in {"unassigned", "publication_preparation", run_id}:
        raise PublicationError(
            f"Preparation run ID {preparation.run_id!r} does not match forecast run ID {run_id!r}."
        )
    authoritative_deadline = pd.to_datetime(
        preparation.target.deadline_time,
        utc=True,
        errors="coerce",
    )
    if not isinstance(authoritative_deadline, pd.Timestamp) or pd.isna(authoritative_deadline):
        raise PublicationError("Prepared official target deadline is missing or malformed.")
    completed_players = _read_optional_parquet(preparation.player_history_path)
    completed_teams = _read_optional_parquet(preparation.team_history_path)
    if len(completed_players) != preparation.current_season_player_rows:
        raise PublicationError("Prepared current-season player row count changed before forecasting.")
    if len(completed_teams) != preparation.current_season_team_rows:
        raise PublicationError("Prepared current-season team row count changed before forecasting.")
    result = refresh_operational(
        season=preparation.season,
        offline=True,
        mock_launch=False,
        force=True,
        run_id=run_id,
        target_gameweek=preparation.target.gameweek,
        normalized_dir=normalized_dir,
        raw_fpl_dir=raw_fpl_dir,
        authoritative_publication=True,
        completed_player_fixtures=completed_players,
        completed_team_fixtures=completed_teams,
        information_cutoff=authoritative_deadline,
    )
    if result.status.state.value != "SUCCEEDED" or result.run_dir is None:
        raise PublicationError(
            f"Official forecast did not succeed: {result.status.state.value}: {result.status.reason}"
        )
    return result


def validate_publication_candidate(
    *,
    run_dir: Path,
    preparation: PublicationPreparation,
    audit_dir: Path,
    public_dir: Path | None = None,
    git_sha: str | None = None,
    now: datetime | None = None,
) -> PublicationValidation:
    config = load_operational_config()
    run_dir = run_dir.resolve()
    missing = [name for name in PUBLIC_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise PublicationError(f"Publication run is missing artifacts: {', '.join(missing)}.")

    manifest = _read_json(run_dir / "run_manifest.json")
    status = _read_json(run_dir / "operational_status.json")
    freshness = _read_json(run_dir / "data_freshness.json")
    lineage = manifest.get("model_lineage") or {}
    projections = pd.read_csv(run_dir / "player_gameweek_projections.csv")
    squad = pd.read_csv(run_dir / "optimized_squad.csv")
    lineup = pd.read_csv(run_dir / "optimized_lineup.csv")

    gates: dict[str, str] = {}
    _gate(
        gates,
        "authoritative_run_class",
        manifest.get("run_class") == "authoritative_publication",
    )
    _gate(gates, "official_source", lineage.get("source_mode") == "official_current_season")
    _gate(gates, "mock_rejected", not manifest.get("warnings") and not status.get("warning"))
    _gate(gates, "season_match", manifest.get("target_season") == preparation.season)
    _gate(gates, "inferred_season_match", manifest.get("inferred_official_season") == preparation.season)
    _gate(gates, "gameweek_match", int(manifest.get("target_gameweek", -1)) == preparation.target.gameweek)
    _gate(gates, "frontend_schema", manifest.get("frontend_schema_version") == config.frontend_schema_version)
    _gate(gates, "projection_keys_unique", not projections.duplicated(["season", "gameweek", "stable_player_id"]).any())
    _gate(gates, "official_prices_present", "price_tenths" in projections and projections["price_tenths"].notna().all())
    _gate(gates, "expected_minutes_valid", _numeric_between(projections, "expected_minutes", 0, 90))
    for column in ("p_appearance", "p_start", "prob_points_ge_5", "prob_points_ge_10"):
        if column in projections:
            _gate(gates, f"{column}_valid", _numeric_between(projections, column, 0, 1))
    _gate(
        gates,
        "appearance_at_least_start",
        pd.to_numeric(projections["p_appearance"], errors="coerce")
        .add(1e-12)
        .ge(pd.to_numeric(projections["p_start"], errors="coerce"))
        .all(),
    )
    _gate(
        gates,
        "nonnegative_xpoints",
        pd.to_numeric(projections["expected_points"], errors="coerce").ge(0).all(),
    )
    _gate(
        gates,
        "conditional_xpoints_required",
        "expected_points_given_appearance" in projections.columns,
    )
    conditional_columns = [
        column for column in projections.columns if "given_appearance" in column or "conditional" in column
    ]
    _gate(
        gates,
        "conditional_xpoints_finite",
        all(_numeric_finite(projections, column) for column in conditional_columns),
    )
    fixture_rows = projections.loc[pd.to_numeric(projections["fixture_count"], errors="coerce").gt(0)]
    _gate(
        gates,
        "opponent_coverage",
        fixture_rows["opponent_display"].notna().all()
        and ~fixture_rows["opponent_display"].astype(str).str.strip().eq("").any(),
    )
    _gate(gates, "identity_coverage", not projections["stable_player_id"].isna().any())
    _gate(gates, "squad_size", len(squad) == 15)
    _gate(
        gates,
        "lineup_size",
        len(squad.loc[_role_values(squad).isin({"starter", "captain", "vice_captain"})]) == 11,
    )
    _gate(gates, "legal_squad", _legal_squad(squad))
    _gate(gates, "legal_lineup", _legal_lineup(squad))
    _gate(gates, "captaincy_valid", _valid_captaincy(squad))
    decision_summary = lineup.iloc[0].to_dict() if not lineup.empty else {}
    optimizer_status = str(
        decision_summary.get("solver_status")
        or lineage.get("decision_solver_status")
        or lineage.get("optimizer_status")
        or ""
    )
    _gate(gates, "optimizer_status", optimizer_status in {"heuristic_feasible", "optimal"})
    refinement = str(
        decision_summary.get("lineup_refinement_status")
        or lineage.get("lineup_refinement_status")
        or ""
    )
    _gate(gates, "lineup_refinement_complete", refinement in {"single_change_local_optimum", "exactly_refined"})
    generated_at = pd.to_datetime(freshness.get("generated_at"), utc=True, errors="coerce")
    current_time = pd.Timestamp(now or datetime.now(UTC))
    _gate(gates, "freshness_present", not pd.isna(generated_at))
    _gate(gates, "not_stale", freshness.get("stale") is False)
    age_hours = (current_time - generated_at).total_seconds() / 3600
    _gate(gates, "freshness_age", 0 <= age_hours <= PUBLICATION_MAX_AGE_HOURS)
    _gate(gates, "source_hashes_present", bool(preparation.source_hashes))
    snapshots = freshness.get("official_snapshots") or {}
    _gate(
        gates,
        "official_snapshot_lineage",
        all(
            snapshots.get(endpoint, {}).get("sha256") == expected_hash
            and bool(snapshots.get(endpoint, {}).get("retrieved_at"))
            for endpoint, expected_hash in preparation.source_hashes.items()
        ),
    )
    _gate(
        gates,
        "current_season_rows_match",
        int(lineage.get("completed_player_fixture_rows_used") or 0)
        == preparation.current_season_player_rows
        and int(lineage.get("completed_team_fixture_rows_used") or 0)
        == preparation.current_season_team_rows,
    )
    _gate(
        gates,
        "current_season_events_reconstructed",
        int(lineage.get("reconstructed_current_season_events") or 0)
        == preparation.reconstructed_events,
    )
    _gate(
        gates,
        "clean_preparation_source",
        not preparation.source_state or preparation.source_state.get("dirty") is False,
    )
    manifest_source_state = manifest.get("source_state") or {}
    _gate(
        gates,
        "preparation_forecast_source_match",
        not preparation.source_state
        or all(
            preparation.source_state.get(key) == manifest_source_state.get(key)
            for key in ("commit", "tracked_diff_sha256", "source_tree_sha256")
        ),
    )
    disclaimer = str(status.get("disclaimer") or freshness.get("disclaimer") or "")
    _gate(gates, "disclaimer_present", bool(disclaimer.strip()))

    inventory_root = public_dir.resolve() if public_dir is not None else run_dir
    if public_dir is not None:
        synced_manifest = _read_json(inventory_root / "run_manifest.json")
        _gate(gates, "synced_run_id", synced_manifest.get("run_id") == manifest.get("run_id"))
        _gate(
            gates,
            "synced_frontend_schema",
            synced_manifest.get("frontend_schema_version") == config.frontend_schema_version,
        )
    public_inventory: dict[str, dict[str, Any]] = {}
    for name in PUBLIC_ARTIFACTS:
        path = inventory_root / name
        if not path.is_file():
            raise PublicationError(f"Sanitized publication is missing {name}.")
        contents = path.read_text(encoding="utf-8")
        _gate(gates, f"{name}_no_local_paths", LOCAL_PATH_PATTERN.search(contents) is None)
        _gate(gates, f"{name}_no_secrets", SECRET_PATTERN.search(contents) is None)
        if name.endswith(".json"):
            parsed = _read_json(path)
            strings = tuple(_string_values(parsed))
            _gate(gates, f"{name}_no_mock_markers", not any(MOCK_PATTERN.search(value) for value in strings))
            _gate(
                gates,
                f"{name}_no_placeholders",
                not any(PLACEHOLDER_PATTERN.search(value) for value in strings),
            )
        else:
            _gate(gates, f"{name}_no_mock_markers", MOCK_PATTERN.search(contents) is None)
            _gate(gates, f"{name}_no_placeholders", PLACEHOLDER_PATTERN.search(contents) is None)
        public_inventory[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "publication_audit.json"
    audit = {
        "schema_version": "phase9b2_publication_audit_v1",
        "run_id": str(manifest.get("run_id")),
        "git_sha": git_sha or _git_sha(),
        "requested_season": preparation.season,
        "target_gameweek_resolution": asdict(preparation.target),
        "historical_source_revision": preparation.source_revision,
        "historical_rows": preparation.historical_rows,
        "source_hashes": preparation.source_hashes,
        "current_season_reconstruction": {
            "events": preparation.reconstructed_events,
            "blank_events": list(preparation.blank_events),
            "player_fixture_rows": preparation.current_season_player_rows,
            "team_fixture_rows": preparation.current_season_team_rows,
            "manifest_path": preparation.reconstruction_manifest_path,
        },
        "model_lineage": {
            key: lineage.get(key)
            for key in (
                "team_model",
                "minutes_models",
                "xpoints_models",
                "decision_optimizer",
                "decision_optimizer_scope",
                "source_mode",
            )
        },
        "optimizer_summary": {
            "status": optimizer_status,
            "lineup_refinement_status": refinement,
            "termination_reason": decision_summary.get("termination_reason"),
            "optimality_scope": decision_summary.get("optimality_scope"),
        },
        "validation_summary": gates,
        "sanitized_publication_inventory": public_inventory,
        "stage_seconds": preparation.stage_seconds,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return PublicationValidation(
        run_id=str(manifest.get("run_id")),
        season=preparation.season,
        gameweek=preparation.target.gameweek,
        gates=gates,
        public_inventory=public_inventory,
        audit_path=audit_path,
    )


def write_preparation(preparation: PublicationPreparation, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(preparation), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_preparation(path: Path) -> PublicationPreparation:
    payload = _read_json(path)
    target = TargetGameweekResolution(**payload.pop("target"))
    payload["historical_seasons"] = tuple(payload["historical_seasons"])
    payload["blank_events"] = tuple(payload.get("blank_events", ()))
    return PublicationPreparation(target=target, **payload)


def _read_optional_parquet(path: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    resolved = Path(path)
    if not resolved.is_file():
        raise PublicationError(f"Prepared current-season artifact is missing: {resolved}.")
    return pd.read_parquet(resolved)


def _validate_historical_publication_frame(frame: pd.DataFrame, *, season: str) -> None:
    required = {
        "season",
        "player_id",
        "player_code",
        "fixture_id",
        "gameweek",
        "kickoff_time",
        "retrieved_at",
    }
    if missing := required.difference(frame.columns):
        raise PublicationError(f"{season} history is missing columns: {', '.join(sorted(missing))}.")
    if any(column.lower() == "xp" for column in frame.columns):
        raise PublicationError(f"{season} normalized history still contains forbidden raw xP.")
    if frame.duplicated(["season", "player_id", "fixture_id"]).any():
        raise PublicationError(f"{season} history contains duplicate player-fixture keys.")
    if frame[list(required)].isna().any().any():
        raise PublicationError(f"{season} history contains missing required IDs or timestamps.")


def _validate_phase2_publication_artifacts(
    *,
    normalized_dir: Path,
    historical_seasons: tuple[str, ...],
) -> None:
    phase2 = normalized_dir / "phase2"
    required_paths = {
        "team identities": phase2 / "dim_team.parquet",
        "team-season identities": phase2 / "team_season_map.parquet",
        "player identities": phase2 / "dim_player.parquet",
        "player-season identities": phase2 / "player_season_map.parquet",
        "fixtures": phase2 / "dim_fixture.parquet",
        "facts": phase2 / "fact_player_fixture.parquet",
        "features": phase2 / "features_player_fixture.parquet",
    }
    for label, path in required_paths.items():
        if not path.is_file():
            raise PublicationError(f"Phase 2 reconstruction is missing {label}: {path}.")

    facts = pd.read_parquet(required_paths["facts"])
    features = pd.read_parquet(required_paths["features"])
    expected = set(historical_seasons)
    if set(facts["season"].astype(str).unique()) != expected:
        raise PublicationError("Phase 2 facts do not cover exactly the requested historical seasons.")
    if set(features["season"].astype(str).unique()) != expected:
        raise PublicationError("Phase 2 features do not cover exactly the requested historical seasons.")
    fact_key = ["season", "player_uid", "fixture_key"]
    if facts[fact_key].isna().any().any() or facts.duplicated(fact_key).any():
        raise PublicationError("Phase 2 facts contain missing or duplicate player-fixture identities.")
    for frame, label in ((facts, "facts"), (features, "features")):
        for column in ("source_available_time", "information_cutoff"):
            if column not in frame or frame[column].isna().any():
                raise PublicationError(f"Phase 2 {label} lack complete {column} values.")
        available = pd.to_datetime(frame["source_available_time"], utc=True, errors="coerce")
        cutoff = pd.to_datetime(frame["information_cutoff"], utc=True, errors="coerce")
        if available.isna().any() or cutoff.isna().any() or available.le(cutoff).any():
            raise PublicationError(f"Phase 2 {label} expose target results at or before the cutoff.")
    for column in (
        "player_max_source_available_time",
        "team_max_source_available_time",
        "opponent_max_source_available_time",
    ):
        if column not in features:
            raise PublicationError(f"Phase 2 features are missing {column}.")
        available = pd.to_datetime(features[column], utc=True, errors="coerce")
        cutoff = pd.to_datetime(features["information_cutoff"], utc=True, errors="coerce")
        observed = available.notna()
        if available.loc[observed].ge(cutoff.loc[observed]).any():
            raise PublicationError(f"Phase 2 features use unavailable history in {column}.")


def _snapshot_source_hashes(raw_dir: Path, *, season: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for endpoint in ("bootstrap_static", "fixtures"):
        metadata = sorted((raw_dir / season / endpoint).glob("*.metadata.json"))
        if not metadata:
            raise PublicationError(f"Missing source metadata for {endpoint}.")
        payload = _read_json(metadata[-1])
        value = str(payload.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise PublicationError(f"Invalid source hash for {endpoint}.")
        hashes[endpoint] = value
    return hashes


def _gate(gates: dict[str, str], name: str, condition: bool) -> None:
    if not bool(condition):
        raise PublicationError(f"Publication gate failed: {name}.")
    gates[name] = "passed"


def _numeric_between(frame: pd.DataFrame, column: str, lower: float, upper: float) -> bool:
    if column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and values.between(lower, upper, inclusive="both").all())


def _numeric_finite(frame: pd.DataFrame, column: str) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and values.map(math.isfinite).all())


def _legal_squad(squad: pd.DataFrame) -> bool:
    if "position" not in squad:
        position_column = "fpl_position"
    else:
        position_column = "position"
    counts = squad[position_column].value_counts().to_dict()
    team_column = "team" if "team" in squad else "player_team_uid"
    cost_column = "price_tenths"
    return (
        counts == {"DEF": 5, "MID": 5, "FWD": 3, "GKP": 2}
        and squad[team_column].value_counts().max() <= 3
        and pd.to_numeric(squad[cost_column], errors="coerce").sum() <= 1000
    )


def _legal_lineup(lineup: pd.DataFrame) -> bool:
    starters = lineup.loc[_role_values(lineup).isin({"starter", "captain", "vice_captain"})]
    position_column = "position" if "position" in starters else "fpl_position"
    counts = starters[position_column].value_counts().to_dict()
    return (
        counts.get("GKP", 0) == 1
        and counts.get("DEF", 0) >= 3
        and counts.get("MID", 0) >= 2
        and counts.get("FWD", 0) >= 1
        and sum(counts.values()) == 11
    )


def _valid_captaincy(lineup: pd.DataFrame) -> bool:
    roles = _role_values(lineup)
    captains = lineup.loc[roles.eq("captain")]
    vice = lineup.loc[roles.eq("vice_captain")]
    if len(captains) != 1 or len(vice) != 1:
        return False
    player_column = "stable_player_id" if "stable_player_id" in lineup else "player_uid"
    return captains.iloc[0][player_column] != vice.iloc[0][player_column]


def _role_values(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["selected_role"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )


def _string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"Unable to read publication JSON {path.name}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or None
