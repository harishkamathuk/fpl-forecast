from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT, RAW_FPL_API_DIR
from fpl_forecast.operations.config import (
    OPERATIONAL_RUNS_DIR,
    STATUS_PATH,
    OperationalPaths,
    load_operational_config,
    resolve_operational_paths,
)
from fpl_forecast.operations.current_panel import reconstruct_completed_current_season
from fpl_forecast.operations.launch import LaunchCheck, check_season_launch
from fpl_forecast.operations.locking import RefreshLock, RefreshLockError
from fpl_forecast.operations.live_results import validate_event_live_for_forecast
from fpl_forecast.operations.model_chain import (
    resolve_official_gameweek_deadline,
    run_operational_model_chain,
)
from fpl_forecast.operations.publication import latest_successful, publish_failure, publish_success
from fpl_forecast.operations.state import OperationalStateName, OperationalStatus, now_utc, write_status


@dataclass(frozen=True)
class RefreshResult:
    status: OperationalStatus
    run_id: str | None
    run_dir: Path | None
    manifest_path: Path | None
    no_op: bool


def refresh_operational(
    *,
    season: str,
    offline: bool = True,
    mock_launch: bool = False,
    force: bool = False,
    run_id: str | None = None,
    fail_stage: str | None = None,
    status_only: bool = False,
    target_gameweek: int = 1,
    completed_player_fixtures: pd.DataFrame | None = None,
    completed_team_fixtures: pd.DataFrame | None = None,
    normalized_dir: Path | str = NORMALIZED_DIR,
    raw_fpl_dir: Path = RAW_FPL_API_DIR,
    operational_root: Path | None = None,
    authoritative_publication: bool = False,
    information_cutoff: str | pd.Timestamp | None = None,
) -> RefreshResult:
    source_state = _source_state()
    if authoritative_publication:
        require_clean_source_state(
            source_state,
            operation="Authoritative publication",
        )
    paths = resolve_operational_paths(operational_root)
    _ensure_dirs(paths)
    config = load_operational_config()
    run_class = (
        "authoritative_publication"
        if authoritative_publication
        else ("test" if mock_launch else "operational")
    )
    latest = None
    try:
        with RefreshLock(paths.lock_path):
            latest = latest_successful(paths.latest_successful_path)
            launch = (
                _mock_launch_check(season)
                if mock_launch
                else check_season_launch(
                    season=season,
                    raw_dir=raw_fpl_dir,
                    offline=offline,
                )
            )
            if status_only or launch.status.state != OperationalStateName.READY_TO_REFRESH:
                status = _with_latest(launch.status, latest)
                write_status(status, paths.status_path)
                return RefreshResult(status, None, None, None, no_op=False)
            run_id = run_id or f"phase8_operational_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            temp_dir = paths.output_dir / f".tmp_{run_id}"
            stages: list[str] = []
            fingerprint = None
            owns_staging = False
            try:
                temp_dir.mkdir(parents=True)
                owns_staging = True
                _maybe_fail("ingestion", fail_stage)
                stages.append("launch_checked")
                cutoff = (
                    None
                    if mock_launch
                    else _resolve_information_cutoff(
                        season=season,
                        target_gameweek=target_gameweek,
                        source_mode="official_current_season",
                        normalized_dir=normalized_dir,
                        information_cutoff=information_cutoff,
                    )
                )
                reconstruction_hashes = {}
                if (
                    not mock_launch
                    and target_gameweek > 1
                    and (completed_player_fixtures is None and completed_team_fixtures is None)
                ):
                    reconstruction = reconstruct_completed_current_season(
                        season=season,
                        target_gameweek=target_gameweek,
                        information_cutoff=cutoff,
                        raw_fpl_dir=raw_fpl_dir,
                        normalized_dir=Path(normalized_dir),
                        run_id=run_id,
                        requested_gameweek=target_gameweek,
                        git_commit=source_state.get("commit"),
                        clean_source=not source_state["dirty"],
                        refresh=not offline,
                    )
                    completed_player_fixtures = _read_reconstructed_history(
                        reconstruction.player_history_path,
                        reconstruction.player_rows,
                        season=season,
                        target_gameweek=target_gameweek,
                        cutoff=cutoff,
                    )
                    completed_team_fixtures = _read_reconstructed_history(
                        reconstruction.team_history_path,
                        reconstruction.team_rows,
                        season=season,
                        target_gameweek=target_gameweek,
                        cutoff=cutoff,
                    )
                    # Archive exact provenance before any later stage can fail.
                    manifest = json.loads(reconstruction.manifest_path.read_text(encoding="utf-8"))
                    if (
                        manifest.get("season") != season
                        or manifest.get("resolved_target_gameweek") != target_gameweek
                        or pd.Timestamp(manifest["information_cutoff"]) != cutoff
                        or manifest.get("player_fixture_rows") != reconstruction.player_rows
                        or manifest.get("team_fixture_rows") != reconstruction.team_rows
                        or not manifest.get("source_hashes")
                    ):
                        raise ValueError("Reconstruction manifest does not match refresh inputs.")
                    reconstruction_hashes = manifest["source_hashes"]
                    shutil.copyfile(
                        reconstruction.manifest_path,
                        temp_dir / "current_season_reconstruction.json",
                    )
                    stages.append("current_season_reconstructed")
                fingerprint = _input_fingerprint(
                    launch,
                    mock_launch=mock_launch,
                    target_gameweek=target_gameweek,
                    completed_player_fixtures=completed_player_fixtures,
                    completed_team_fixtures=completed_team_fixtures,
                    normalized_dir=normalized_dir,
                    source_state=source_state,
                    run_class=run_class,
                )
                if cutoff is not None:
                    fingerprint = hashlib.sha256(
                        json.dumps(
                            {
                                "base": fingerprint,
                                "official_deadline": cutoff.isoformat(),
                                "reconstruction_source_hashes": reconstruction_hashes,
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                if latest and latest.get("input_fingerprint") == fingerprint and not force:
                    status = OperationalStatus(
                        state=OperationalStateName.SUCCEEDED,
                        target_season=season,
                        inferred_official_season=launch.status.inferred_official_season,
                        checked_at=now_utc(),
                        latest_official_deadline=launch.status.latest_official_deadline,
                        latest_successful_run_id=latest.get("run_id"),
                        reason="Inputs, code and configuration are unchanged; latest successful run reused.",
                        dashboard_can_display_forecasts=True,
                        retry_automatically=False,
                        extra={"no_op": True, "input_fingerprint": fingerprint},
                    )
                    (temp_dir / "current_season_reconstruction.json").unlink(missing_ok=True)
                    temp_dir.rmdir()
                    write_status(status, paths.status_path)
                    return RefreshResult(
                        status,
                        latest.get("run_id"),
                        Path(latest["run_dir"]),
                        Path(latest["manifest_path"]),
                        True,
                    )

                _assert_run_id_available(run_id, paths=paths, temp_dir=None)
                _maybe_fail("modeling", fail_stage)
                frontend = _build_frontend_artifacts(
                    temp_dir,
                    season=season,
                    run_id=run_id,
                    config=config,
                    target_gameweek=target_gameweek,
                    completed_player_fixtures=completed_player_fixtures,
                    completed_team_fixtures=completed_team_fixtures,
                    normalized_dir=normalized_dir,
                    source_mode="mock" if mock_launch else "official_current_season",
                    information_cutoff=cutoff,
                )
                stages.append("frontend_artifacts_built")
                _validate_frontend_artifacts(frontend)
                _maybe_fail("optimization", fail_stage)
                stages.append("frontend_artifacts_validated")
                completed_at = now_utc()
                manifest = {
                    "schema_version": config.schema_version,
                    "frontend_schema_version": config.frontend_schema_version,
                    "run_id": run_id,
                    "target_season": season,
                    "target_gameweek": target_gameweek,
                    "inferred_official_season": launch.status.inferred_official_season,
                    "launch_state": launch.status.state.value,
                    "created_at": completed_at,
                    "completed_at": completed_at,
                    "input_fingerprint": fingerprint,
                    "code_revision": _git(["rev-parse", "--short", "HEAD"]),
                    "dirty_worktree": source_state["dirty"],
                    "source_state": source_state,
                    "run_class": run_class,
                    "models": config.default_models,
                    "model_lineage": _read_json(temp_dir / "model_lineage.json"),
                    "warnings": ["mock_target_season_transition"] if mock_launch else [],
                    "fall_back_flags": _lineage_fallback_flags(
                        _read_json(temp_dir / "model_lineage.json")
                    ),
                    "stages": stages,
                    "previous_latest_successful_run_id": latest.get("run_id") if latest else None,
                    "completion_stage": "published",
                }
                _maybe_fail("publication", fail_stage)
                final_dir = publish_success(
                    temp_dir,
                    run_id=run_id,
                    manifest={**manifest, "input_fingerprint": fingerprint},
                    runs_dir=paths.runs_dir,
                    pointer_path=paths.latest_successful_path,
                )
                status = OperationalStatus(
                    state=OperationalStateName.SUCCEEDED,
                    target_season=season,
                    inferred_official_season=launch.status.inferred_official_season,
                    checked_at=completed_at,
                    latest_official_deadline=launch.status.latest_official_deadline,
                    latest_successful_run_id=run_id,
                    reason="Operational refresh published atomically.",
                    dashboard_can_display_forecasts=True,
                    retry_automatically=False,
                    extra={"run_dir": str(final_dir), "input_fingerprint": fingerprint},
                )
                try:
                    write_status(status, paths.status_path)
                except OSError as status_exc:
                    # The pointer already committed. Never report last-success
                    # failure or attempt to republish the renamed staging tree.
                    status = OperationalStatus(**{
                        **status.__dict__,
                        "warning": f"Publication committed but status could not be written: {status_exc}",
                    })
                return RefreshResult(
                    status, run_id, final_dir, final_dir / "run_manifest.json", no_op=False
                )
            except Exception as exc:
                failed_at = now_utc()
                manifest = {
                    "schema_version": config.schema_version,
                    "frontend_schema_version": config.frontend_schema_version,
                    "run_id": run_id,
                    "target_season": season,
                    "inferred_official_season": launch.status.inferred_official_season,
                    "created_at": failed_at,
                    "completed_at": failed_at,
                    "input_fingerprint": fingerprint,
                    "source_state": source_state,
                    "run_class": run_class,
                    "completion_stage": fail_stage or "unknown",
                    "error": str(exc),
                    "previous_latest_successful_run_id": latest.get("run_id") if latest else None,
                    "stages": stages,
                }
                failed_dir = None
                failure_publication_error = None
                if owns_staging:
                    try:
                        failed_dir = publish_failure(
                            temp_dir,
                            run_id=run_id,
                            manifest=manifest,
                            failed_runs_dir=paths.failed_dir,
                        )
                    except Exception as publication_exc:
                        # Never overwrite an older failed run or touch unowned staging.
                        # Retain our staging evidence and still report the original failure.
                        failure_publication_error = str(publication_exc)
                status_extra = {"failed_dir": str(failed_dir) if failed_dir else None}
                if owns_staging and temp_dir.exists():
                    status_extra["staging_dir"] = str(temp_dir)
                if failure_publication_error:
                    status_extra["failure_publication_error"] = failure_publication_error
                if owns_staging and not temp_dir.exists() and (paths.runs_dir / run_id).exists():
                    status_extra["unreferenced_run_dir"] = str(paths.runs_dir / run_id)
                status = OperationalStatus(
                    state=OperationalStateName.FAILED_USING_LAST_SUCCESS,
                    target_season=season,
                    inferred_official_season=launch.status.inferred_official_season,
                    checked_at=failed_at,
                    latest_official_deadline=launch.status.latest_official_deadline,
                    latest_successful_run_id=latest.get("run_id") if latest else None,
                    reason=f"Refresh failed at {fail_stage or 'unknown'}; latest successful pointer was not changed.",
                    warning=str(exc),
                    dashboard_can_display_forecasts=bool(latest),
                    retry_automatically=True,
                    extra=status_extra,
                )
                write_status(status, paths.status_path)
                return RefreshResult(status, None, None, None, no_op=False)
    except Exception as exc:
        # Launch/pointer parsing can fail before staging exists. Report failure
        # without touching any caller-owned staging or successful-run evidence.
        if isinstance(exc, RefreshLockError):
            try:
                latest = latest_successful(paths.latest_successful_path)
            except (OSError, ValueError):
                latest = None
        status = OperationalStatus(
            state=OperationalStateName.FAILED_USING_LAST_SUCCESS,
            target_season=season,
            inferred_official_season=None,
            checked_at=now_utc(),
            latest_successful_run_id=latest.get("run_id") if latest else None,
            reason=str(exc),
            dashboard_can_display_forecasts=bool(latest),
            retry_automatically=True,
        )
        write_status(status, paths.status_path)
        return RefreshResult(status, None, None, None, no_op=False)


def operational_status_lines() -> list[str]:
    status_path = STATUS_PATH
    pointer = latest_successful()
    if not status_path.exists():
        return ["state=UNKNOWN", "reason=No operational status has been written yet."]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    lines = [
        f"state={status['state']}",
        f"target_season={status['target_season']}",
        f"inferred_official_season={status.get('inferred_official_season')}",
        f"latest_successful_run_id={status.get('latest_successful_run_id')}",
        f"dashboard_can_display_forecasts={status.get('dashboard_can_display_forecasts')}",
        f"reason={status.get('reason')}",
        f"status_path={status_path}",
    ]
    if pointer:
        lines.append(f"latest_successful={pointer['run_id']}")
    return lines


def verify_operational_readiness() -> list[str]:
    pointer = latest_successful()
    status_exists = STATUS_PATH.exists()
    runs_dir_exists = OPERATIONAL_RUNS_DIR.exists()
    return [
        "operational_readiness=ok",
        f"status_exists={status_exists}",
        f"latest_successful_run_id={pointer.get('run_id') if pointer else None}",
        f"runs_dir_exists={runs_dir_exists}",
        "dashboard_contract=phase9_frontend_v1",
    ]


def _build_frontend_artifacts(
    temp_dir: Path,
    *,
    season: str,
    run_id: str,
    config,
    target_gameweek: int = 1,
    completed_player_fixtures: pd.DataFrame | None = None,
    completed_team_fixtures: pd.DataFrame | None = None,
    normalized_dir: Path | str = NORMALIZED_DIR,
    source_mode: str = "mock",
    information_cutoff: str | pd.Timestamp | None = None,
) -> dict[str, Path]:
    selected_model = config.default_models["xpoints"]
    cutoff = _resolve_information_cutoff(
        season=season,
        target_gameweek=target_gameweek,
        source_mode=source_mode,
        normalized_dir=normalized_dir,
        information_cutoff=information_cutoff,
    )
    if completed_player_fixtures is not None and "official_event_total_points" in completed_player_fixtures.columns:
        issues = validate_event_live_for_forecast(completed_player_fixtures, information_cutoff=cutoff)
        if issues:
            raise ValueError(
                f"Completed-result publication safety gate failed: {', '.join(issues)}"
            )
    result = run_operational_model_chain(
        season=season,
        run_id=run_id,
        output_dir=temp_dir,
        target_gameweek=target_gameweek,
        completed_player_fixtures=completed_player_fixtures,
        completed_team_fixtures=completed_team_fixtures,
        normalized_dir=normalized_dir,
        source_mode=source_mode,
        as_of=cutoff,
    )

    projections = result.decision_candidates.loc[
        result.decision_candidates["model_name"].eq(selected_model)
    ].copy()
    projections = projections.rename(
        columns={
            "player_uid": "stable_player_id",
            "player_name": "player",
            "fpl_position": "position",
            "player_team_uid": "team",
        }
    )
    projections["schema_version"] = config.frontend_schema_version
    projections["season"] = season
    projections["status"] = projections["status"].fillna("a")
    projections["news"] = projections["news"].fillna("")
    projections["model_variant"] = selected_model
    projections["data_timestamp"] = now_utc()
    projections["team_model_run_id"] = result.lineage["team_model_run_id"]
    projections["minutes_model_run_id"] = result.lineage["minutes_model_run_id"]
    projections["xpoints_model_run_id"] = result.lineage["xpoints_model_run_id"]
    projections["decision_run_id"] = result.lineage["decision_run_id"]
    projection_columns = [
        "schema_version",
        "season",
        "gameweek",
        "stable_player_id",
        "player",
        "team",
        "position",
        "price_tenths",
        "fixture_count",
        "opponent_display",
        "opponent_short_names",
        "opponent_official_names",
        "opponent_team_uids",
        "home_away_sequence",
        "kickoff_times",
        "expected_points",
        "expected_points_given_appearance",
        "points_std",
        "points_p10",
        "points_p50",
        "points_p90",
        "expected_minutes",
        "p_appearance",
        "p_start",
        "prob_points_ge_5",
        "prob_points_ge_10",
        "status",
        "news",
        "cold_start_no_history",
        "fallback_flag",
        "lineage_note",
        "model_variant",
        "data_timestamp",
        "team_model_run_id",
        "minutes_model_run_id",
        "xpoints_model_run_id",
        "decision_run_id",
    ]
    projections_path = temp_dir / "player_gameweek_projections.csv"
    projections[[column for column in projection_columns if column in projections.columns]].to_csv(
        projections_path,
        index=False,
    )

    squad_path = temp_dir / "optimized_squad.csv"
    result.optimized_squad.to_csv(squad_path, index=False)
    lineup_path = temp_dir / "optimized_lineup.csv"
    lineup = result.optimized_lineup.copy()
    lineup["objective"] = lineup["expected_team_points"]
    lineup.to_csv(lineup_path, index=False)
    comparison_path = temp_dir / "model_comparison.csv"
    result.model_comparison.to_csv(comparison_path, index=False)
    lineage_path = temp_dir / "model_lineage.json"
    lineage_path.write_text(
        json.dumps(result.lineage, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    freshness_path = temp_dir / "data_freshness.json"
    freshness_path.write_text(
        json.dumps(
            {
                "schema_version": config.frontend_schema_version,
                "generated_at": now_utc(),
                "source": _source_description(result.lineage),
                "source_mode": result.lineage.get("source_mode"),
                "lineage_artifact": "model_lineage.json",
                "team_model_run_id": result.lineage["team_model_run_id"],
                "minutes_model_run_id": result.lineage["minutes_model_run_id"],
                "xpoints_model_run_id": result.lineage["xpoints_model_run_id"],
                "decision_run_id": result.lineage["decision_run_id"],
                "official_deadline": result.lineage.get("official_deadline"),
                "official_snapshots": result.lineage.get("official_snapshots"),
                "current_player_count": result.lineage.get("current_player_count"),
                "current_team_count": result.lineage.get("current_team_count"),
                "target_fixture_count": result.lineage.get("target_fixture_count"),
                "identity_coverage": result.lineage.get("identity_coverage"),
                "cold_start_count": result.lineage.get("cold_start_count"),
                "neutral_team_fallback_count": result.lineage.get("neutral_team_fallback_count"),
                "price_source": result.lineage.get("price_source"),
                "rules_source": result.lineage.get("rules_source"),
                "stale": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    status_path = temp_dir / "operational_status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": config.frontend_schema_version,
                "state": OperationalStateName.SUCCEEDED.value,
                "target_season": season,
                "target_gameweek": target_gameweek,
                "run_id": run_id,
                "reason": "Operational refresh published genuinely generated target-season projections.",
                "warning": _status_warning(result.lineage),
                "disclaimer": (
                    "Experimental forecast. This project is not affiliated with or endorsed by "
                    "the Premier League or Fantasy Premier League."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "operational_status": status_path,
        "player_gameweek_projections": projections_path,
        "optimized_squad": squad_path,
        "optimized_lineup": lineup_path,
        "model_comparison": comparison_path,
        "data_freshness": freshness_path,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_frontend_artifacts(paths: dict[str, Path]) -> None:
    required = set(paths)
    if required != {
        "operational_status",
        "player_gameweek_projections",
        "optimized_squad",
        "optimized_lineup",
        "model_comparison",
        "data_freshness",
    }:
        raise ValueError("Frontend artifact contract is incomplete.")
    projections = pd.read_csv(paths["player_gameweek_projections"])
    required_columns = {
        "schema_version",
        "stable_player_id",
        "player",
        "team",
        "position",
        "price_tenths",
        "fixture_count",
        "opponent_display",
        "expected_points",
        "expected_points_given_appearance",
    }
    missing = required_columns.difference(projections.columns)
    if missing:
        raise ValueError(f"Projection artifact missing columns: {', '.join(sorted(missing))}")


def _source_description(lineage: dict[str, Any]) -> str:
    if lineage.get("source_mode") == "official_current_season":
        return "official current-season FPL API snapshots"
    return "mocked target-season production model chain"


def _status_warning(lineage: dict[str, Any]) -> str | None:
    if lineage.get("source_mode") == "official_current_season":
        return None
    return "Representative mocked target-season run; real 2026-27 remains unproven."


def _lineage_fallback_flags(lineage: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if int(lineage.get("cold_start_count") or 0) > 0:
        flags.append("cold_start_new_player")
    if int(lineage.get("neutral_team_fallback_count") or 0) > 0:
        flags.append("neutral_promoted_team")
    return flags


def _mock_launch_check(season: str) -> LaunchCheck:
    return LaunchCheck(
        status=OperationalStatus(
            state=OperationalStateName.READY_TO_REFRESH,
            target_season=season,
            inferred_official_season=season,
            checked_at=now_utc(),
            latest_official_deadline=f"{season[:4]}-08-15T17:30:00Z",
            reason="Mocked target-season launch payload is compatible with verified rules.",
            retry_automatically=False,
        ),
        bootstrap_path=None,
        fixtures_path=None,
        rule_diff={"material_changes": {}},
    )


def _input_fingerprint(
    launch: LaunchCheck,
    *,
    mock_launch: bool,
    target_gameweek: int = 1,
    completed_player_fixtures: pd.DataFrame | None = None,
    completed_team_fixtures: pd.DataFrame | None = None,
    normalized_dir: Path | str = NORMALIZED_DIR,
    source_state: dict[str, Any] | None = None,
    run_class: str = "operational",
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(load_operational_config().__dict__, sort_keys=True, default=str).encode())
    digest.update(json.dumps(source_state or _source_state(), sort_keys=True).encode())
    digest.update(run_class.encode())
    digest.update(str(mock_launch).encode())
    digest.update(str(target_gameweek).encode())
    digest.update(str(Path(normalized_dir)).encode())
    for frame in (completed_player_fixtures, completed_team_fixtures):
        if frame is not None and not frame.empty:
            digest.update(pd.util.hash_pandas_object(frame.sort_index(axis=1), index=True).values.tobytes())
    for path in (launch.bootstrap_path, launch.fixtures_path):
        if path and path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _ensure_dirs(paths: OperationalPaths) -> None:
    for path in (paths.output_dir, paths.runs_dir, paths.failed_dir):
        path.mkdir(parents=True, exist_ok=True)


def _assert_run_id_available(
    run_id: str,
    *,
    paths: OperationalPaths,
    temp_dir: Path | None,
) -> None:
    existing = [
        path
        for path in (paths.runs_dir / run_id, paths.failed_dir / run_id, temp_dir)
        if path is not None and path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to reuse operational run id; existing path(s): "
            + ", ".join(str(path) for path in existing)
        )


def _with_latest(status: OperationalStatus, latest: dict[str, Any] | None) -> OperationalStatus:
    return OperationalStatus(
        **{
            **status.__dict__,
            "latest_successful_run_id": latest.get("run_id") if latest else None,
            "dashboard_can_display_forecasts": bool(latest) and status.dashboard_can_display_forecasts,
        }
    )


def _maybe_fail(stage: str, fail_stage: str | None) -> None:
    if fail_stage == stage:
        raise RuntimeError(f"Injected failure at {stage}.")


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _git_bytes(args: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    return result.stdout


def _source_state() -> dict[str, Any]:
    status = _git(["status", "--short", "--untracked-files=all"])
    tracked_diff = _git_bytes(["diff", "--binary", "HEAD"])
    listed = _git_bytes(
        ["ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    source_tree = hashlib.sha256()
    files = sorted(path for path in listed.split(b"\0") if path)
    for encoded_path in files:
        relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
        source_tree.update(encoded_path)
        source_tree.update(b"\0")
        path = PROJECT_ROOT / relative_path
        if path.is_file():
            source_tree.update(path.read_bytes())
        else:
            source_tree.update(b"<missing>")
        source_tree.update(b"\0")
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_tree_sha256": source_tree.hexdigest(),
        "source_file_count": len(files),
        "source_tree_definition": (
            "sorted Git tracked and untracked non-ignored repository paths and worktree contents; "
            "missing tracked files use an explicit marker"
        ),
    }


def require_clean_source_state(
    source_state: dict[str, Any],
    *,
    operation: str,
) -> None:
    if source_state.get("dirty"):
        raise RuntimeError(
            f"{operation} requires a clean Git worktree; commit or remove tracked and "
            "untracked source changes before retrying."
        )


def _default_information_cutoff(season: str, *, target_gameweek: int) -> pd.Timestamp:
    if target_gameweek <= 1:
        return pd.Timestamp(f"{season[:4]}-08-01T10:00:00Z")
    day = 15 + (target_gameweek - 1) * 7 - 1
    return pd.Timestamp(f"{season[:4]}-08-{day:02d}T10:00:00Z")


def _resolve_information_cutoff(
    *,
    season: str,
    target_gameweek: int,
    source_mode: str,
    normalized_dir: Path | str,
    information_cutoff: str | pd.Timestamp | None,
) -> pd.Timestamp:
    if source_mode == "mock":
        return _default_information_cutoff(season, target_gameweek=target_gameweek)
    if source_mode != "official_current_season":
        raise ValueError(f"Unknown operational source mode: {source_mode}")
    events = pd.read_parquet(Path(normalized_dir) / season / "current_events.parquet")
    event = events.loc[pd.to_numeric(events["gameweek"], errors="coerce").eq(target_gameweek)]
    if "season" in event and not event["season"].eq(season).all():
        raise ValueError("Official event season does not match refresh season.")
    official_deadline = resolve_official_gameweek_deadline(
        season,
        normalized_dir=normalized_dir,
        target_gameweek=target_gameweek,
        events=events,
    )
    if information_cutoff is None:
        return official_deadline
    supplied = pd.to_datetime(information_cutoff, utc=True, errors="coerce")
    if not isinstance(supplied, pd.Timestamp) or pd.isna(supplied):
        raise ValueError("Authoritative official deadline is missing or malformed.")
    if supplied != official_deadline:
        raise ValueError(
            f"Official current gameweek {target_gameweek} deadline does not match the "
            "publication preparation deadline."
        )
    return supplied


def _read_reconstructed_history(
    path: Path | None,
    expected_rows: int,
    *,
    season: str,
    target_gameweek: int,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing reconstructed history: {path}")
    frame = pd.read_parquet(path)
    if len(frame) != expected_rows:
        raise ValueError("Reconstructed history row count changed before forecasting.")
    if frame.empty:
        return frame  # Legitimate blank events have no synthetic rows.
    if not frame["season"].eq(season).all():
        raise ValueError("Reconstructed history contains a different season.")
    gameweeks = pd.to_numeric(frame["gameweek"], errors="coerce")
    available = pd.to_datetime(frame["source_available_time"], utc=True, errors="coerce")
    if (
        gameweeks.isna().any()
        or not gameweeks.between(1, target_gameweek - 1).all()
        or available.isna().any()
        or available.ge(cutoff).any()
        or frame["source_available_method"].isna().any()
        or not frame["source_available_method"]
        .isin({
            "latest_of_bootstrap_fixtures_and_event_live_retrieval",
            "latest_of_bootstrap_fixtures_event_live_and_historical_club_evidence",
        })
        .all()
    ):
        raise ValueError("Reconstructed history has invalid temporal provenance.")
    return frame
