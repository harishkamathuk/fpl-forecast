"""Operational wiring regressions; never execute the forecast/model chain."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from fpl_forecast.operations import orchestrator as ops


@pytest.fixture
def wiring(monkeypatch, tmp_path):
    normalized = tmp_path / "normalized"
    season_dir = normalized / "2026-27"
    season_dir.mkdir(parents=True)
    pd.DataFrame(
        {"season": ["2026-27"], "gameweek": [10], "deadline_time": ["2026-11-07T11:00:00Z"]}
    ).to_parquet(season_dir / "current_events.parquet")
    calls = []
    frame = pd.DataFrame(
        {
            "season": ["2026-27"],
            "gameweek": [9],
            "source_available_time": ["2026-11-02T12:00:00Z"],
            "source_available_method": ["latest_of_bootstrap_fixtures_and_event_live_retrieval"],
            "value": [1],
        }
    )
    manifest = {
        "season": "2026-27",
        "resolved_target_gameweek": 10,
        "information_cutoff": "2026-11-07T11:00:00Z",
        "player_fixture_rows": 1,
        "team_fixture_rows": 1,
        "source_hashes": {"event_live_9": "hash"},
    }

    def reconstruct(**kwargs):
        assert paths.lock_path.exists()
        calls.append(kwargs)
        player = season_dir / "players.parquet"
        team = season_dir / "teams.parquet"
        frame.to_parquet(player)
        frame.to_parquet(team)
        provenance = season_dir / "current_season_reconstruction.json"
        provenance.write_text(json.dumps(manifest))
        return SimpleNamespace(
            player_history_path=player,
            team_history_path=team,
            player_rows=1,
            team_rows=1,
            manifest_path=provenance,
        )

    forwarded = []

    def frontend(temp_dir, **kwargs):
        assert paths.lock_path.exists()
        forwarded.append(kwargs)
        return {}

    paths = ops.resolve_operational_paths(tmp_path / "operational")
    real_frontend = ops._build_frontend_artifacts
    monkeypatch.setattr(ops, "_source_state", lambda: {"dirty": False, "commit": "fixed"})
    monkeypatch.setattr(ops, "check_season_launch", lambda **kw: ops._mock_launch_check("2026-27"))
    monkeypatch.setattr(ops, "reconstruct_completed_current_season", reconstruct)
    monkeypatch.setattr(ops, "_build_frontend_artifacts", frontend)
    monkeypatch.setattr(ops, "_validate_frontend_artifacts", lambda _: None)
    monkeypatch.setattr(
        ops, "run_operational_model_chain", lambda **kw: pytest.fail("model executed")
    )

    def run(run_id="first", **kwargs):
        return ops.refresh_operational(
            season="2026-27",
            target_gameweek=10,
            run_id=run_id,
            normalized_dir=normalized,
            operational_root=paths.output_dir,
            **kwargs,
        )

    return SimpleNamespace(
        run=run,
        calls=calls,
        forwarded=forwarded,
        frame=frame,
        manifest=manifest,
        paths=paths,
        normalized=normalized,
        real_frontend=real_frontend,
    )


def test_reconstruction_noop_revision_and_provenance(wiring):
    first = wiring.run()
    assert first.status.state.value == "SUCCEEDED"
    assert wiring.calls[0]["refresh"] is False
    assert wiring.calls[0]["information_cutoff"] == pd.Timestamp("2026-11-07T11:00:00Z")
    pd.testing.assert_frame_equal(wiring.forwarded[0]["completed_player_fixtures"], wiring.frame)
    assert (first.run_dir / "current_season_reconstruction.json").exists()
    before = wiring.paths.latest_successful_path.read_bytes()
    second = wiring.run("first")
    assert second.no_op
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert not (wiring.paths.output_dir / ".tmp_first").exists()
    wiring.frame["value"] = 2
    assert not wiring.run("third").no_op
    assert len(wiring.forwarded) == 2
    assert not wiring.run("forced", force=True).no_op


@pytest.mark.parametrize("supplied", ["player", "team", "both"])
def test_explicit_partial_and_paired_inputs_preserved(wiring, supplied):
    kwargs = {}
    if supplied in {"player", "both"}:
        kwargs["completed_player_fixtures"] = wiring.frame
    if supplied in {"team", "both"}:
        kwargs["completed_team_fixtures"] = wiring.frame
    assert wiring.run(**kwargs).status.state.value == "SUCCEEDED"
    assert not wiring.calls
    for key, value in kwargs.items():
        assert wiring.forwarded[0][key] is value


@pytest.mark.parametrize(
    "defect", ["missing", "count", "cutoff", "method", "manifest", "fetch", "fingerprint"]
)
def test_failed_evidence_preserves_previous_pointer(wiring, monkeypatch, defect):
    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()
    if defect == "missing":
        monkeypatch.setattr(
            ops,
            "_read_reconstructed_history",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("missing")),
        )
    elif defect == "count":
        wiring.frame.loc[1] = wiring.frame.iloc[0]
    elif defect == "cutoff":
        wiring.frame["source_available_time"] = "2026-11-07T11:00:00Z"
    elif defect == "method":
        wiring.frame["source_available_method"] = None
    elif defect == "manifest":
        wiring.manifest["resolved_target_gameweek"] = 9
    elif defect == "fetch":
        monkeypatch.setattr(
            ops,
            "reconstruct_completed_current_season",
            lambda **kw: (_ for _ in ()).throw(FileNotFoundError("event-live unavailable offline")),
        )
    else:
        monkeypatch.setattr(
            ops,
            "_input_fingerprint",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("fingerprint")),
        )
    failed = wiring.run("failure")
    assert failed.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert len(wiring.forwarded) == 1
    assert (wiring.paths.failed_dir / "failure" / "run_manifest.json").exists()


@pytest.mark.parametrize("defect", ["missing", "duplicate", "invalid", "season"])
def test_invalid_event_deadline_fails_closed(wiring, defect):
    path = wiring.normalized / "2026-27" / "current_events.parquet"
    events = pd.read_parquet(path)
    if defect == "missing":
        path.unlink()
    else:
        if defect == "duplicate":
            events = pd.concat([events, events])
        elif defect == "invalid":
            events["deadline_time"] = "bad"
        else:
            events["season"] = "2025-26"
        events.to_parquet(path)
    assert wiring.run().status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert not wiring.calls
    assert not wiring.forwarded


def test_online_mode_is_explicit(wiring):
    wiring.run(offline=False)
    assert wiring.calls[0]["refresh"] is True


@pytest.mark.parametrize("mode", ["status", "mock", "gw1", "not_ready"])
def test_bypass_reconstruction(wiring, monkeypatch, mode):
    if mode == "gw1":
        # Exercise the existing public default target.
        path = wiring.normalized / "2026-27" / "current_events.parquet"
        events = pd.read_parquet(path)
        events["gameweek"] = 1
        events.to_parquet(path)
        result = ops.refresh_operational(
            season="2026-27",
            run_id="gw1",
            normalized_dir=wiring.normalized,
            operational_root=wiring.paths.output_dir,
        )
    else:
        kwargs = {"status_only": True} if mode == "status" else {"mock_launch": True}
        if mode == "not_ready":
            launch = ops._mock_launch_check("2026-27")
            launch = replace(
                launch,
                status=replace(
                    launch.status, state=ops.OperationalStateName.WAITING_FOR_SEASON_LAUNCH
                ),
            )
            monkeypatch.setattr(ops, "check_season_launch", lambda **kw: launch)
            kwargs = {}
        result = wiring.run(**kwargs)
    assert result.status
    assert not wiring.calls


@pytest.mark.parametrize("gw", [9, 10])
@pytest.mark.parametrize(
    "available", ["2026-11-01T00:00:00Z", "2026-11-07T11:00:00Z", "2026-11-08T00:00:00Z"]
)
def test_frontend_uses_official_deadline(wiring, monkeypatch, gw, available):
    frontend = wiring.real_frontend
    path = wiring.normalized / "2026-27" / "current_events.parquet"
    events = pd.read_parquet(path)
    events["gameweek"] = gw
    events.to_parquet(path)
    observed = []

    def validate(frame, *, information_cutoff):
        observed.append(information_cutoff)
        return ["late"] if pd.Timestamp(available) >= information_cutoff else []

    monkeypatch.setattr(ops, "validate_event_live_for_forecast", validate)

    class StopModel(Exception):
        pass

    def stop(**kwargs):
        raise StopModel

    monkeypatch.setattr(ops, "run_operational_model_chain", stop)
    frame = pd.DataFrame({"official_event_total_points": [1]})
    expected = StopModel if available == "2026-11-01T00:00:00Z" else ValueError
    with pytest.raises(expected):
        frontend(
            wiring.paths.output_dir,
            season="2026-27",
            run_id="gate",
            config=ops.load_operational_config(),
            target_gameweek=gw,
            completed_player_fixtures=frame,
            normalized_dir=wiring.normalized,
            source_mode="official_current_season",
        )
    assert observed == [pd.Timestamp("2026-11-07T11:00:00Z")]


@pytest.mark.parametrize("missing_live", [False, True])
def test_cached_official_reconstruction_end_to_end(wiring, monkeypatch, tmp_path, missing_live):
    from test_phase9b3_current_season import _official_case, _patch_team_identities
    from fpl_forecast.operations.current_panel import reconstruct_completed_current_season

    case = _official_case(
        tmp_path / "official", target_gameweek=2, omit_event_live={1} if missing_live else set()
    )
    _patch_team_identities(monkeypatch)
    monkeypatch.setattr(
        ops, "reconstruct_completed_current_season", reconstruct_completed_current_season
    )
    result = ops.refresh_operational(
        season=case["season"],
        target_gameweek=2,
        offline=True,
        run_id="cached",
        raw_fpl_dir=case["raw_fpl_dir"],
        normalized_dir=case["normalized_dir"],
        operational_root=wiring.paths.output_dir,
    )
    if missing_live:
        assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
        assert not wiring.forwarded
        assert not wiring.paths.latest_successful_path.exists()
    else:
        assert result.status.state.value == "SUCCEEDED"
        assert len(wiring.forwarded[0]["completed_player_fixtures"]) == 2
        assert len(wiring.forwarded[0]["completed_team_fixtures"]) == 1


def test_history_file_must_exist_and_match_count(tmp_path):
    path = tmp_path / "absent.parquet"
    kwargs = dict(season="2026-27", target_gameweek=10, cutoff=pd.Timestamp("2026-11-07T11:00:00Z"))
    with pytest.raises(FileNotFoundError):
        ops._read_reconstructed_history(path, 1, **kwargs)
    pd.DataFrame().to_parquet(path)
    with pytest.raises(ValueError, match="row count"):
        ops._read_reconstructed_history(path, 1, **kwargs)


def test_refresh_lock_prevents_reconstruction(wiring):
    from fpl_forecast.operations.locking import RefreshLock

    with RefreshLock(wiring.paths.lock_path):
        result = wiring.run()
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert not wiring.calls
    assert not wiring.forwarded


def test_revised_source_hash_and_deadline_invalidate_noop(wiring):
    wiring.run()
    wiring.manifest["source_hashes"]["event_live_9"] = "revised"
    assert not wiring.run("source_revision").no_op
    path = wiring.normalized / "2026-27" / "current_events.parquet"
    events = pd.read_parquet(path)
    events["deadline_time"] = "2026-11-07T12:00:00Z"
    events.to_parquet(path)
    wiring.manifest["information_cutoff"] = "2026-11-07T12:00:00Z"
    assert not wiring.run("deadline_revision").no_op


def test_modeling_failure_preserves_pointer_and_reconstruction_evidence(wiring):
    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()
    result = wiring.run("model_failure", fail_stage="modeling", force=True)
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert (
        wiring.paths.failed_dir / "model_failure" / "current_season_reconstruction.json"
    ).exists()
    assert len(wiring.forwarded) == 1


def test_existing_staging_is_preserved_and_failure_status_written(wiring):
    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()
    staging = wiring.paths.output_dir / ".tmp_collision"
    staging.mkdir()
    marker = staging / "current_season_reconstruction.json"
    marker.write_bytes(b"unowned evidence")
    result = wiring.run("collision")
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert marker.read_bytes() == b"unowned evidence"
    assert list(staging.iterdir()) == [marker]
    assert len(wiring.calls) == 1


@pytest.mark.parametrize("kind", ["success", "failed"])
def test_run_id_collisions_preserve_evidence_and_pointer(wiring, kind):
    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()
    collision = (
        wiring.paths.runs_dir / "first" if kind == "success" else wiring.paths.failed_dir / "reused"
    )
    if kind == "failed":
        collision.mkdir()
        (collision / "run_manifest.json").write_text("old failure evidence")
    evidence = {p.name: p.read_bytes() for p in collision.iterdir() if p.is_file()}
    result = wiring.run("first" if kind == "success" else "reused", force=True)
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert {p.name: p.read_bytes() for p in collision.iterdir() if p.is_file()} == evidence
    if kind == "failed":
        assert result.status.extra["failure_publication_error"]
        assert (wiring.paths.output_dir / ".tmp_reused").exists()
    assert len(wiring.forwarded) == 1


@pytest.mark.parametrize("failure", ["manifest", "copy", "unlink", "rmdir", "publish_failure"])
def test_preparation_cleanup_and_failure_publisher_exceptions(wiring, monkeypatch, failure):
    from pathlib import Path

    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()
    if failure == "manifest":
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "current_season_reconstruction.json":
                return "{broken"
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    elif failure == "copy":
        monkeypatch.setattr(
            ops.shutil, "copyfile", lambda *args: (_ for _ in ()).throw(OSError("copy failed"))
        )
    elif failure in {"unlink", "rmdir"}:
        original = getattr(Path, failure)

        def cleanup(path, *args, **kwargs):
            if path == wiring.paths.output_dir / ".tmp_attempt" or (
                path.parent == wiring.paths.output_dir / ".tmp_attempt"
            ):
                raise OSError("cleanup failed")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, failure, cleanup)
    else:
        monkeypatch.setattr(
            ops,
            "publish_failure",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failure storage unavailable")),
        )
    kwargs = {"fail_stage": "modeling", "force": True} if failure == "publish_failure" else {}
    result = wiring.run("attempt", **kwargs)
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert len(wiring.forwarded) == 1
    if failure == "publish_failure":
        assert result.status.extra["failure_publication_error"]


@pytest.mark.parametrize(
    "defect", ["missing_event", "duplicate_event", "empty_players", "empty_teams"]
)
def test_prior_fixture_coverage_cannot_silently_be_empty(wiring, monkeypatch, tmp_path, defect):
    from test_phase9b3_current_season import _official_case, _patch_team_identities
    from fpl_forecast.operations import current_panel

    case = _official_case(tmp_path / "coverage", target_gameweek=2)
    _patch_team_identities(monkeypatch)
    monkeypatch.setattr(
        ops,
        "reconstruct_completed_current_season",
        current_panel.reconstruct_completed_current_season,
    )
    event_path = case["normalized_dir"] / "2026-27" / "current_events.parquet"
    events = pd.read_parquet(event_path)
    if defect == "missing_event":
        events.loc[events["gameweek"].eq(2)].to_parquet(event_path)
    elif defect == "duplicate_event":
        pd.concat([events, events.loc[events["gameweek"].eq(1)]]).to_parquet(event_path)
    else:
        builder = (
            "build_current_player_fixture_history"
            if defect == "empty_players"
            else "build_current_team_fixture_history"
        )
        monkeypatch.setattr(current_panel, builder, lambda **kwargs: pd.DataFrame())
    result = ops.refresh_operational(
        season="2026-27",
        target_gameweek=2,
        offline=True,
        run_id="coverage",
        raw_fpl_dir=case["raw_fpl_dir"],
        normalized_dir=case["normalized_dir"],
        operational_root=wiring.paths.output_dir,
    )
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert not wiring.forwarded
    assert not wiring.paths.latest_successful_path.exists()
    failed = json.loads((wiring.paths.failed_dir / "coverage" / "run_manifest.json").read_text())
    assert "metadata" in failed["error"] or "cover prior official fixtures" in failed["error"]


def test_launch_parse_failure_preserves_previous_pointer(wiring, monkeypatch):
    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()

    def broken_launch(**kwargs):
        raise ValueError("invalid cached snapshot JSON")

    monkeypatch.setattr(ops, "check_season_launch", broken_launch)
    result = wiring.run("launch_failure")
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert len(wiring.calls) == 1


def test_noop_keeps_immutable_run_artifacts(wiring):
    first = wiring.run()
    before = {p.name: p.read_bytes() for p in first.run_dir.iterdir()}
    result = wiring.run("different_id")
    assert result.no_op
    assert {p.name: p.read_bytes() for p in first.run_dir.iterdir()} == before
    assert not (wiring.paths.output_dir / ".tmp_different_id").exists()


def test_all_prior_events_blank_can_reconstruct_empty_history(wiring, monkeypatch, tmp_path):
    from test_phase9b3_current_season import _official_case, _patch_team_identities
    from fpl_forecast.operations.current_panel import reconstruct_completed_current_season

    case = _official_case(tmp_path / "blank", target_gameweek=2, blank_events={1})
    _patch_team_identities(monkeypatch)
    monkeypatch.setattr(
        ops, "reconstruct_completed_current_season", reconstruct_completed_current_season
    )
    result = ops.refresh_operational(
        season="2026-27",
        target_gameweek=2,
        offline=True,
        run_id="blank",
        raw_fpl_dir=case["raw_fpl_dir"],
        normalized_dir=case["normalized_dir"],
        operational_root=wiring.paths.output_dir,
    )
    assert result.status.state.value == "SUCCEEDED"
    assert wiring.forwarded[0]["completed_player_fixtures"].empty
    assert wiring.forwarded[0]["completed_team_fixtures"].empty


@pytest.mark.parametrize("fault", ["candidate_write", "pointer_replace", "rollback"])
def test_publication_fault_preserves_pointer_and_reports_evidence(wiring, monkeypatch, fault):
    from pathlib import Path

    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()
    original_write = Path.write_text
    original_replace = Path.replace
    original_rename = Path.rename
    candidate = wiring.paths.latest_successful_path.with_suffix(".json.tmp")
    final = wiring.paths.runs_dir / "publication_fault"

    def write(path, *args, **kwargs):
        if path == candidate and fault == "candidate_write":
            raise OSError("pointer candidate write failed")
        return original_write(path, *args, **kwargs)

    def replace_pointer(path, target):
        if path == candidate:
            assert final.is_dir()  # Fault after staging has been renamed.
            raise OSError("pointer replacement failed")
        return original_replace(path, target)

    def rename(path, target):
        if path == final and fault == "rollback":
            raise OSError("rollback rename failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(Path, "replace", replace_pointer)
    monkeypatch.setattr(Path, "rename", rename)
    result = wiring.run("publication_fault", force=True)
    assert result.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert wiring.paths.latest_successful_path.read_bytes() == before
    assert not candidate.exists()
    if fault == "rollback":
        assert final.is_dir()
        assert result.status.extra["unreferenced_run_dir"] == str(final)
        assert "rollback failed" in result.status.warning
    else:
        assert not final.exists()
        assert (wiring.paths.failed_dir / "publication_fault" / "run_manifest.json").is_file()
    assert not (wiring.paths.output_dir / ".tmp_publication_fault").exists()


def test_committed_publication_has_atomic_fingerprint_without_second_write(wiring, monkeypatch):
    from pathlib import Path

    original = Path.write_text

    def guard(path, *args, **kwargs):
        if path == wiring.paths.latest_successful_path:
            pytest.fail("non-atomic second pointer write")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guard)
    result = wiring.run()
    assert result.status.state.value == "SUCCEEDED"
    pointer = json.loads(wiring.paths.latest_successful_path.read_text())
    manifest = json.loads(result.manifest_path.read_text())
    assert pointer["input_fingerprint"] == manifest["input_fingerprint"]
    assert wiring.run("reuse").no_op


def test_status_storage_failure_after_commit_does_not_claim_old_pointer(wiring, monkeypatch):
    wiring.run()
    before = wiring.paths.latest_successful_path.read_bytes()

    def unavailable(*args, **kwargs):
        raise OSError("status storage unavailable")

    monkeypatch.setattr(ops, "write_status", unavailable)
    result = wiring.run("committed", force=True)
    assert result.status.state.value == "SUCCEEDED"
    assert result.status.latest_successful_run_id == "committed"
    assert "Publication committed" in result.status.warning
    assert wiring.paths.latest_successful_path.read_bytes() != before
    assert json.loads(wiring.paths.latest_successful_path.read_text())["run_id"] == "committed"
    assert result.run_dir.is_dir()
    assert not (wiring.paths.failed_dir / "committed").exists()


def test_pointer_candidate_cleanup_failure_cannot_undo_success(wiring, monkeypatch):
    from pathlib import Path

    original = Path.unlink
    candidate = wiring.paths.latest_successful_path.with_suffix(".json.tmp")

    def cleanup(path, *args, **kwargs):
        if path == candidate:
            raise OSError("candidate cleanup unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", cleanup)
    result = wiring.run()
    assert result.status.state.value == "SUCCEEDED"
    assert json.loads(wiring.paths.latest_successful_path.read_text())["run_id"] == "first"


@pytest.mark.parametrize("grain", ["player", "team"])
def test_reconstructed_histories_reject_duplicate_observations(monkeypatch, tmp_path, grain):
    from test_phase9b3_current_season import _official_case, _patch_team_identities
    from fpl_forecast.operations import current_panel

    case = _official_case(tmp_path / "duplicates", target_gameweek=2)
    _patch_team_identities(monkeypatch)
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    builder = f"build_current_{grain}_fixture_history"
    real_builder = getattr(current_panel, builder)
    monkeypatch.setattr(current_panel, builder, capture)
    with pytest.raises(ValueError, match="cover prior official fixtures"):
        current_panel.reconstruct_completed_current_season(**case)
    if grain == "team":
        captured["fixtures"] = pd.concat([captured["fixtures"], captured["fixtures"]])
        with pytest.raises(ValueError, match="duplicate fixture observations"):
            real_builder(**captured)
    else:
        # Identical official live rows are deduplicated, conflicting ones fail.
        base = real_builder(**captured)
        captured["live_results"] = pd.concat([captured["live_results"], captured["live_results"]])
        assert len(real_builder(**captured)) == len(base)
        captured["live_results"].iloc[-1, captured["live_results"].columns.get_loc("minutes")] += 1
        with pytest.raises(ValueError, match="conflicting duplicate"):
            real_builder(**captured)
