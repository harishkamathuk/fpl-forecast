"""Isolated official-evidence provisioning tests; no real HTTP or model execution."""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from fpl_forecast.ingest.fpl_api import FPLApiClient
from fpl_forecast.ingest.snapshots import (
    latest_snapshot_path,
    metadata_for_raw_path,
    write_raw_snapshot,
)
from fpl_forecast.normalize.current import normalize_current
from fpl_forecast.operations.prior_event_snapshots import snapshot_prior_event_live
from fpl_forecast.operations.live_results import VALUE_COMPONENTS

SEASON = "2026-27"


class OfficialCase:
    def __init__(self, root, monkeypatch, target=2, blank=(), double=False):
        self.raw = root / "raw"
        self.normalized = root / "normalized"
        self.target = target
        self.now = (
            pd.Timestamp("2026-08-15T11:00:00Z")
            + pd.Timedelta(weeks=target - 1, days=-1)
        ).to_pydatetime()
        self.retrieved = self.now - timedelta(hours=1)
        self.calls = []
        self.responses = {}
        self.summary_responses = {}
        self.bootstrap = {
            "events": [
                dict(
                    id=gw,
                    name=f"Gameweek {gw}",
                    deadline_time=(
                        pd.Timestamp("2026-08-15T11:00:00Z")
                        + pd.Timedelta(weeks=gw - 1)
                    ).isoformat(),
                    finished=gw < target,
                    data_checked=gw < target,
                    is_current=gw == target - 1,
                    is_next=gw == target,
                    released=True,
                )
                for gw in range(1, 39)
            ],
            "teams": [
                dict(id=i, code=i, name=f"Club {i}", short_name=f"C{i}", strength=3)
                for i in range(1, 21)
            ],
            "elements": [
                dict(
                    id=i,
                    code=1000 + i,
                    web_name=f"Name {i}",
                    team=i,
                    element_type=3,
                    now_cost=50,
                )
                for i in range(1, 21)
            ],
            "game_settings": {},
            "phases": [],
            "total_players": 20,
            "element_stats": [],
            "element_types": [],
        }
        self.fixtures = [
            dict(
                id=(gw - 1) * 10 + match + 1,
                code=(gw - 1) * 10 + match + 1,
                event=gw if gw not in blank else 38,
                team_h=match * 2 + 1,
                team_a=match * 2 + 2,
                kickoff_time=(
                    pd.Timestamp("2026-08-15T15:00:00Z")
                    + pd.Timedelta(weeks=(gw if gw not in blank else 38) - 1)
                ).isoformat(),
                finished=gw < target and gw not in blank,
                finished_provisional=gw < target and gw not in blank,
                started=gw < target and gw not in blank,
                team_h_score=0 if gw < target and gw not in blank else None,
                team_a_score=0 if gw < target and gw not in blank else None,
            )
            for gw in range(1, 39)
            for match in range(10)
        ]
        if double:
            self.fixtures[-1].update(
                event=1,
                finished=True,
                finished_provisional=True,
                started=True,
                team_h_score=0,
                team_a_score=0,
                kickoff_time=self.fixtures[0]["kickoff_time"],
            )
        self.client = FPLApiClient(
            raw_dir=self.raw, transport=httpx.MockTransport(self.handle)
        )
        monkeypatch.setattr(
            "fpl_forecast.ingest.snapshots.utc_now", lambda: self.retrieved
        )
        self.context()

    def context(self):
        self.write("bootstrap_static", self.bootstrap)
        self.write("fixtures", self.fixtures)
        normalize_current(
            season=SEASON, raw_dir=self.raw, normalized_dir=self.normalized
        )

    def url(self, endpoint):
        if endpoint == "bootstrap_static":
            return self.client.bootstrap_static_url()
        if endpoint == "fixtures":
            return self.client.fixtures_url()
        return self.client.event_live_url(int(endpoint.rsplit("_", 1)[1]))

    def write(self, endpoint, payload, at=None):
        return write_raw_snapshot(
            self.raw,
            season=SEASON,
            endpoint_name=endpoint,
            content=json.dumps(payload).encode(),
            source_url=self.url(endpoint),
            http_status=200,
            source="fpl_api",
            retrieved_at=at or self.retrieved,
        )

    def live(self, gw):
        elements = []
        for player in self.bootstrap["elements"]:
            fixtures = [
                f
                for f in self.fixtures
                if f["event"] == gw and player["team"] in (f["team_h"], f["team_a"])
            ]
            elements.append(
                dict(
                    id=player["id"],
                    stats={
                        **dict.fromkeys(VALUE_COMPONENTS, 0),
                        "minutes": 90 * len(fixtures),
                        "starts": len(fixtures),
                        "total_points": 2 * len(fixtures),
                    },
                    explain=[
                        dict(
                            fixture=f["id"],
                            stats=[dict(identifier="minutes", value=90, points=2)],
                        )
                        for f in fixtures
                    ],
                )
            )
        return {"elements": elements}

    def handle(self, request):
        self.calls.append(request.url.path)
        if "/element-summary/" in request.url.path:
            player_id = int(request.url.path.rstrip("/").split("/")[-1])
            return self.summary_responses.get(player_id, httpx.Response(404))
        gw = int(request.url.path.split("/")[-3])
        response = self.responses.get(gw)
        if isinstance(response, Exception):
            raise response
        return response or httpx.Response(200, json=self.live(gw))

    def run(self, **kwargs):
        return snapshot_prior_event_live(
            season=SEASON,
            target_gameweek=self.target,
            raw_fpl_dir=self.raw,
            normalized_dir=self.normalized,
            client=self.client,
            clock=lambda: self.now,
            **kwargs,
        )

    def path(self, endpoint):
        return latest_snapshot_path(
            self.raw, season=SEASON, endpoint_name=endpoint, content_type="json"
        )


def inventory(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "target,blank,double",
    [(1, (), False), (2, (), False), (3, (2,), False), (2, (), True)],
)
def test_retrieve_missing_and_reuse_exact_cache(
    tmp_path, monkeypatch, target, blank, double
):
    case = OfficialCase(tmp_path, monkeypatch, target, blank, double)
    normalized = inventory(case.normalized)
    records = case.run()
    assert [r.endpoint_name for r in records] == [
        f"event_live_{i}" for i in range(1, target)
    ]
    assert case.calls == [f"/api/event/{i}/live/" for i in range(1, target)]
    before = inventory(tmp_path)
    assert case.run() == records
    assert inventory(tmp_path) == before
    assert inventory(case.normalized) == normalized
    assert len(case.calls) == target - 1
    assert all(
        pd.Timestamp(r.retrieved_at) == pd.Timestamp(case.retrieved) for r in records
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "other"),
        ("source_url", "https://example.com"),
        ("endpoint_name", "event_live_2"),
        ("season", "2025-26"),
        ("target_season", "2025-26"),
        ("http_status", 500),
        ("source_version", "2025-26"),
        ("sha256", "bad"),
        ("content_length", 1),
        ("raw_path", "elsewhere"),
        ("retrieved_at", "invalid"),
        ("retrieved_at", "2026-08-20T10:00:00"),
        ("retrieved_at", "2026-08-30T00:00:00Z"),
    ],
)
def test_corrupt_latest_never_falls_back_or_fetches(
    tmp_path, monkeypatch, field, value
):
    case = OfficialCase(tmp_path, monkeypatch)
    case.write("event_live_1", case.live(1), case.retrieved - timedelta(hours=1))
    latest = case.write("event_live_1", case.live(1))
    sidecar = latest.metadata_path
    metadata = json.loads(sidecar.read_text())
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata))
    before = inventory(tmp_path)
    with pytest.raises(ValueError, match="GW1"):
        case.run()
    assert not case.calls
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("endpoint", ["bootstrap_static", "fixtures", "event_live_1"])
@pytest.mark.parametrize("mode", ["missing_metadata", "corrupt_content", "at_deadline"])
def test_all_official_sources_validated(tmp_path, monkeypatch, endpoint, mode):
    case = OfficialCase(tmp_path, monkeypatch)
    case.write("event_live_1", case.live(1))
    path = case.path(endpoint)
    if mode == "missing_metadata":
        metadata_for_raw_path(path).unlink()
    elif mode == "corrupt_content":
        path.write_bytes(b"{}")
    else:
        payload = json.loads(path.read_text())
        case.write(
            endpoint,
            payload,
            pd.Timestamp(case.bootstrap["events"][1]["deadline_time"]).to_pydatetime(),
        )
        case.now = pd.Timestamp(
            case.bootstrap["events"][1]["deadline_time"]
        ).to_pydatetime()
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        case.run()
    assert not case.calls


@pytest.mark.parametrize(
    "table,column,value",
    [
        ("events", "deadline_time", "2026-08-01T00:00:00Z"),
        ("events", "data_checked", False),
        ("fixtures", "finished", False),
        ("fixtures", "team_h_score", 5),
        ("players", "team_id", 20),
        ("teams", "team_code", 999),
        ("events", "raw_snapshot_path", "older.json"),
    ],
)
def test_normalized_contradictions_fail_before_fetch(
    tmp_path, monkeypatch, table, column, value
):
    case = OfficialCase(tmp_path, monkeypatch)
    path = case.normalized / SEASON / f"current_{table}.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, column] = value
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Normalized"):
        case.run()
    assert not case.calls


@pytest.mark.parametrize(
    "defect",
    [
        "started_target",
        "completed_target",
        "blank_target",
        "missing_event",
        "duplicate_event",
        "fractional_event",
        "naive_deadline",
        "wrong_season",
        "missing_score",
    ],
)
def test_official_schedule_defects_fail_before_fetch(tmp_path, monkeypatch, defect):
    case = OfficialCase(tmp_path, monkeypatch)
    if defect == "unfinished_event":
        case.bootstrap["events"][0]["finished"] = False
    elif defect == "unchecked_event":
        case.bootstrap["events"][0]["data_checked"] = False
    elif defect == "unfinished_fixture":
        case.fixtures[0]["finished"] = False
    elif defect == "provisional":
        case.fixtures[0]["finished_provisional"] = False
    elif defect == "started_target":
        case.fixtures[10]["started"] = True
    elif defect == "completed_target":
        case.bootstrap["events"][1]["finished"] = True
    elif defect == "blank_target":
        for fixture in case.fixtures:
            if fixture["event"] == 2:
                fixture["event"] = 38
    elif defect == "missing_event":
        case.bootstrap["events"].pop(0)
    elif defect == "duplicate_event":
        case.bootstrap["events"][0]["id"] = 2
    elif defect == "fractional_event":
        case.bootstrap["events"][0]["id"] = 1.5
    elif defect == "naive_deadline":
        case.bootstrap["events"][1]["deadline_time"] = "2026-08-22T11:00:00"
    elif defect == "wrong_season":
        for fixture in case.fixtures:
            fixture["kickoff_time"] = "2025-08-15T15:00:00Z"
    elif defect == "missing_score":
        case.fixtures[0]["team_h_score"] = None
    case.retrieved += timedelta(seconds=1)
    # Some corrupt raw inputs cannot even be normalized; raw validity still gates provisioning.
    try:
        case.context()
    except (ValueError, TypeError):
        pass
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        case.run()
    assert not case.calls


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"elements": []}),
        httpx.ReadTimeout("timeout"),
    ],
)
def test_http_and_empty_evidence_fail_closed(tmp_path, monkeypatch, response):
    case = OfficialCase(tmp_path, monkeypatch)
    # Avoid retry delays for the timeout regression.
    monkeypatch.setattr("fpl_forecast.ingest.fpl_api.time.sleep", lambda _: None)
    case.responses[1] = response
    before = inventory(case.normalized)
    with pytest.raises(ValueError, match="GW1"):
        case.run()
    assert inventory(case.normalized) == before


@pytest.mark.parametrize(
    "defect",
    [
        "missing_player",
        "duplicate",
        "wrong_fixture",
        "wrong_event",
        "missing_stats",
        "missing_component",
        "absent_explain",
        "missing_awarded_points",
        "unsupported_component",
        "scoring",
        "missing_fixture",
        "missing_player_fixture",
    ],
)
def test_player_and_fixture_evidence_rejected(tmp_path, monkeypatch, defect):
    case = OfficialCase(tmp_path, monkeypatch, double=True)
    payload = case.live(1)
    if defect == "missing_player":
        removed = payload["elements"].pop()
        # Detect truncation from an independent official result reference,
        # rather than assuming the latest bootstrap is the historical roster.
        case.fixtures[9]["stats"] = [{"identifier": "bps", "h": [],
                                      "a": [{"element": removed["id"], "value": 1}]}]
        case.retrieved += timedelta(seconds=1)
        case.context()
    elif defect == "duplicate":
        payload["elements"].append(payload["elements"][0])
    elif defect == "wrong_fixture":
        payload["elements"][0]["explain"][0]["fixture"] = 9999
    elif defect == "wrong_event":
        payload["elements"][0]["explain"][0]["fixture"] = 11
    elif defect == "missing_stats":
        payload["elements"][0]["stats"] = {}
    elif defect == "missing_component":
        payload["elements"][0]["stats"].pop("goals_scored")
    elif defect == "absent_explain":
        payload["elements"][0]["explain"] = []
    elif defect == "missing_awarded_points":
        payload["elements"][0]["explain"][0]["stats"][0].pop("points")
    elif defect == "unsupported_component":
        payload["elements"][0]["explain"][0]["stats"][0]["identifier"] = (
            "new_scoring_rule"
        )
    elif defect == "scoring":
        payload["elements"][0]["stats"]["total_points"] += 1
    else:
        # Drop one double-GW row while retaining reconciled surviving totals.
        affected = (
            payload["elements"][-2:]
            if defect == "missing_fixture"
            else payload["elements"][-1:]
        )
        removed_fixture = affected[-1]["explain"][-1]["fixture"]
        fixture = next(f for f in case.fixtures if f["id"] == removed_fixture)
        fixture["stats"] = [{"identifier": "bps", "h": [],
                             "a": [{"element": affected[-1]["id"], "value": 1}]}]
        case.retrieved += timedelta(seconds=1)
        case.context()
        for player in affected:
            player["explain"].pop()
            player["stats"].update(total_points=2, minutes=90, starts=1)
    case.write("event_live_1", payload)
    with pytest.raises(ValueError, match="GW1"):
        case.run()
    assert not case.calls


def test_blank_evidence_cannot_hide_outcomes(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch, target=3, blank=(2,))
    payload = case.live(2)
    payload["elements"][0]["stats"]["total_points"] = 1
    case.write("event_live_2", payload)
    with pytest.raises(ValueError, match="blank event"):
        case.run()


def test_partial_retrieval_preserves_existing_outputs(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch, target=3)
    outputs = tmp_path / "operational"
    outputs.mkdir()
    (outputs / "latest_successful.json").write_text('{"run_id":"historical"}')
    (outputs / "forecast.csv").write_bytes(b"immutable history")
    before = inventory(outputs)
    case.responses[2] = httpx.Response(404)
    with pytest.raises(ValueError, match="GW2"):
        case.run()
    assert case.path("event_live_1") is not None
    assert inventory(outputs) == before
    assert not list(case.normalized.rglob("*history*"))


def test_crossing_deadline_and_context_drift_fail(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    original = case.handle

    def drift(request):
        response = original(request)
        case.now = pd.Timestamp(
            case.bootstrap["events"][1]["deadline_time"]
        ).to_pydatetime()
        return response

    case.client.transport = httpx.MockTransport(drift)
    with pytest.raises(ValueError, match="deadline reached"):
        case.run()
    case.now -= timedelta(days=1)
    # Change normalized evidence after the initial context check, during live loading.
    original_fetch = case.client.fetch_event_live

    def change(**kwargs):
        record = original_fetch(**kwargs)
        path = case.normalized / SEASON / "current_events.parquet"
        path.write_bytes(path.read_bytes() + b"drift")
        return record

    monkeypatch.setattr(case.client, "fetch_event_live", change)
    with pytest.raises(ValueError, match="Evidence changed"):
        case.run()


def test_cli_is_narrow_and_offline_option_preserved(tmp_path, monkeypatch):
    from fpl_forecast import cli

    calls = []
    monkeypatch.setattr(
        cli, "snapshot_prior_event_live", lambda **kw: calls.append(kw) or []
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "snapshot-prior-event-live",
            "--season",
            SEASON,
            "--target-gameweek",
            "2",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--normalized-dir",
            str(tmp_path / "normalized"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["target_gameweek"] == 2
    assert calls[0]["raw_fpl_dir"] == tmp_path / "raw"
    monkeypatch.setattr(
        cli,
        "snapshot_prior_event_live",
        lambda **kw: (_ for _ in ()).throw(ValueError("GW1 missing")),
    )
    result = runner.invoke(
        cli.app,
        ["snapshot-prior-event-live", "--season", SEASON, "--target-gameweek", "2"],
    )
    assert result.exit_code == 1
    assert "GW1 missing" in result.output
    from types import SimpleNamespace

    forwarded = []
    monkeypatch.setattr(
        cli,
        "refresh_operational_run",
        lambda **kw: (
            forwarded.append(kw)
            or SimpleNamespace(
                status=SimpleNamespace(
                    state=SimpleNamespace(value="SUCCEEDED"), reason="stub"
                ),
                run_id="stub",
                run_dir=tmp_path,
                manifest_path=tmp_path / "manifest",
                no_op=True,
            )
        ),
    )
    result = runner.invoke(
        cli.app,
        [
            "refresh-operational",
            "--season",
            SEASON,
            "--target-gameweek",
            "2",
            "--offline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert forwarded[0]["offline"] is True


@pytest.mark.parametrize("offset,valid", [(-1, True), (0, False), (1, False)])
@pytest.mark.parametrize(
    "source_name", ["bootstrap_static", "fixtures", "event_live_1"]
)
def test_strict_deadline_for_each_official_source(
    tmp_path, monkeypatch, offset, valid, source_name
):
    from dataclasses import replace
    from fpl_forecast.ingest.fpl_api import load_latest_fpl_snapshot
    from fpl_forecast.operations.prior_event_snapshots import _validate_live

    case = OfficialCase(tmp_path, monkeypatch)
    case.write("event_live_1", case.live(1))
    sources = {
        endpoint: load_latest_fpl_snapshot(
            raw_dir=case.raw, season=SEASON, endpoint_name=endpoint
        )
        for endpoint in ("bootstrap_static", "fixtures", "event_live_1")
    }
    cutoff = pd.Timestamp(case.bootstrap["events"][1]["deadline_time"])
    source = sources[source_name]
    sources[source_name] = replace(
        source,
        metadata={
            **source.metadata,
            "retrieved_at": (cutoff + pd.Timedelta(microseconds=offset)).isoformat(),
        },
    )
    kwargs = dict(
        season=SEASON,
        gameweek=1,
        bootstrap=sources["bootstrap_static"],
        fixtures=sources["fixtures"],
        live=sources["event_live_1"],
        cutoff=cutoff,
    )
    if valid:
        _validate_live(**kwargs)
    else:
        with pytest.raises(ValueError, match="before target deadline"):
            _validate_live(**kwargs)


def test_provision_then_offline_orchestrator_with_stubbed_forecast(
    tmp_path, monkeypatch
):
    from fpl_forecast.operations import orchestrator as ops

    case = OfficialCase(tmp_path, monkeypatch)
    case.run()
    paths = ops.resolve_operational_paths(tmp_path / "operational")
    captured = []
    monkeypatch.setattr(
        ops, "_source_state", lambda: {"dirty": False, "commit": "fixed"}
    )
    monkeypatch.setattr(
        ops, "check_season_launch", lambda **kw: ops._mock_launch_check(SEASON)
    )
    monkeypatch.setattr(
        ops, "_build_frontend_artifacts", lambda *args, **kw: captured.append(kw) or {}
    )
    monkeypatch.setattr(ops, "_validate_frontend_artifacts", lambda _: None)
    monkeypatch.setattr(
        ops,
        "run_operational_model_chain",
        lambda **kw: pytest.fail("real model executed"),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.current_panel._current_team_identities",
        lambda teams, **kw: teams.assign(
            team_uid=teams["team_id"].map(lambda x: f"team_{x}")
        ),
    )

    def refresh(run_id):
        return ops.refresh_operational(
            season=SEASON,
            target_gameweek=2,
            offline=True,
            run_id=run_id,
            raw_fpl_dir=case.raw,
            normalized_dir=case.normalized,
            operational_root=paths.output_dir,
        )

    first = refresh("first")
    assert first.status.state.value == "SUCCEEDED", first.status
    assert len(captured[0]["completed_player_fixtures"]) == 20
    assert len(captured[0]["completed_team_fixtures"]) == 10
    assert (first.run_dir / "current_season_reconstruction.json").exists()
    historical = inventory(first.run_dir)
    repeat = refresh("repeat")
    assert repeat.no_op is True
    assert len(captured) == 1
    revised = case.live(1)
    for player in revised["elements"]:
        player["stats"].update(total_points=3, bonus=1)
        player["explain"][0]["stats"].append(
            dict(identifier="bonus", value=1, points=1)
        )
    case.write("event_live_1", revised, case.retrieved + timedelta(seconds=1))
    case.run()
    changed = refresh("revised")
    assert changed.status.state.value == "SUCCEEDED"
    assert not changed.no_op
    assert len(captured) == 2
    pointer = paths.latest_successful_path.read_bytes()
    for path in (case.raw / SEASON / "event_live_1").glob("*"):
        path.unlink()
    failed = refresh("missing")
    assert failed.status.state.value == "FAILED_USING_LAST_SUCCESS"
    assert paths.latest_successful_path.read_bytes() == pointer
    assert inventory(first.run_dir) == historical
    assert len(captured) == 2
    assert case.calls == ["/api/event/1/live/"]


@pytest.mark.parametrize("target", [0, 39, 1.5, True, "2"])
def test_invalid_target_is_rejected(tmp_path, monkeypatch, target):
    case = OfficialCase(tmp_path, monkeypatch)
    case.target = target
    with pytest.raises(ValueError, match="integer from 1 to 38"):
        case.run()
    assert not case.calls


def test_dnp_and_non_utc_retrieval_and_http_date(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    payload = case.live(1)
    payload["elements"][0]["stats"] = {
        **dict.fromkeys(VALUE_COMPONENTS, 0),
        "starts": 0,
        "total_points": 0,
    }
    payload["elements"][0]["explain"] = [{"fixture": 1, "stats": []}]
    case.responses[1] = httpx.Response(
        200, json=payload, headers={"Date": "Sun, 01 Aug 1999 00:00:00 GMT"}
    )
    records = case.run()
    assert pd.Timestamp(records[0].retrieved_at) == pd.Timestamp(case.retrieved)
    path = metadata_for_raw_path(records[0].raw_path)
    metadata = json.loads(path.read_text())
    metadata["retrieved_at"] = (
        pd.Timestamp(metadata["retrieved_at"]).tz_convert("Europe/London").isoformat()
    )
    path.write_text(json.dumps(metadata))
    assert case.run()[0].retrieved_at == metadata["retrieved_at"]
    assert len(case.calls) == 1


def test_latest_snapshot_change_fails(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    original = case.client.fetch_event_live

    def change(**kwargs):
        record = original(**kwargs)
        case.write("fixtures", case.fixtures, case.retrieved + timedelta(seconds=1))
        return record

    monkeypatch.setattr(case.client, "fetch_event_live", change)
    with pytest.raises(ValueError, match="Selected official snapshot changed"):
        case.run()


def test_retrieval_before_fixture_kickoff_fails(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    kickoff = pd.Timestamp(case.fixtures[0]["kickoff_time"]).to_pydatetime()
    case.write("event_live_1", case.live(1), kickoff)
    with pytest.raises(ValueError, match="predates fixture kickoff"):
        case.run()
    assert not case.calls


def test_changed_client_record_fails(tmp_path, monkeypatch):
    from dataclasses import replace

    case = OfficialCase(tmp_path, monkeypatch)
    original = case.client.fetch_event_live
    monkeypatch.setattr(
        case.client,
        "fetch_event_live",
        lambda **kw: replace(original(**kw), checksum_sha256="corrupt"),
    )
    with pytest.raises(ValueError, match="changed during retrieval"):
        case.run()
