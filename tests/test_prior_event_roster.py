"""Narrow historical-population regressions; only temporary evidence and mocked HTTP."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import timedelta

import pandas as pd
import pytest

from fpl_forecast.operations import prior_event_snapshots as provision
from test_prior_event_snapshots import OfficialCase, SEASON, inventory


def _collection(case, *, gameweek=1, population=20, truncate=False):
    if population != 20:
        template = case.bootstrap["elements"][0]
        case.bootstrap["elements"] = [
            {**template, "id": i, "code": 1000 + i, "team": (i - 1) % 20 + 1}
            for i in range(1, population + 1)
        ]
        case.retrieved += timedelta(seconds=1)
        case.context()
    historical = deepcopy(case.bootstrap)
    payload = case.live(gameweek)
    if truncate:
        payload["elements"].pop(0)
    record = case.write(f"event_live_{gameweek}", payload)
    folder = case.raw.parent / "recorded-collection"
    folder.mkdir()
    files = {
        "01-bootstrap-static.json": json.dumps(historical).encode(),
        f"02-gw{gameweek:02d}-live.json": record.raw_path.read_bytes(),
        f"04-gw{gameweek:02d}-fixtures.json": json.dumps(
            [r for r in case.fixtures if r["event"] == gameweek]
        ).encode(),
    }
    manifest = {
        "collector": "Touchline public API",
        "api_base": "https://fantasy.premierleague.com/api",
        "season": SEASON,
        "gameweek": gameweek,
        "collected_at_utc": case.retrieved.isoformat(),
        "files": {},
    }
    for name, raw in files.items():
        (folder / name).write_bytes(raw)
        manifest["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    path = folder / "09-manifest.json"
    path.write_text(json.dumps(manifest))
    _pin(record, path)
    # The later bootstrap contains one player without any historical result.
    template = case.bootstrap["elements"][0]
    case.bootstrap["elements"].append({**template, "id": population + 1, "code": 1001 + population})
    case.retrieved += timedelta(seconds=1)
    case.context()
    return record, path


def _pin(record, path):
    metadata = json.loads(record.metadata_path.read_text())
    metadata.update(
        historical_roster_manifest=str(path),
        historical_roster_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    record.metadata_path.write_text(json.dumps(metadata))


def _validate(case, gw=1):
    def load(endpoint, url):
        return provision._load_verified(
            raw_dir=case.raw, season=SEASON, endpoint=endpoint, url=url, now=pd.Timestamp(case.now)
        )[0]

    evidence = {}
    provision._validate_live(
        season=SEASON,
        gameweek=gw,
        live=load(f"event_live_{gw}", case.client.event_live_url(gw)),
        bootstrap=load("bootstrap_static", case.client.bootstrap_static_url()),
        fixtures=load("fixtures", case.client.fixtures_url()),
        cutoff=pd.Timestamp(case.bootstrap["events"][case.target - 1]["deadline_time"]),
        evidence=evidence,
    )
    return evidence


def test_gw8_669_matches_captured_roster_not_later_670(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch, target=9)
    record, manifest = _collection(case, gameweek=8, population=669)
    seen = []
    real_normalize = provision.normalize_event_live

    def observe(**kw):
        frame = real_normalize(**kw)
        seen.append(frame)
        return frame

    monkeypatch.setattr(provision, "normalize_event_live", observe)
    before = inventory(tmp_path)
    evidence = _validate(case, 8)
    assert len(case.bootstrap["elements"]) == 670
    assert set(seen[0]["player_id"]) == set(range(1, 670))
    assert len(seen[0]) == 669  # no fabricated row for the later ID 670
    assert manifest in evidence
    assert len(evidence) == 4  # manifest and all three original collection files
    assert inventory(tmp_path) == before
    assert not case.calls
    assert json.loads(record.metadata_path.read_text())["retrieved_at"] == record.retrieved_at


def test_gw8_genuinely_truncated_capture_rejected(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch, target=9)
    _collection(case, gameweek=8, population=669, truncate=True)
    with pytest.raises(ValueError, match="player coverage"):
        _validate(case, 8)
    assert not case.calls


def test_b2_uses_pinned_roster_without_retrieval_or_cache_mutation(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    _collection(case)
    before = inventory(tmp_path)
    records = case.run()
    assert len(records) == 1
    assert not case.calls
    assert inventory(tmp_path) == before


def test_official_response_does_not_require_a_collection_manifest(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    record, _ = _collection(case)
    metadata = json.loads(record.metadata_path.read_text())
    metadata.pop("historical_roster_manifest")
    record.metadata_path.write_text(json.dumps(metadata))
    case.run()
    assert not case.calls


@pytest.mark.parametrize(
    "defect",
    [
        "checksum",
        "missing_file",
        "source",
        "season",
        "event",
        "late",
        "future",
        "fixture",
        "identity",
        "manifest_live_binding",
    ],
)
def test_invalid_historical_evidence_fails_closed(tmp_path, monkeypatch, defect):
    case = OfficialCase(tmp_path, monkeypatch)
    record, path = _collection(case)
    data = json.loads(path.read_text())
    if defect == "checksum":
        path.write_bytes(path.read_bytes() + b" ")
    elif defect == "missing_file":
        (path.parent / "01-bootstrap-static.json").unlink()
    else:
        if defect == "source":
            data["api_base"] = "https://example.com/api"
        elif defect == "season":
            data["season"] = "2025-26"
        elif defect == "event":
            data["gameweek"] = 8
        elif defect == "late":
            data["collected_at_utc"] = case.bootstrap["events"][1]["deadline_time"]
        elif defect == "future":
            data["collected_at_utc"] = (case.now + timedelta(seconds=1)).isoformat()
        else:
            name = {
                "fixture": "04-gw01-fixtures.json",
                "identity": "01-bootstrap-static.json",
                "manifest_live_binding": "02-gw01-live.json",
            }[defect]
            payload = json.loads((path.parent / name).read_text())
            if defect == "fixture":
                payload[0]["team_h_score"] = 3
            elif defect == "identity":
                payload["elements"][0]["team"] = 3
            else:
                payload["elements"].pop()
            raw = json.dumps(payload).encode()
            (path.parent / name).write_bytes(raw)
            data["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        path.write_text(json.dumps(data))
        _pin(record, path)
    with pytest.raises((ValueError, FileNotFoundError)):
        case.run()
    assert not case.calls


def test_pinned_roster_does_not_bypass_scoring(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    # Record the same corrupt scoring in the live file and the manifest: trusted
    # population evidence must not turn internally inconsistent results into success.
    original = case.live

    def mismatch(gw):
        payload = original(gw)
        payload["elements"][0]["stats"]["total_points"] += 1
        return payload

    case.live = mismatch
    _collection(case)
    with pytest.raises(ValueError, match="scoring validation"):
        case.run()


def test_pinned_roster_is_rechecked_for_drift(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    _, path = _collection(case)
    original = provision._validate_live

    def change(**kwargs):
        original(**kwargs)
        path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(provision, "_validate_live", change)
    with pytest.raises(ValueError, match="Evidence changed"):
        case.run()


def test_pinned_roster_does_not_bypass_player_fixture_coverage(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch, double=True)
    original = case.live

    def incomplete(gw):
        payload = original(gw)
        player = payload["elements"][-1]
        player["explain"].pop()
        player["stats"].update(total_points=2, minutes=90, starts=1)
        return payload

    case.live = incomplete
    fixture = case.fixtures[-1]
    fixture["stats"] = [{"identifier": "bps", "h": [], "a": [{"element": 20, "value": 1}]}]
    case.retrieved += timedelta(seconds=1)
    case.context()
    _collection(case)
    with pytest.raises(ValueError, match="fixture coverage"):
        case.run()


def test_population_only_fix_preserves_offline_reconstruction(tmp_path, monkeypatch):
    from test_p1_compatibility import reconstruct

    case = OfficialCase(tmp_path, monkeypatch)
    _collection(case)
    case.run()
    result = reconstruct(case, monkeypatch)
    rows = pd.read_parquet(result.player_history_path)
    assert result.player_rows == 20
    assert set(rows["player_id"]) == set(range(1, 21))
    assert not case.calls
