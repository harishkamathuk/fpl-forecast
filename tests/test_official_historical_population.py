"""Official historical population is independent of a later current roster."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from fpl_forecast.operations import prior_event_snapshots as provision
from test_prior_event_roster import _collection, _validate
from test_prior_event_snapshots import OfficialCase, inventory


def official_case(tmp_path, monkeypatch, *, truncate=False):
    case = OfficialCase(tmp_path, monkeypatch)
    record, _ = _collection(case, population=610, truncate=truncate)
    metadata = json.loads(record.metadata_path.read_text())
    metadata.pop("historical_roster_manifest")
    metadata.pop("historical_roster_manifest_sha256")
    record.metadata_path.write_text(json.dumps(metadata))
    template = case.bootstrap["elements"][0]
    case.bootstrap["elements"].extend(
        {**template, "id": i, "code": 1000+i} for i in range(612, 671)
    )
    case.retrieved += timedelta(seconds=1)
    case.context()
    return case


def test_official_610_ids_against_later_670_without_manifest(tmp_path, monkeypatch):
    case = official_case(tmp_path, monkeypatch)
    frames = []
    original = provision.normalize_event_live
    def observe(**kwargs):
        frame = original(**kwargs)
        frames.append(frame)
        return frame
    monkeypatch.setattr(provision, "normalize_event_live", observe)
    before = inventory(tmp_path)
    _validate(case)
    assert len(case.bootstrap["elements"]) == 670
    assert set(frames[0]["player_id"]) == set(range(1, 611))
    assert len(frames[0]) == 610
    assert inventory(tmp_path) == before
    assert not case.calls


def test_missing_official_fixture_stat_player_is_detected(tmp_path, monkeypatch):
    case = official_case(tmp_path, monkeypatch, truncate=True)
    case.fixtures[0]["stats"] = [
        {"identifier": "bps", "h": [{"element": 1, "value": 20}], "a": []}
    ]
    case.retrieved += timedelta(seconds=1)
    case.context()
    with pytest.raises(ValueError, match="fixture-stat players missing"):
        _validate(case)


def test_omitting_every_player_from_a_fixture_is_detected(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    payload = case.live(1)
    payload["elements"] = [r for r in payload["elements"] if r["id"] not in (19, 20)]
    case.write("event_live_1", payload)
    with pytest.raises(ValueError, match="fixture coverage"):
        _validate(case)


def test_unknown_historical_identity_is_not_invented(tmp_path, monkeypatch):
    case = OfficialCase(tmp_path, monkeypatch)
    payload = case.live(1)
    payload["elements"][0]["id"] = 99999
    case.write("event_live_1", payload)
    with pytest.raises(ValueError, match="unknown official player IDs"):
        _validate(case)
