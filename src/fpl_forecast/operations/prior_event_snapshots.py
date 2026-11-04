"""Provision official prior-event evidence without modeling or panel writes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, RAW_FPL_API_DIR
from fpl_forecast.ingest.fpl_api import (
    BOOTSTRAP_STATIC,
    FIXTURES,
    FPLApiClient,
    LoadedSnapshot,
)
from fpl_forecast.ingest.season import infer_and_validate_current_season
from fpl_forecast.ingest.snapshots import (
    SnapshotRecord,
    latest_snapshot_path,
    metadata_for_raw_path,
    timestamp_for_path,
)
from fpl_forecast.normalize.current import (
    _events_frame,
    _fixtures_frame,
    _players_frame,
    _teams_frame,
)
from fpl_forecast.operations.live_results import (
    POINT_COMPONENTS,
    VALUE_COMPONENTS,
    audit_event_live_scoring,
    normalize_event_live,
    validate_event_live_for_forecast,
)
from fpl_forecast.operations.target_gameweek import resolve_target_gameweek
from fpl_forecast.operations.temporal_clubs import HistoricalClubResolver


def _timestamp(value: object) -> pd.Timestamp:
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        raise ValueError(f"Invalid aware timestamp: {value!r}")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"Invalid aware timestamp: {value!r}")
    return stamp.tz_convert("UTC")


def _load_verified(
    *,
    raw_dir: Path,
    season: str,
    endpoint: str,
    url: str,
    now: pd.Timestamp,
) -> tuple[LoadedSnapshot, dict[Path, bytes]]:
    path = latest_snapshot_path(
        raw_dir, season=season, endpoint_name=endpoint, content_type="json"
    )
    if path is None:
        raise ValueError(f"Missing official snapshot {endpoint}")
    content = path.read_bytes()
    sidecar = metadata_for_raw_path(path)
    sidecar_bytes = sidecar.read_bytes()
    metadata = json.loads(sidecar_bytes)
    expected = {
        "source": "fpl_api",
        "endpoint_name": endpoint,
        "source_url": url,
        "season": season,
        "target_season": season,
        "sha256": hashlib.sha256(content).hexdigest(),
        "content_length": len(content),
        "raw_path": str(path),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{endpoint}: invalid snapshot provenance field {key}")
    if metadata.get("source_version") not in (None, season):
        raise ValueError(f"{endpoint}: contradictory source version")
    status = metadata.get("http_status")
    if type(status) is not int or not 200 <= status < 300:
        raise ValueError(f"{endpoint}: invalid HTTP status")
    retrieved = _timestamp(metadata.get("retrieved_at"))
    if retrieved > now:
        raise ValueError(f"{endpoint}: retrieval timestamp is in the future")
    if path.name != f"{timestamp_for_path(retrieved.to_pydatetime())}.json":
        raise ValueError(
            f"{endpoint}: retrieval timestamp contradicts snapshot filename"
        )
    payload = json.loads(content)
    FPLApiClient._validate_top_level_schema(endpoint, payload)
    return LoadedSnapshot(endpoint, payload, str(path), metadata), {
        path: content,
        sidecar: sidecar_bytes,
    }


def _normalized_context(
    *,
    season: str,
    normalized_dir: Path,
    bootstrap: LoadedSnapshot,
    fixtures: LoadedSnapshot,
) -> tuple[dict[str, pd.DataFrame], dict[Path, bytes]]:
    """Recreate tables in memory, checking values and lineage without writing panels."""
    frames = {}
    evidence = {}
    for name, source, builder, key in (
        ("events", bootstrap, _events_frame, "gameweek"),
        ("fixtures", fixtures, _fixtures_frame, "fixture_id"),
        ("players", bootstrap, _players_frame, "player_id"),
        ("teams", bootstrap, _teams_frame, "team_id"),
    ):
        path = normalized_dir / season / f"current_{name}.parquet"
        evidence[path] = path.read_bytes()
        actual = pd.read_parquet(path)
        expected = builder(
            source.payload,
            metadata=source.metadata,
            raw_path=source.raw_path,
            season=season,
        )
        if (
            expected.empty
            or expected[key].isna().any()
            or expected[key].duplicated().any()
        ):
            raise ValueError(f"Invalid official {name} identities")
        try:
            pd.testing.assert_frame_equal(
                actual[expected.columns].sort_values(key).reset_index(drop=True),
                expected.sort_values(key).reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except (AssertionError, KeyError) as exc:
            raise ValueError(
                f"Normalized current_{name} contradicts selected official snapshot"
            ) from exc
        frames[name] = expected
    return frames, evidence


def _validate_schedule(
    bootstrap: LoadedSnapshot, fixtures: LoadedSnapshot, *, target_gameweek: int,
) -> None:
    events = bootstrap.payload["events"]
    if any(type(row.get("id")) is not int for row in events):
        raise ValueError("Official event IDs must be integers")
    if sorted(row["id"] for row in events) != list(range(1, 39)):
        raise ValueError("Official events must uniquely cover gameweeks 1..38")
    for row in events:
        _timestamp(row.get("deadline_time"))
        for flag in ("finished", "data_checked", "is_current", "is_next"):
            if type(row.get(flag)) is not bool:
                raise ValueError(f"Official event {row['id']} has invalid {flag}")
    teams = bootstrap.payload["teams"]
    team_ids = {row["id"] for row in teams}
    if len(team_ids) != 20 or any(type(value) is not int for value in team_ids):
        raise ValueError("Invalid official team identities")
    for player in bootstrap.payload["elements"]:
        if (
            type(player.get("id")) is not int
            or type(player.get("team")) is not int
            or player["team"] not in team_ids
            or type(player.get("element_type")) is not int
            or player["element_type"] not in range(1, 6)
        ):
            raise ValueError("Invalid official player identity, team or position")
    for row in fixtures.payload:
        if type(row.get("id")) is not int or "event" not in row or "kickoff_time" not in row:
            raise ValueError("Invalid official fixture/event identity")
        event = row["event"]
        if event is not None and (type(event) is not int or event not in range(1, 39)):
            raise ValueError("Official fixture refers to an unknown event")
        if (
            row.get("team_h") not in team_ids
            or row.get("team_a") not in team_ids
            or row["team_h"] == row["team_a"]
        ):
            raise ValueError("Invalid official fixture sides")
        for flag in ("finished", "finished_provisional", "started"):
            if type(row.get(flag)) is not bool:
                raise ValueError(f"Official fixture {row['id']} has invalid {flag}")
        result_bearing = any(row[flag] for flag in ("finished", "finished_provisional", "started"))
        result_bearing |= any(row.get(score) is not None for score in ("team_h_score", "team_a_score"))
        if row["kickoff_time"] is not None:
            _timestamp(row["kickoff_time"])
        elif result_bearing or (event is not None and event <= target_gameweek):
            raise ValueError(f"Official fixture {row['id']} requires a known kickoff_time")
        if event is None and result_bearing:
            raise ValueError(f"Official fixture {row['id']} has results without an event")
        if (event is not None and event < target_gameweek
                and row["finished"] and row["finished_provisional"]
                and _timestamp(row["kickoff_time"]) >= _timestamp(next(event_row["deadline_time"] for event_row in events if event_row["id"] == target_gameweek))):
            raise ValueError("Prior official fixture is finalized at or after the target cutoff")
        if row["finished"]:
            for score in ("team_h_score", "team_a_score"):
                if type(row.get(score)) is not int or row[score] < 0:
                    raise ValueError(
                        f"Official fixture {row['id']} has invalid final scores"
                    )


def _recorded_roster(
    *,
    season: str,
    gameweek: int,
    live: LoadedSnapshot,
    bootstrap: LoadedSnapshot,
    fixtures: LoadedSnapshot,
    cutoff: pd.Timestamp,
) -> tuple[list[dict], dict[Path, bytes]]:
    """Verify one explicitly linked Touchline collection, not an older-cache fallback.

    The canonical live sidecar must pin historical_roster_manifest and its
    historical_roster_manifest_sha256. Preserve the collection's own timestamp;
    it is additional availability evidence, never a replacement retrieved_at.
    This supports population differences only, not historical club resolution.
    """
    reference = live.metadata.get("historical_roster_manifest")
    if not isinstance(reference, str) or not Path(reference).is_absolute():
        raise ValueError(
            f"GW{gameweek}: incomplete official player coverage; missing historical roster "
            "evidence: an explicitly linked contemporaneous collection manifest is required"
        )
    path = Path(reference)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"GW{gameweek}: missing historical roster manifest: {path}") from exc
    if hashlib.sha256(content).hexdigest() != live.metadata.get(
        "historical_roster_manifest_sha256"
    ):
        raise ValueError(f"GW{gameweek}: historical roster manifest checksum mismatch")
    manifest = json.loads(content)
    if (
        manifest.get("api_base") != "https://fantasy.premierleague.com/api"
        or manifest.get("season") != season
        or manifest.get("gameweek") != gameweek
    ):
        raise ValueError(f"GW{gameweek}: historical roster collection source/season/event mismatch")
    collected = _timestamp(manifest.get("collected_at_utc"))
    known_available = max(
        _timestamp(s.metadata["retrieved_at"]) for s in (bootstrap, fixtures, live)
    )
    if collected > known_available or collected >= cutoff:
        raise ValueError(f"GW{gameweek}: historical roster collection is future or late evidence")
    evidence = {path: content}

    def read(name: str):
        raw_path = path.parent / name
        raw = raw_path.read_bytes()
        entry = manifest["files"][name]
        if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != entry["bytes"]:
            raise ValueError(
                f"GW{gameweek}: historical roster collection file checksum mismatch: {name}"
            )
        evidence[raw_path] = raw
        return json.loads(raw)

    live_name = f"02-gw{gameweek:02d}-live.json"
    read(live_name)
    if manifest["files"][live_name]["sha256"] != live.metadata["sha256"]:
        raise ValueError(
            f"GW{gameweek}: historical roster manifest does not bind selected event-live"
        )
    historical = read("01-bootstrap-static.json")
    FPLApiClient._validate_top_level_schema(BOOTSTRAP_STATIC, historical)
    recorded_fixtures = read(f"04-gw{gameweek:02d}-fixtures.json")
    fields = (
        "id",
        "event",
        "team_h",
        "team_a",
        "kickoff_time",
        "finished",
        "finished_provisional",
        "team_h_score",
        "team_a_score",
    )
    expected = [r for r in fixtures.payload if r["event"] == gameweek]
    if not isinstance(recorded_fixtures, list) or sorted(
        tuple(r[k] for k in fields) for r in recorded_fixtures
    ) != sorted(tuple(r[k] for k in fields) for r in expected):
        raise ValueError(
            f"GW{gameweek}: historical roster collection contradicts official fixtures"
        )
    event = [r for r in historical["events"] if r["id"] == gameweek]
    current_event = [r for r in bootstrap.payload["events"] if r["id"] == gameweek]
    if (
        len(event) != 1
        or len(current_event) != 1
        or _timestamp(event[0]["deadline_time"]) != _timestamp(current_event[0]["deadline_time"])
        or _timestamp(event[0]["deadline_time"]) >= collected
        or any(_timestamp(r["kickoff_time"]) >= collected for r in expected)
    ):
        raise ValueError(
            f"GW{gameweek}: historical roster collection has incompatible event timing"
        )
    teams = historical["teams"]
    if len(teams) != len(bootstrap.payload["teams"]) or {r["id"]: r["code"] for r in teams} != {
        r["id"]: r["code"] for r in bootstrap.payload["teams"]
    }:
        raise ValueError(f"GW{gameweek}: historical roster team identities differ")
    rows = historical["elements"]
    ids = [r["id"] for r in rows]
    current = {r["id"]: r for r in bootstrap.payload["elements"]}
    if not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"GW{gameweek}: invalid historical roster player identities")
    for row in rows:
        if row["id"] not in current or any(
            type(row.get(k)) is not int or row[k] != current[row["id"]].get(k)
            for k in ("code", "team", "element_type")
        ):
            raise ValueError(
                f"GW{gameweek}: historical roster identity/team/position evidence "
                f"incompatible for player {row['id']}"
            )
    return [current[i] for i in ids], evidence


def _validate_live(
    *,
    season: str,
    gameweek: int,
    live: LoadedSnapshot,
    bootstrap: LoadedSnapshot,
    fixtures: LoadedSnapshot,
    cutoff: pd.Timestamp,
    evidence: dict[Path, bytes] | None = None,
    normalized_dir: Path | None = None,
    client: FPLApiClient | None = None,
) -> None:
    available = max(
        _timestamp(source.metadata["retrieved_at"])
        for source in (bootstrap, fixtures, live)
    )
    if available >= cutoff:
        raise ValueError(
            f"GW{gameweek}: official evidence was not retrieved before target deadline"
        )
    elements = live.payload.get("elements")
    if not isinstance(elements, list) or any(
        not isinstance(row, dict) for row in elements
    ):
        raise ValueError(f"GW{gameweek}: invalid elements list")
    ids = [row.get("id") for row in elements]
    if not ids or any(type(value) is not int for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"GW{gameweek}: incomplete or duplicate official player coverage")
    roster = bootstrap.payload["elements"]
    if "historical_roster_manifest" in live.metadata:
        roster, roster_evidence = _recorded_roster(
            season=season,
            gameweek=gameweek,
            live=live,
            bootstrap=bootstrap,
            fixtures=fixtures,
            cutoff=cutoff,
        )
        if evidence is not None:
            evidence.update(roster_evidence)
        if set(ids) != {row["id"] for row in roster}:
            raise ValueError(f"GW{gameweek}: incomplete or duplicate official player coverage")
    known_ids = {row["id"] for row in roster}
    # The official historical response defines its population. A later bootstrap
    # is an identity lookup, not a registration history or a source of missing DNPs.
    roster = [row for row in roster if row["id"] in set(ids)]
    event_fixtures = [
        row for row in fixtures.payload
        if row["event"] == gameweek
        and row["finished"] and row["finished_provisional"]
        and _timestamp(row["kickoff_time"]) < cutoff
    ]
    required_result_keys = {
        (fixture["id"], entry["element"])
        for fixture in event_fixtures
        for stat in fixture.get("stats", [])
        for side in ("h", "a")
        for entry in stat[side]
    }
    if {player for _, player in required_result_keys} - set(ids):
        raise ValueError(f"GW{gameweek}: incomplete official player coverage; fixture-stat players missing")
    for row in elements:
        stats = row.get("stats")
        explain = row.get("explain")
        if not isinstance(stats, dict) or not isinstance(explain, list):
            raise ValueError(f"GW{gameweek}: missing player stats/explain evidence")
        for field in (*VALUE_COMPONENTS, "starts", "total_points"):
            if type(stats.get(field)) is not int:
                raise ValueError(f"GW{gameweek}: invalid or missing player {field}")
        if stats["minutes"] < 0 or stats["starts"] < 0:
            raise ValueError(f"GW{gameweek}: negative player minutes or starts")
        if not explain:
            if any(
                stats[field] != 0
                for field in (*VALUE_COMPONENTS, "starts", "total_points")
            ):
                raise ValueError(
                    f"GW{gameweek}: blank event or absent explanations hide player outcomes"
                )
        for block in explain:
            if (
                not isinstance(block, dict)
                or type(block.get("fixture")) is not int
                or not isinstance(block.get("stats"), list)
            ):
                raise ValueError(f"GW{gameweek}: invalid fixture explanation")
            for component in block["stats"]:
                if (
                    not isinstance(component, dict)
                    or component.get("identifier")
                    not in (*VALUE_COMPONENTS, *POINT_COMPONENTS)
                    or any(
                        type(component.get(key)) is not int
                        for key in ("value", "points")
                    )
                    or type(component.get("points_modification", 0)) is not int
                ):
                    raise ValueError(
                        f"GW{gameweek}: incomplete or unsupported scoring component"
                    )
    if event_fixtures:
        latest_kickoff = max(_timestamp(row["kickoff_time"]) for row in event_fixtures)
        if any(
            _timestamp(source.metadata["retrieved_at"]) <= latest_kickoff
            for source in (bootstrap, fixtures, live)
        ):
            raise ValueError(
                f"GW{gameweek}: finalized evidence predates fixture kickoff"
            )
    expected_fixtures = {row["id"] for row in event_fixtures}
    resolved_kwargs = {}
    if normalized_dir is not None and expected_fixtures:
        resolver = HistoricalClubResolver(
            season=season,
            bootstrap_payload=bootstrap.payload,
            bootstrap_metadata=bootstrap.metadata,
            bootstrap_raw_path=bootstrap.raw_path,
            fixtures_payload=fixtures.payload,
            raw_fpl_dir=Path(bootstrap.raw_path).parents[2],
            normalized_dir=normalized_dir,
            client=client or FPLApiClient(raw_dir=Path(bootstrap.raw_path).parents[2]),
            information_cutoff=cutoff,
            allow_network=True,
            source_context={"source_mode": "official_current_season"},
        )
        # Reuse develop's fixture-specific identity/club evidence so offline
        # reconstruction can consume the same verified cached inputs.
        resolved = resolver.resolve_event(
            gameweek=gameweek,
            payload=live.payload,
            eligible_fixture_ids=expected_fixtures,
        )
        known_ids.update(resolved.identities)
        resolved_kwargs = {
            "resolved_fixture_blocks": resolved.fixture_blocks,
            "player_identity_evidence": resolved.identities,
            "historical_club_assignments": resolved.club_assignments,
        }
        for entry in resolver.snapshot_entries.values():
            path = Path(entry["raw_snapshot_path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Historical club evidence changed during provisioning.")
            available = max(available, _timestamp(entry["retrieved_at"]))
            if evidence is not None:
                evidence[path] = raw
                metadata_path = metadata_for_raw_path(path)
                evidence[metadata_path] = metadata_path.read_bytes()
    if set(ids) - known_ids:
        raise ValueError(f"GW{gameweek}: incomplete official player coverage; unknown official player IDs")
    frame = normalize_event_live(
        season=season,
        gameweek=gameweek,
        payload=live.payload,
        retrieved_at=live.metadata["retrieved_at"],
        raw_snapshot_path=live.raw_path,
        bootstrap_payload={**bootstrap.payload, "elements": roster},
        fixtures_payload=fixtures.payload,
        raw_fpl_dir=Path(bootstrap.raw_path).parents[2],
        eligible_fixture_ids=expected_fixtures,
        **resolved_kwargs,
    )
    for source in frame.attrs.get("historical_bootstrap_snapshots", []):
        path = Path(source["raw_path"])
        raw = path.read_bytes()
        metadata_path = metadata_for_raw_path(path)
        if (hashlib.sha256(raw).hexdigest() != source["metadata"]["sha256"]
                or json.loads(metadata_path.read_bytes()) != source["metadata"]):
            raise ValueError("Historical team evidence changed during normalization.")
        if evidence is not None:
            evidence[path] = raw
            evidence[metadata_path] = metadata_path.read_bytes()
    expected_fixtures = {row["id"] for row in event_fixtures}
    if (set(frame["fixture_id"]) != expected_fixtures
            or not required_result_keys.issubset(set(zip(frame["fixture_id"], frame["player_id"], strict=True)))):
        raise ValueError(f"GW{gameweek}: incomplete official fixture coverage")
    if frame.empty:
        return
    frame["source_available_time"] = available
    issues = validate_event_live_for_forecast(frame, information_cutoff=cutoff)
    audit = audit_event_live_scoring(frame)["player_event_reconciliation"]
    if issues or audit["audit_status"].ne("exact_match").any():
        raise ValueError(f"GW{gameweek}: player scoring validation failed: {issues}")
    # Fixture membership comes from explicit official explanations, never from
    # a player's current club (which may have changed after these results).
    expected_keys = {
        (block["fixture"], element["id"])
        for element in elements for block in element["explain"]
        if block["fixture"] in expected_fixtures
    }
    if resolved_kwargs:
        expected_keys = {
            (block["fixture"], player_id)
            for player_id, blocks in resolved_kwargs["resolved_fixture_blocks"].items()
            for block in blocks
        }
    if set(zip(frame["fixture_id"], frame["player_id"], strict=True)) != expected_keys:
        raise ValueError(f"GW{gameweek}: incomplete player-fixture coverage")


def snapshot_prior_event_live(
    *,
    season: str,
    target_gameweek: int,
    raw_fpl_dir: Path = RAW_FPL_API_DIR,
    normalized_dir: Path = NORMALIZED_DIR,
    client: FPLApiClient | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[SnapshotRecord]:
    """Reuse validated latest evidence, retrieving only missing prior event snapshots.

    Partial retrievals remain append-only on failure. This helper never writes
    normalized data, forecast status, successful-run pointers or model artifacts.
    """
    if type(target_gameweek) is not int or target_gameweek not in range(1, 39):
        raise ValueError("Target gameweek must be an integer from 1 to 38")
    raw_fpl_dir, normalized_dir = Path(raw_fpl_dir), Path(normalized_dir)
    client = client or FPLApiClient(raw_dir=raw_fpl_dir)
    if Path(client.raw_dir).resolve() != raw_fpl_dir.resolve():
        raise ValueError("Client raw directory does not match provisioning directory")
    evidence: dict[Path, bytes] = {}
    selected: dict[str, Path] = {}

    def load(endpoint: str, url: str) -> LoadedSnapshot:
        source, contents = _load_verified(
            raw_dir=raw_fpl_dir,
            season=season,
            endpoint=endpoint,
            url=url,
            now=_timestamp(clock()),
        )
        selected[endpoint] = Path(source.raw_path)
        evidence.update(contents)
        return source

    bootstrap = load(BOOTSTRAP_STATIC, client.bootstrap_static_url())
    fixtures = load(FIXTURES, client.fixtures_url())
    infer_and_validate_current_season(
        requested_season=season,
        bootstrap_payload=bootstrap.payload,
        fixtures_payload=fixtures.payload,
    )
    _validate_schedule(bootstrap, fixtures, target_gameweek=target_gameweek)
    frames, normalized_evidence = _normalized_context(
        season=season,
        normalized_dir=normalized_dir,
        bootstrap=bootstrap,
        fixtures=fixtures,
    )
    evidence.update(normalized_evidence)
    target = resolve_target_gameweek(
        season=season,
        events=frames["events"],
        fixtures=frames["fixtures"],
        requested_gameweek=target_gameweek,
        now=_timestamp(clock()).to_pydatetime(),
    )
    cutoff = _timestamp(target.deadline_time)
    if any(
        _timestamp(source.metadata["retrieved_at"]) >= cutoff
        for source in (bootstrap, fixtures)
    ):
        raise ValueError("Official context was not retrieved before target deadline")
    records = []
    for gameweek in range(1, target_gameweek):
        endpoint = f"event_live_{gameweek}"
        event_fixtures = [row for row in fixtures.payload if row["event"] == gameweek]
        if event_fixtures and not any(
            row["finished"] and row["finished_provisional"]
            and _timestamp(row["kickoff_time"]) < cutoff for row in event_fixtures
        ):
            continue
        try:
            # Never refresh corrupt/late latest evidence or silently fall back.
            record = client.fetch_event_live(
                season=season,
                gameweek=gameweek,
                refresh=False,
                offline=False,
            )
            live = load(endpoint, client.event_live_url(gameweek))
            if (
                Path(live.raw_path) != record.raw_path
                or live.metadata["sha256"] != record.checksum_sha256
                or live.metadata["retrieved_at"] != record.retrieved_at
                or live.metadata["source_url"] != record.source_url
                or live.metadata["content_length"] != record.content_length
            ):
                raise ValueError("Selected event-live changed during retrieval")
            _validate_live(
                season=season,
                gameweek=gameweek,
                live=live,
                bootstrap=bootstrap,
                fixtures=fixtures,
                cutoff=cutoff,
                evidence=evidence,
                normalized_dir=normalized_dir,
                client=client,
            )
            records.append(record)
        except Exception as exc:
            raise ValueError(
                f"GW{gameweek}: prior event-live evidence unavailable: {exc}"
            ) from exc
    for endpoint, path in selected.items():
        if (
            latest_snapshot_path(
                raw_fpl_dir, season=season, endpoint_name=endpoint, content_type="json"
            )
            != path
        ):
            raise ValueError(f"Selected official snapshot changed: {endpoint}")
    for path, contents in evidence.items():
        if path.read_bytes() != contents:
            raise ValueError(f"Evidence changed during provisioning: {path.name}")
    if _timestamp(clock()) >= cutoff:
        raise ValueError("Target deadline reached during evidence provisioning")
    return records
