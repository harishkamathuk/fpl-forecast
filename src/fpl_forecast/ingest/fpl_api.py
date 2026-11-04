from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any

import httpx

from fpl_forecast.config import (
    FPL_API_BASE_URL,
    HttpSettings,
    RAW_FPL_API_DIR,
)
from fpl_forecast.ingest.snapshots import (
    SnapshotRecord,
    latest_snapshot_path,
    read_json_snapshot,
    read_metadata,
    write_raw_snapshot,
)
from fpl_forecast.ingest.season import SeasonIdentityError, infer_and_validate_current_season


BOOTSTRAP_STATIC = "bootstrap_static"
FIXTURES = "fixtures"
ELEMENT_SUMMARY = "element_summary"
EVENT_LIVE = "event_live"

ENDPOINT_PATHS = {
    BOOTSTRAP_STATIC: "bootstrap-static/",
    FIXTURES: "fixtures/",
}

REQUIRED_TOP_LEVEL_FIELDS: dict[str, set[str]] = {
    BOOTSTRAP_STATIC: {
        "events",
        "game_settings",
        "phases",
        "teams",
        "total_players",
        "elements",
        "element_stats",
        "element_types",
    },
    ELEMENT_SUMMARY: {"fixtures", "history", "history_past"},
    EVENT_LIVE: {"elements"},
}


class FPLApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedSnapshot:
    endpoint_name: str
    payload: Any
    raw_path: str
    metadata: dict[str, Any]


class FPLApiClient:
    def __init__(
        self,
        *,
        raw_dir=RAW_FPL_API_DIR,
        settings: HttpSettings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.settings = settings or HttpSettings()
        self.transport = transport

    def bootstrap_static_url(self) -> str:
        return f"{FPL_API_BASE_URL}{ENDPOINT_PATHS[BOOTSTRAP_STATIC]}"

    def fixtures_url(self) -> str:
        return f"{FPL_API_BASE_URL}{ENDPOINT_PATHS[FIXTURES]}"

    def element_summary_url(self, player_id: int) -> str:
        return f"{FPL_API_BASE_URL}element-summary/{player_id}/"

    def event_live_url(self, gameweek: int) -> str:
        return f"{FPL_API_BASE_URL}event/{gameweek}/live/"

    def fetch_endpoint(
        self,
        *,
        season: str,
        endpoint_name: str,
        url: str,
        refresh: bool = False,
        offline: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> SnapshotRecord:
        cached_path = latest_snapshot_path(
            self.raw_dir,
            season=season,
            endpoint_name=endpoint_name,
            content_type="json",
        )
        if cached_path is not None and not refresh:
            metadata = read_metadata(cached_path)
            return SnapshotRecord(
                endpoint_name=endpoint_name,
                season=season,
                raw_path=cached_path,
                metadata_path=cached_path.with_suffix(cached_path.suffix + ".metadata.json"),
                retrieved_at=metadata["retrieved_at"],
                source_url=metadata["source_url"],
                source_version=metadata.get("source_version"),
                checksum_sha256=metadata["sha256"],
                content_length=metadata["content_length"],
            )

        if offline:
            raise FPLApiError(
                f"No cached {endpoint_name!r} snapshot exists for season {season}; "
                "offline mode cannot retrieve a new one."
            )

        content, status_code, headers = self._get_json_bytes(url)
        payload = self._parse_json(content, endpoint_name=endpoint_name, url=url)
        self._validate_top_level_schema(endpoint_name, payload)
        return write_raw_snapshot(
            self.raw_dir,
            season=season,
            endpoint_name=endpoint_name,
            content=content,
            source_url=url,
            http_status=status_code,
            response_headers=dict(headers),
            target_season=season,
            source="fpl_api",
            source_version=None,
            content_type="json",
            extra_metadata=extra_metadata,
        )

    def snapshot_current(
        self,
        *,
        season: str,
        refresh: bool = False,
        offline: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[SnapshotRecord]:
        cached_records = self._cached_current_records(season=season)
        if len(cached_records) == 2 and not refresh:
            bootstrap = read_json_snapshot(cached_records[0].raw_path)
            fixtures = read_json_snapshot(cached_records[1].raw_path)
            try:
                infer_and_validate_current_season(
                    requested_season=season,
                    bootstrap_payload=bootstrap,
                    fixtures_payload=fixtures,
                )
            except SeasonIdentityError as exc:
                raise FPLApiError(f"Cached FPL snapshots fail season identity checks: {exc}") from exc
            return cached_records
        if offline:
            raise FPLApiError(
                f"Cached current FPL snapshots are incomplete for season {season}; "
                "offline mode cannot retrieve new snapshots."
            )

        bootstrap_content, bootstrap_status, bootstrap_headers = self._get_json_bytes(
            self.bootstrap_static_url()
        )
        fixtures_content, fixtures_status, fixtures_headers = self._get_json_bytes(self.fixtures_url())
        bootstrap_payload = self._parse_json(
            bootstrap_content,
            endpoint_name=BOOTSTRAP_STATIC,
            url=self.bootstrap_static_url(),
        )
        fixtures_payload = self._parse_json(
            fixtures_content,
            endpoint_name=FIXTURES,
            url=self.fixtures_url(),
        )
        self._validate_top_level_schema(BOOTSTRAP_STATIC, bootstrap_payload)
        self._validate_top_level_schema(FIXTURES, fixtures_payload)
        try:
            season_identity = infer_and_validate_current_season(
                requested_season=season,
                bootstrap_payload=bootstrap_payload,
                fixtures_payload=fixtures_payload,
            )
        except SeasonIdentityError as exc:
            raise FPLApiError(f"FPL current snapshots failed season identity checks: {exc}") from exc

        season_metadata = {**season_identity.as_metadata(), **(extra_metadata or {})}
        return [
            write_raw_snapshot(
                self.raw_dir,
                season=season,
                endpoint_name=BOOTSTRAP_STATIC,
                content=bootstrap_content,
                source_url=self.bootstrap_static_url(),
                http_status=bootstrap_status,
                response_headers=dict(bootstrap_headers),
                target_season=season,
                source="fpl_api",
                source_version=season_identity.inferred_season,
                content_type="json",
                extra_metadata=season_metadata,
            ),
            write_raw_snapshot(
                self.raw_dir,
                season=season,
                endpoint_name=FIXTURES,
                content=fixtures_content,
                source_url=self.fixtures_url(),
                http_status=fixtures_status,
                response_headers=dict(fixtures_headers),
                target_season=season,
                source="fpl_api",
                source_version=season_identity.inferred_season,
                content_type="json",
                extra_metadata=season_metadata,
            ),
        ]

    def fetch_element_summary(
        self,
        *,
        season: str,
        player_id: int,
        refresh: bool = False,
        offline: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> SnapshotRecord:
        return self.fetch_endpoint(
            season=season,
            endpoint_name=f"{ELEMENT_SUMMARY}_{player_id}",
            url=self.element_summary_url(player_id),
            refresh=refresh,
            offline=offline,
            extra_metadata=extra_metadata,
        )

    def fetch_event_live(
        self,
        *,
        season: str,
        gameweek: int,
        refresh: bool = False,
        offline: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> SnapshotRecord:
        return self.fetch_endpoint(
            season=season,
            endpoint_name=f"{EVENT_LIVE}_{gameweek}",
            url=self.event_live_url(gameweek),
            refresh=refresh,
            offline=offline,
            extra_metadata=extra_metadata,
        )

    def _get_json_bytes(self, url: str) -> tuple[bytes, int, httpx.Headers]:
        headers = {"User-Agent": self.settings.user_agent, "Accept": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                with httpx.Client(
                    timeout=self.settings.timeout_seconds,
                    headers=headers,
                    transport=self.transport,
                    follow_redirects=True,
                ) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.content, response.status_code, response.headers
            except httpx.HTTPStatusError as exc:
                message = (
                    f"FPL API returned HTTP {exc.response.status_code} for {url}: "
                    f"{exc.response.text[:300]}"
                )
                raise FPLApiError(message) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= self.settings.retries:
                    break
                time.sleep(self.settings.backoff_seconds * (2**attempt))
        raise FPLApiError(f"FPL API request failed for {url}: {last_error}") from last_error

    def _cached_current_records(self, *, season: str) -> list[SnapshotRecord]:
        records: list[SnapshotRecord] = []
        for endpoint_name in (BOOTSTRAP_STATIC, FIXTURES):
            cached_path = latest_snapshot_path(
                self.raw_dir,
                season=season,
                endpoint_name=endpoint_name,
                content_type="json",
            )
            if cached_path is None:
                continue
            metadata = read_metadata(cached_path)
            records.append(
                SnapshotRecord(
                    endpoint_name=endpoint_name,
                    season=season,
                    raw_path=cached_path,
                    metadata_path=cached_path.with_suffix(cached_path.suffix + ".metadata.json"),
                    retrieved_at=metadata["retrieved_at"],
                    source_url=metadata["source_url"],
                    source_version=metadata.get("source_version"),
                    checksum_sha256=metadata["sha256"],
                    content_length=metadata["content_length"],
                )
            )
        return records

    @staticmethod
    def _parse_json(content: bytes, *, endpoint_name: str, url: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise FPLApiError(f"Invalid JSON from FPL endpoint {endpoint_name!r} at {url}") from exc

    @staticmethod
    def _validate_top_level_schema(endpoint_name: str, payload: Any) -> None:
        if endpoint_name == FIXTURES:
            if not isinstance(payload, list):
                raise FPLApiError("FPL fixtures response must be a JSON list.")
            return

        schema_name = endpoint_name
        if endpoint_name.startswith(f"{ELEMENT_SUMMARY}_"):
            schema_name = ELEMENT_SUMMARY
        if endpoint_name.startswith(f"{EVENT_LIVE}_"):
            schema_name = EVENT_LIVE

        required_fields = REQUIRED_TOP_LEVEL_FIELDS.get(schema_name)
        if required_fields is None:
            return
        if not isinstance(payload, dict):
            raise FPLApiError(f"FPL {endpoint_name} response must be a JSON object.")
        missing = sorted(required_fields.difference(payload))
        if missing:
            raise FPLApiError(
                f"FPL {endpoint_name} response is missing required fields: {', '.join(missing)}"
            )


def load_latest_fpl_snapshot(*, raw_dir=RAW_FPL_API_DIR, season: str, endpoint_name: str) -> LoadedSnapshot:
    raw_path = latest_snapshot_path(
        raw_dir,
        season=season,
        endpoint_name=endpoint_name,
        content_type="json",
    )
    if raw_path is None:
        raise FPLApiError(f"No cached FPL {endpoint_name!r} snapshot exists for season {season}.")
    metadata = read_metadata(raw_path)
    return LoadedSnapshot(
        endpoint_name=endpoint_name,
        payload=read_json_snapshot(raw_path),
        raw_path=str(raw_path),
        metadata=metadata,
    )


def load_bootstrap_before_fixture(*, raw_dir: Path, season: str, kickoff: str) -> LoadedSnapshot:
    """Read the latest recorded bootstrap before kickoff; never fetch or backdate."""
    cutoff = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
    candidates = []
    for path in (Path(raw_dir) / season / BOOTSTRAP_STATIC).glob("*.json"):
        if path.name.endswith(".metadata.json"):
            continue
        metadata = read_metadata(path)
        retrieved = datetime.fromisoformat(metadata["retrieved_at"].replace("Z", "+00:00"))
        if retrieved.tzinfo is None or cutoff.tzinfo is None:
            raise FPLApiError("Historical team evidence requires aware timestamps.")
        retrieved = retrieved.astimezone(UTC)
        if retrieved < cutoff:
            candidates.append((retrieved, path, metadata))
    if not candidates:
        raise FPLApiError("No authoritative historical team bootstrap before fixture kickoff.")
    retrieved, path, metadata = max(candidates, key=lambda row: (row[0], row[1].name))
    content = path.read_bytes()
    expected = {
        "source": "fpl_api", "season": season, "target_season": season,
        "endpoint_name": BOOTSTRAP_STATIC,
        "source_url": f"{FPL_API_BASE_URL}bootstrap-static/",
        "sha256": hashlib.sha256(content).hexdigest(), "content_length": len(content),
    }
    if (any(metadata.get(k) != v for k, v in expected.items())
            or metadata.get("http_status") != 200
            or Path(metadata["raw_path"]).resolve() != path.resolve()
            or path.name != retrieved.strftime("%Y%m%dT%H%M%S%fZ") + ".json"):
        raise FPLApiError("Invalid authoritative historical team bootstrap provenance.")
    payload = json.loads(content)
    FPLApiClient._validate_top_level_schema(BOOTSTRAP_STATIC, payload)
    return LoadedSnapshot(BOOTSTRAP_STATIC, payload, str(path.resolve()), metadata)
