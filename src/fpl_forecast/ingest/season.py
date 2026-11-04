from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class SeasonIdentityError(ValueError):
    pass


@dataclass(frozen=True)
class BootstrapSeasonIdentity:
    inferred_season: str
    first_event_deadline: str
    last_event_deadline: str
    event_count: int


@dataclass(frozen=True)
class SeasonIdentity:
    requested_season: str
    inferred_season: str
    first_event_deadline: str
    last_event_deadline: str
    first_fixture_kickoff: str
    last_fixture_kickoff: str
    event_count: int
    team_count: int
    fixture_count: int

    def as_metadata(self) -> dict[str, Any]:
        return {
            "requested_season": self.requested_season,
            "inferred_season": self.inferred_season,
            "season_identity": {
                "requested_season": self.requested_season,
                "inferred_season": self.inferred_season,
                "first_event_deadline": self.first_event_deadline,
                "last_event_deadline": self.last_event_deadline,
                "first_fixture_kickoff": self.first_fixture_kickoff,
                "last_fixture_kickoff": self.last_fixture_kickoff,
                "event_count": self.event_count,
                "team_count": self.team_count,
                "fixture_count": self.fixture_count,
                "method": (
                    "Inferred from FPL event deadline_time values and fixture kickoff_time "
                    "values; structure checked for a standard 20-team Premier League season."
                ),
            },
        }


def parse_season_label(season: str) -> tuple[int, int]:
    parts = season.split("-")
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
        raise SeasonIdentityError(
            f"Season must use YYYY-YY format such as 2025-26; received {season!r}."
        )
    try:
        start_year = int(parts[0])
        short_end_year = int(parts[1])
    except ValueError as exc:
        raise SeasonIdentityError(
            f"Season must use YYYY-YY format such as 2025-26; received {season!r}."
        ) from exc
    expected_short_end = (start_year + 1) % 100
    if short_end_year != expected_short_end:
        raise SeasonIdentityError(
            f"Season {season!r} has inconsistent end year; expected {start_year}-{expected_short_end:02d}."
        )
    return start_year, start_year + 1


def infer_and_validate_current_season(
    *,
    requested_season: str,
    bootstrap_payload: dict[str, Any],
    fixtures_payload: list[dict[str, Any]],
) -> SeasonIdentity:
    requested_start, requested_end = parse_season_label(requested_season)
    events = bootstrap_payload.get("events")
    teams = bootstrap_payload.get("teams")
    if not isinstance(teams, list):
        raise SeasonIdentityError("Cannot infer season: bootstrap teams are missing.")
    if not isinstance(fixtures_payload, list) or not fixtures_payload:
        raise SeasonIdentityError("Cannot infer season: fixtures are missing or empty.")

    bootstrap_identity = infer_bootstrap_season(bootstrap_payload)
    # Unplayed fixtures can legitimately be unassigned/unscheduled. Infer
    # season from the known dates without deleting those fixtures or inventing
    # a kickoff. Result-bearing rows still require their actual kickoff.
    known_kickoffs = []
    for fixture in fixtures_payload:
        if not isinstance(fixture, dict) or "kickoff_time" not in fixture:
            raise SeasonIdentityError("Cannot infer season: missing fixtures.kickoff_time field.")
        kickoff = fixture["kickoff_time"]
        if kickoff is None:
            if any(fixture.get(flag) for flag in ("started", "finished", "finished_provisional")):
                raise SeasonIdentityError("Result-bearing fixture has no kickoff_time.")
            if any(fixture.get(score) is not None for score in ("team_h_score", "team_a_score")):
                raise SeasonIdentityError("Scored fixture has no kickoff_time.")
        else:
            known_kickoffs.append(kickoff)
    kickoff_times = _parse_required_datetimes(known_kickoffs, field_name="fixtures.kickoff_time")

    first_deadline = _parse_datetime(bootstrap_identity.first_event_deadline)
    last_deadline = _parse_datetime(bootstrap_identity.last_event_deadline)
    first_kickoff = min(kickoff_times)
    last_kickoff = max(kickoff_times)
    inferred_start = first_kickoff.year if first_kickoff.month >= 6 else first_kickoff.year - 1
    inferred_end = inferred_start + 1
    inferred_season = f"{inferred_start}-{inferred_end % 100:02d}"

    errors: list[str] = []
    if inferred_season != requested_season:
        errors.append(
            f"requested season {requested_season} conflicts with inferred payload season "
            f"{inferred_season}"
        )
    if not (
        first_deadline.year in {requested_start, requested_end}
        and last_deadline.year in {requested_start, requested_end}
        and first_kickoff.year in {requested_start, requested_end}
        and last_kickoff.year in {requested_start, requested_end}
    ):
        errors.append(
            "event deadlines and fixture kickoffs do not all fall within the requested "
            f"{requested_start}-{requested_end} season window"
        )
    if len(teams) != 20:
        errors.append(f"expected 20 teams for a standard Premier League season, found {len(teams)}")
    if len(events) != 38:
        errors.append(f"expected 38 events for a standard Premier League season, found {len(events)}")
    if len(fixtures_payload) != 380:
        errors.append(
            f"expected 380 fixtures for a standard Premier League season, found {len(fixtures_payload)}"
        )
    if errors:
        raise SeasonIdentityError("; ".join(errors))

    return SeasonIdentity(
        requested_season=requested_season,
        inferred_season=inferred_season,
        first_event_deadline=_format_datetime(first_deadline),
        last_event_deadline=_format_datetime(last_deadline),
        first_fixture_kickoff=_format_datetime(first_kickoff),
        last_fixture_kickoff=_format_datetime(last_kickoff),
        event_count=len(events),
        team_count=len(teams),
        fixture_count=len(fixtures_payload),
    )


def infer_bootstrap_season(bootstrap_payload: dict[str, Any]) -> BootstrapSeasonIdentity:
    """Infer the season represented by one bootstrap-static payload.

    FPL element IDs are allocated within a season and can be reused for a
    different player in a later season. Archive consumers therefore need a
    payload-level season check even when a snapshot was stored beneath a
    caller-supplied season directory.
    """

    events = bootstrap_payload.get("events")
    if not isinstance(events, list) or not events:
        raise SeasonIdentityError("Cannot infer season: bootstrap events are missing or empty.")
    deadline_times = _parse_required_datetimes(
        [event.get("deadline_time") for event in events if isinstance(event, dict)],
        field_name="events.deadline_time",
    )
    first_deadline = min(deadline_times)
    last_deadline = max(deadline_times)
    inferred_start = first_deadline.year if first_deadline.month >= 6 else first_deadline.year - 1
    inferred_end = inferred_start + 1
    inferred_season = f"{inferred_start}-{inferred_end % 100:02d}"
    if any(value.year not in {inferred_start, inferred_end} for value in deadline_times):
        raise SeasonIdentityError(
            "Cannot infer season: bootstrap event deadlines span more than one season window."
        )
    return BootstrapSeasonIdentity(
        inferred_season=inferred_season,
        first_event_deadline=_format_datetime(first_deadline),
        last_event_deadline=_format_datetime(last_deadline),
        event_count=len(events),
    )


def _parse_required_datetimes(values: list[Any], *, field_name: str) -> list[datetime]:
    parsed: list[datetime] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise SeasonIdentityError(f"Cannot infer season: missing {field_name}.")
        try:
            parsed.append(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC))
        except ValueError as exc:
            raise SeasonIdentityError(
                f"Cannot infer season: malformed {field_name} value {value!r}."
            ) from exc
    if not parsed:
        raise SeasonIdentityError(f"Cannot infer season: no parseable {field_name} values.")
    return parsed


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
