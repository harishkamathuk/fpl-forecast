"""Pure official target validation shared by ingestion and publication."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd


class PublicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetGameweekResolution:
    season: str
    gameweek: int
    deadline_time: str
    resolved_at: str
    method: str
    prior_events_verified: int


def resolve_target_gameweek(
    *,
    season: str,
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    requested_gameweek: int | None,
    now: datetime | None = None,
) -> TargetGameweekResolution:
    required_event = {
        "gameweek",
        "deadline_time",
        "finished",
        "data_checked",
        "is_current",
        "is_next",
    }
    required_fixture = {"gameweek", "finished", "finished_provisional"}
    if missing := required_event.difference(events.columns):
        raise PublicationError(f"Current events are missing fields: {', '.join(sorted(missing))}.")
    if missing := required_fixture.difference(fixtures.columns):
        raise PublicationError(f"Current fixtures are missing fields: {', '.join(sorted(missing))}.")

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    frame = events.copy()
    frame["gameweek"] = pd.to_numeric(frame["gameweek"], errors="coerce")
    frame["deadline"] = pd.to_datetime(frame["deadline_time"], utc=True, errors="coerce")
    if frame["gameweek"].isna().any() or frame["deadline"].isna().any():
        raise PublicationError("Official event metadata contains invalid gameweeks or deadlines.")
    if frame["gameweek"].duplicated().any():
        raise PublicationError("Official event metadata contains duplicate gameweeks.")

    if requested_gameweek is None:
        candidates = frame.loc[
            ~frame["finished"].fillna(False).astype(bool)
            & frame["deadline"].gt(current_time)
        ].sort_values(["deadline", "gameweek"])
        if candidates.empty:
            raise PublicationError("No uncompleted official gameweek with a future deadline is available.")
        target = candidates.iloc[0]
        method = "earliest_unfinished_future_deadline"
    else:
        target_rows = frame.loc[frame["gameweek"].eq(requested_gameweek)]
        if len(target_rows) != 1:
            raise PublicationError(f"Requested gameweek {requested_gameweek} is missing or ambiguous.")
        target = target_rows.iloc[0]
        method = "validated_workflow_input"

    gameweek = int(target["gameweek"])
    if bool(target["finished"]):
        raise PublicationError(f"Gameweek {gameweek} is already completed.")
    if target["deadline"].to_pydatetime() <= current_time:
        raise PublicationError(
            f"Gameweek {gameweek} deadline has passed; pre-deadline publication is no longer valid."
        )
    target_fixtures = fixtures.loc[pd.to_numeric(fixtures["gameweek"], errors="coerce").eq(gameweek)]
    if target_fixtures.empty:
        raise PublicationError(
            f"Gameweek {gameweek} is a globally blank official event; the current model chain "
            "does not support publishing a zero-fixture event."
        )
    target_started = target_fixtures.get("started", pd.Series(False, index=target_fixtures.index))
    target_finished = target_fixtures["finished"].fillna(False).astype(bool)
    if target_started.fillna(False).astype(bool).any() or target_finished.any():
        raise PublicationError(f"Gameweek {gameweek} contains a started or finished target fixture.")

    prior = frame.loc[frame["gameweek"].lt(gameweek)]

    return TargetGameweekResolution(
        season=season,
        gameweek=gameweek,
        deadline_time=target["deadline"].isoformat(),
        resolved_at=current_time.isoformat(),
        method=method,
        prior_events_verified=len(prior),
    )
