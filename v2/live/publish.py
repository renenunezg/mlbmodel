"""Poll current MLB games and publish snapshots across the Supabase boundary.

Defaults to one read-only pass. --watch polls until interrupted; --publish opts
into database writes and still requires the backend database write guard.
Example: python -m v2.live.publish --artifact models/live_win_expectancy.json --watch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from v2.live.feed import fetch_feed, score_feed
from v2.live.model import WinExpectancy

log = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 60


def scheduled_games() -> list[dict]:
    today = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    response = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
        "sportId": 1, "startDate": (today - timedelta(days=1)).isoformat(),
        "endDate": today.isoformat(),
    }, timeout=15)
    response.raise_for_status()
    games = {game["gamePk"]: game for day in response.json().get("dates", []) for game in day["games"]
             if game.get("gameType") in {"R", "F", "D", "L", "W"}
             and game.get("status", {}).get("abstractGameState") in {"Live", "Final"}}
    return list(games.values())


def snapshot(game_pk: int, model: WinExpectancy, artifact_sha256: str) -> dict:
    return {**score_feed(fetch_feed(game_pk), model), "schema_version": 1,
            "artifact_sha256": artifact_sha256, "probability_source": "league_average_game_state"}


def publish_snapshots(
    snapshots: list[dict], published: dict[int, tuple[str, float, str]], *, now: float | None = None,
) -> set[int]:
    """Write changed states or a live heartbeat; acknowledge only committed batches."""
    now = time.monotonic() if now is None else now
    rows = []
    acknowledgements = {}
    for item in snapshots:
        source = item.get("source_timestamp")
        if not source:
            continue
        pk = item["game_pk"]
        # Source/fetch clocks advance without a pitch or a probability change.
        content = {k: v for k, v in item.items()
                   if k not in {"fetched_at", "source_timestamp", "source_age_seconds", "stale"}}
        signature = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
        previous = published.get(pk)
        # score_feed normalizes source timestamps to UTC ISO strings.
        if previous and source < previous[2]:
            continue
        if previous and signature == previous[0] and (
            item.get("abstract_state") == "Final" or now - previous[1] < HEARTBEAT_SECONDS
        ):
            published[pk] = (previous[0], previous[1], source)
            continue
        rows.append(dict(game_pk=pk, updated_at=item["fetched_at"], source_timestamp=source, payload=item))
        acknowledgements[pk] = (signature, now, source)
    if not rows:
        return set()

    # Import lazily so the default read-only run needs no database credentials.
    from sqlalchemy import text

    from backend.db import engine

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO live_win_probability (game_pk, updated_at, source_timestamp, payload)
            SELECT game_pk, updated_at, source_timestamp, payload
            FROM jsonb_to_recordset(CAST(:snapshots AS jsonb))
                AS incoming(game_pk integer, updated_at timestamptz,
                            source_timestamp timestamptz, payload jsonb)
            ON CONFLICT (game_pk) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                source_timestamp = EXCLUDED.source_timestamp,
                payload = EXCLUDED.payload
            WHERE live_win_probability.source_timestamp <= EXCLUDED.source_timestamp
                AND live_win_probability.updated_at < EXCLUDED.updated_at
        """), {"snapshots": json.dumps(rows, allow_nan=False)})
    published.update(acknowledgements)
    return set(acknowledgements)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--duration", type=int, help="Stop a watch worker cleanly after this many seconds")
    args = ap.parse_args()
    if args.interval < 15:
        ap.error("--interval must be at least 15 seconds")
    if args.duration is not None and (args.duration <= 0 or not args.watch):
        ap.error("--duration requires --watch and a positive number of seconds")
    if args.publish:
        from backend.db import writes_allowed

        if not writes_allowed():
            ap.error("Publication requires MLBMODEL_DB_WRITES=1 or GitHub Actions")
    model = WinExpectancy.load(args.artifact)
    digest = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    completed = {}
    published = {}
    games = []
    next_schedule = 0.
    deadline = time.monotonic() + args.duration if args.duration else float("inf")
    with ThreadPoolExecutor(max_workers=4) as workers:
        while True:
            started = time.monotonic()
            try:
                if started >= next_schedule:
                    games = scheduled_games()
                    next_schedule = started + 60
                fingerprints = {g["gamePk"]: json.dumps([g["status"], g["teams"]], sort_keys=True)
                                for g in games}
                pending = [g for g in games if g["status"]["abstractGameState"] == "Live"
                           or completed.get(g["gamePk"]) != fingerprints[g["gamePk"]]]
                jobs = {workers.submit(snapshot, g["gamePk"], model, digest): g["gamePk"] for g in pending}
                snapshots = []
                for job in as_completed(jobs):
                    try:
                        snapshots.append(job.result())
                    except (requests.RequestException, ValueError, KeyError):
                        log.exception("Could not refresh game %s", jobs[job])
                written = publish_snapshots(snapshots, published) if args.publish else set()
                for item in snapshots:
                    if item.get("abstract_state") == "Final" and item.get("home_win_probability") is not None:
                        completed[item["game_pk"]] = fingerprints[item["game_pk"]]
                    log.info("%s %s @ %s: %s, home WP %s%s", item["game_pk"], item["away_team"],
                             item["home_team"], item["status"], item["home_win_probability"],
                             " (published)" if item["game_pk"] in written else
                             " (not written)" if args.publish else " (read only)")
                # Bound state across long-running workers and season changes.
                completed = {pk: fp for pk, fp in completed.items() if pk in fingerprints}
                published = {pk: state for pk, state in published.items() if pk in fingerprints}
            except Exception:
                if not args.watch:
                    raise
                log.exception("Refresh failed; previous snapshots will age visibly on the site")
            if not args.watch or time.monotonic() >= deadline:
                return
            time.sleep(max(0, min(deadline - time.monotonic(),
                                 max(1, args.interval - (time.monotonic() - started)))))
            if time.monotonic() >= deadline:
                return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
