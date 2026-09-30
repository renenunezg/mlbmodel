"""No-write live scorer: python -m v2.live.score --artifact PATH --game-pk ID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.live.feed import fetch_feed, score_feed
from v2.live.model import WinExpectancy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--game-pk", type=int)
    source.add_argument("--feed", type=Path, help="Replay a saved feed snapshot")
    args = ap.parse_args()
    feed = json.loads(args.feed.read_text()) if args.feed else fetch_feed(args.game_pk)
    print(json.dumps(score_feed(feed, WinExpectancy.load(args.artifact)), indent=2))


if __name__ == "__main__":
    main()
