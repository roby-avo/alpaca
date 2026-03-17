#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wikidata_stats import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    fetch_wikidata_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch current Wikidata item count from the rendered Wikidata:Statistics page "
            "via the MediaWiki parse API. Intended as a manual baseline/sanity check."
        )
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Custom HTTP User-Agent (Wikimedia may reject generic defaults).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be > 0")

        snapshot = fetch_wikidata_stats(
            timeout_seconds=float(args.timeout_seconds),
            user_agent=str(args.user_agent),
        )

        if args.json:
            print(
                json.dumps(
                    {
                        "items": snapshot.items,
                        "edits": snapshot.edits,
                        "fetched_at_utc": snapshot.fetched_at_utc,
                        "source_page_url": snapshot.source_page_url,
                        "source_api_url": snapshot.source_api_url,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        print("Wikidata live statistics snapshot (manual baseline helper)")
        print(f"  items={snapshot.items:,}")
        if snapshot.edits is not None:
            print(f"  edits={snapshot.edits:,}")
        print(f"  fetched_at_utc={snapshot.fetched_at_utc}")
        print(f"  source_page={snapshot.source_page_url}")
        print(f"  source_api={snapshot.source_api_url}")
        print()
        print("# Example manual baseline note / constant update:")
        print(
            f"WIKIDATA_ITEMS_BASELINE = {snapshot.items}  # fetched {snapshot.fetched_at_utc} from Wikidata:Statistics"
        )
        return 0
    except (ValueError, json.JSONDecodeError, HTTPError, URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
