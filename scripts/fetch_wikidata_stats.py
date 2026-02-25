#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_USER_AGENT = "alpaca-wikidata-stats/1.0 (+manual baseline helper)"
WIKIDATA_STATS_PAGE_URL = "https://www.wikidata.org/wiki/Wikidata%3AStatistics"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ITEMS_RE = re.compile(
    r"\bWikidata\s+currently\s+contains\s+([0-9][0-9,\.]*)\s+items\b",
    flags=re.IGNORECASE,
)
_EDITS_RE = re.compile(
    r"\b([0-9][0-9,\.]*)\s+edits\s+have\s+been\s+made\s+since\s+the\s+project\s+launch\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class WikidataStatsSnapshot:
    items: int
    edits: int | None
    fetched_at_utc: str
    source_page_url: str
    source_api_url: str


def _fetch_url(url: str, *, timeout_seconds: float, user_agent: str) -> bytes:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout_seconds) as response:
        return response.read()


def _strip_html(raw_html: str) -> str:
    # We only need a small phrase from the rendered page; a simple tag strip is enough.
    text = _HTML_TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_int_group(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    raw = match.group(1)
    digits = raw.replace(",", "").replace(".", "")
    if not digits.isdigit():
        return None
    return int(digits)


def fetch_wikidata_stats(*, timeout_seconds: float, user_agent: str) -> WikidataStatsSnapshot:
    api_query = urlencode(
        {
            "action": "parse",
            "page": "Wikidata:Statistics",
            "prop": "text",
            "format": "json",
        }
    )
    api_url = f"{WIKIDATA_API_URL}?{api_query}"
    payload = _fetch_url(api_url, timeout_seconds=timeout_seconds, user_agent=user_agent)
    parsed = json.loads(payload)

    raw_html = (
        parsed.get("parse", {}).get("text", {}).get("*", "")
        if isinstance(parsed, dict)
        else ""
    )
    if not isinstance(raw_html, str) or not raw_html.strip():
        raise ValueError("Wikidata parse API response did not contain rendered page HTML.")

    text = _strip_html(raw_html)
    items = _parse_int_group(_ITEMS_RE, text)
    if items is None:
        raise ValueError(
            "Could not extract item count from Wikidata:Statistics rendered text. "
            "Page wording may have changed."
        )
    edits = _parse_int_group(_EDITS_RE, text)

    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return WikidataStatsSnapshot(
        items=items,
        edits=edits,
        fetched_at_utc=fetched_at,
        source_page_url=WIKIDATA_STATS_PAGE_URL,
        source_api_url=api_url,
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
