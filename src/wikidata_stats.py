from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TextIO
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


def resolve_expected_entity_total(
    *,
    manual_total: int | None,
    fetch_live: bool,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    log_stream: TextIO | None = None,
) -> int | None:
    if manual_total is not None and manual_total <= 0:
        raise ValueError("--expected-entity-total must be > 0 when provided")
    if manual_total is not None and fetch_live:
        raise ValueError("Use either --expected-entity-total or --fetch-live-entity-total, not both.")
    if not fetch_live:
        return manual_total
    if timeout_seconds <= 0:
        raise ValueError("--live-entity-total-timeout-seconds must be > 0")

    try:
        snapshot = fetch_wikidata_stats(timeout_seconds=timeout_seconds, user_agent=user_agent)
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not fetch live Wikidata item total: {exc}") from exc
    if log_stream is not None:
        print(
            f"Fetched live Wikidata item total: {snapshot.items:,} "
            f"(source: {snapshot.source_page_url}, fetched_at_utc={snapshot.fetched_at_utc})",
            file=log_stream,
        )
    return snapshot.items
