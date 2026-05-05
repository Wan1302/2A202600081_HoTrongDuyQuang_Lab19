from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "wikipedia_ai_company_corpus.jsonl"
DEFAULT_MANIFEST = DATA_DIR / "wikipedia_ai_company_manifest.json"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "LabDay19GraphRAG/1.0 (educational Wikipedia corpus builder)"


# More than 100 seeds are listed because Wikipedia page names can change or some
# pages can be redirects/disambiguation pages. The scraper stops after it gets
# the requested number of valid article extracts.
AI_COMPANY_SEEDS = [
    "OpenAI",
    "Anthropic",
    "DeepMind",
    "Google DeepMind",
    "Google",
    "Microsoft",
    "Meta Platforms",
    "Nvidia",
    "Apple Inc.",
    "Amazon (company)",
    "IBM",
    "Oracle Corporation",
    "Salesforce",
    "Adobe Inc.",
    "Intel",
    "Advanced Micro Devices",
    "Qualcomm",
    "Arm Holdings",
    "Taiwan Semiconductor Manufacturing Company",
    "Tesla, Inc.",
    "xAI",
    "Hugging Face",
    "Stability AI",
    "Midjourney",
    "Mistral AI",
    "Cohere",
    "Scale AI",
    "DataRobot",
    "Palantir Technologies",
    "Databricks",
    "Snowflake Inc.",
    "ServiceNow",
    "UiPath",
    "Automation Anywhere",
    "C3.ai",
    "SambaNova Systems",
    "Cerebras Systems",
    "Graphcore",
    "Groq",
    "Tenstorrent",
    "Waymo",
    "Cruise LLC",
    "Mobileye",
    "Aurora Innovation",
    "Zoox",
    "Nuro",
    "Baidu",
    "Alibaba Group",
    "Tencent",
    "Huawei",
    "ByteDance",
    "SenseTime",
    "Megvii",
    "iFlytek",
    "Yitu Technology",
    "CloudWalk Technology",
    "Naver Corporation",
    "Kakao",
    "Samsung Electronics",
    "LG Electronics",
    "Sony",
    "NEC",
    "Fujitsu",
    "Hitachi",
    "SoftBank Group",
    "Rakuten",
    "SAP",
    "Siemens",
    "Bosch (company)",
    "Philips",
    "Thales Group",
    "Atos",
    "Capgemini",
    "Accenture",
    "Deloitte",
    "Booz Allen Hamilton",
    "Lockheed Martin",
    "RTX Corporation",
    "Northrop Grumman",
    "Anduril Industries",
    "Shield AI",
    "Grammarly",
    "Duolingo",
    "Canva",
    "Snap Inc.",
    "Pinterest",
    "Spotify",
    "Netflix",
    "Uber",
    "Inflection AI",
    "Character.ai",
    "Perplexity AI",
    "Runway (company)",
    "Replit",
    "Adept AI Labs",
    "Veritone",
    "SoundHound AI",
    "Appen (company)",
    "BenevolentAI",
    "Insilico Medicine",
    "Recursion Pharmaceuticals",
    "Tempus AI",
    "PathAI",
    "Atomwise",
    "Exscientia",
    "Owkin",
    "Darktrace",
    "Sift (company)",
    "FICO",
    "SAS Institute",
    "Alteryx",
    "RapidMiner",
    "KNIME",
    "Wolfram Research",
    "OpenText",
    "Clarifai",
    "Element AI",
    "Vicarious (company)",
    "Affectiva",
    "iRobot",
    "Boston Dynamics",
    "Agility Robotics",
    "Fetch Robotics",
    "Covariant (company)",
    "Open Robotics",
    "Symbotic",
    "Zebra Technologies",
    "LivePerson",
    "Yext",
    "Cerence",
    "Nuance Communications",
    "Kensho Technologies",
    "Two Sigma",
    "Jane Street Capital",
    "Bloomberg L.P.",
]


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return value.strip("_")


def clean_text(value: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= max_chars:
        return value
    cut = value[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "."


def api_get(session: requests.Session, params: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(6):
        try:
            response = session.get(WIKIPEDIA_API, params=params, timeout=45)
        except requests.RequestException:
            if attempt == 5:
                raise
            delay = min(30, 2 ** attempt)
            print(f"request failed; retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * (attempt + 1))
            print(f"Wikipedia rate limit; retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError("Wikipedia API kept returning rate-limit responses.")


def search_title(session: requests.Session, query: str) -> str | None:
    data = api_get(
        session,
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
        },
    )
    rows = data.get("query", {}).get("search", [])
    if not rows:
        return None
    return rows[0].get("title")


def fetch_article(
    session: requests.Session,
    title: str,
    max_chars: int,
) -> dict[str, str] | None:
    data = api_get(
        session,
        {
            "action": "query",
            "format": "json",
            "redirects": 1,
            "prop": "extracts|info|pageprops",
            "inprop": "url",
            "explaintext": 1,
            "exlimit": "max",
            "exintro": 1,
            "exsectionformat": "plain",
            "titles": title,
        },
    )
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page or "disambiguation" in page.get("pageprops", {}):
        return None
    if page["title"].startswith("List of "):
        return None

    extract = clean_text(page.get("extract", ""), max_chars)
    if len(extract) < 250:
        return None
    return {
        "title": page["title"],
        "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{page['title']}"),
        "text": extract,
    }


def fetch_articles_batch(
    session: requests.Session,
    titles: list[str],
    max_chars: int,
) -> list[dict[str, str]]:
    data = api_get(
        session,
        {
            "action": "query",
            "format": "json",
            "redirects": 1,
            "prop": "extracts|info|pageprops",
            "inprop": "url",
            "explaintext": 1,
            "exlimit": "max",
            "exintro": 1,
            "exsectionformat": "plain",
            "titles": "|".join(titles),
        },
    )
    rows: list[dict[str, str]] = []
    for page in data.get("query", {}).get("pages", {}).values():
        if "missing" in page or "disambiguation" in page.get("pageprops", {}):
            continue
        extract = clean_text(page.get("extract", ""), max_chars)
        if len(extract) < 250:
            continue
        rows.append(
            {
                "title": page["title"],
                "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{page['title']}"),
                "text": extract,
            }
        )
    return rows


def build_corpus(limit: int, max_chars: int, sleep_seconds: float) -> list[dict[str, str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    docs: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for seed in AI_COMPANY_SEEDS:
        if len(docs) >= limit:
            break
        try:
            article = fetch_article(session, seed, max_chars)
        except requests.RequestException as exc:
            print(f"skip {seed}: {exc}", flush=True)
            if sleep_seconds:
                time.sleep(sleep_seconds)
            continue
        if article is None:
            print(f"skip {seed}: no article extract", flush=True)
            if sleep_seconds:
                time.sleep(sleep_seconds)
            continue
        normalized_title = article["title"].casefold()
        if normalized_title in seen_titles:
            if sleep_seconds:
                time.sleep(sleep_seconds)
            continue
        seen_titles.add(normalized_title)
        docs.append(
            {
                "id": f"wiki_{len(docs) + 1:03d}_{slug(article['title'])}",
                "title": article["title"],
                "url": article["url"],
                "text": article["text"],
            }
        )
        print(f"{len(docs):03d}/{limit} {article['title']}", flush=True)
        if sleep_seconds:
            time.sleep(sleep_seconds)

    if len(docs) < limit:
        raise RuntimeError(
            f"Only collected {len(docs)} valid Wikipedia articles; requested {limit}."
        )
    return docs


def write_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(rows: list[dict[str, str]], path: Path, max_chars: int) -> None:
    payload = {
        "source": "Wikipedia MediaWiki API",
        "api_url": WIKIPEDIA_API,
        "article_count": len(rows),
        "max_chars_per_article": max_chars,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "articles": [{"id": row["id"], "title": row["title"], "url": row["url"]} for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 100-article AI company corpus from Wikipedia.")
    parser.add_argument("--limit", type=int, default=100, help="Number of Wikipedia articles to collect.")
    parser.add_argument("--max-chars", type=int, default=2500, help="Max characters kept per article.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between article requests.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output manifest path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_corpus(args.limit, args.max_chars, args.sleep)
    write_jsonl(rows, args.output)
    write_manifest(rows, args.manifest, args.max_chars)
    print(f"Wrote {len(rows)} articles to {args.output}", flush=True)
    print(f"Wrote manifest to {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
