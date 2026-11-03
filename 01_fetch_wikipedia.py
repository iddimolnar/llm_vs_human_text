"""
01 - Ember által írt szövegek gyűjtése a magyar Wikipédiából.

A MediaWiki API-t használja: egyszerre 10 véletlen cikket kér,
majd azok bevezetőjét tölti le. Így sokkal kevesebb API-hívás szükséges.

Használat:
    python 01_fetch_wikipedia.py --n 300

Kimenet:
    data/human.jsonl  (mezők: id, title, text, n_words, source, label)
"""

import argparse
import json
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "human.jsonl"

API_URL = "https://hu.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "NLP-felevi-feladat/1.0 (egyetemi projekt)"}

MIN_CHARS = 200
MAX_CHARS = 2000
BATCH = 10  # egyszerre ennyi véletlen cikket kérünk


def get_random_titles(session, n=BATCH):
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": n,
        "format": "json",
    }
    resp = session.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return [p["title"] for p in resp.json()["query"]["random"]]


def get_extracts(session, titles):
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": "|".join(titles),
        "format": "json",
    }
    resp = session.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    result = {}
    for page in pages.values():
        title = page.get("title", "")
        text = page.get("extract", "").strip()
        if text and len(text) >= MIN_CHARS:
            result[title] = text[:MAX_CHARS]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="kívánt minták száma")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    seen_titles = set()
    records = []
    session = requests.Session()

    while len(records) < args.n:
        try:
            titles = get_random_titles(session, BATCH)
            time.sleep(1.0)
            extracts = get_extracts(session, titles)
            time.sleep(1.0)
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            wait = 30 if code == 429 else 5
            print(f"  HTTP {code} hiba, várakozás {wait}s...")
            time.sleep(wait)
            continue
        except requests.RequestException as e:
            print(f"  hálózati hiba: {e}, várakozás 5s...")
            time.sleep(5)
            continue

        for title, text in extracts.items():
            if title in seen_titles:
                continue
            seen_titles.add(title)
            records.append({
                "id": len(records),
                "title": title,
                "text": text,
                "n_words": len(text.split()),
                "source": "wikipedia",
                "label": "human",
            })
            if len(records) % 25 == 0:
                print(f"  {len(records)}/{args.n} összegyűjtve")
            if len(records) >= args.n:
                break

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Kész: {len(records)} ember-írta minta -> {OUT_FILE}")


if __name__ == "__main__":
    main()
