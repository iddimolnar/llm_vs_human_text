"""
02 - LLM-generált szövegek előállítása a Groq API-val (ingyenes keret).

Minden ember-írta Wikipédia-bevezetőhöz (human.jsonl) legenerál egy
Wikipédia-stílusú, azonos témájú és hasonló hosszúságú magyar szöveget.
Az azonos 'id' köti össze az ember-írta és az LLM-generált párt, így a
témát és a hosszt kontrolláljuk -> az osztályozónak a STÍLUST kell
megtanulnia, nem a témát.

Előfeltétel:
    pip install groq
    Ingyenes API-kulcs: https://console.groq.com
    export GROQ_API_KEY="..."

Használat:
    python 02_generate_llm.py

Kimenet:
    data/llm.jsonl  (mezők: id, title, text, n_words, source, label)
"""

import json
import os
import sys
import time
from pathlib import Path

from groq import Groq

DATA_DIR = Path(__file__).parent / "data"
IN_FILE = DATA_DIR / "human.jsonl"
OUT_FILE = DATA_DIR / "llm.jsonl"

MODEL = "qwen/qwen3.8-27b"
MAX_RETRIES = 5
SLEEP = 2.5  # 30 RPM limit -> 2s elég, 2.5s biztonságos

PROMPT_TEMPLATE = (
    "Írj egy magyar nyelvű, Wikipédia-stílusú bevezető szócikket a következő "
    "témáról: \"{title}\". A szöveg hossza legyen körülbelül {n_words} szó. "
    "Csak a szócikk folyó szövegét add vissza, cím, felsorolás és formázás "
    "nélkül."
)


def load_human():
    if not IN_FILE.exists():
        sys.exit(f"Hiányzik: {IN_FILE}. Futtasd előbb a 01_fetch_wikipedia.py-t.")
    with IN_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def already_done():
    if not OUT_FILE.exists():
        return set()
    done = set()
    with OUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["id"])
    return done


def generate(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                wait = 30 * (attempt + 1)
                print(f"  rate-limit, várakozás {wait}s... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    return None


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Állítsd be a GROQ_API_KEY környezeti változót "
                 "(ingyenes kulcs: https://console.groq.com).")

    client = Groq(api_key=api_key)
    human = load_human()
    done = already_done()
    remaining = [r for r in human if r["id"] not in done]
    print(f"  {len(done)} már kész, {len(remaining)} generálandó")

    with OUT_FILE.open("a", encoding="utf-8") as out:
        for i, rec in enumerate(remaining):
            prompt = PROMPT_TEMPLATE.format(title=rec["title"], n_words=rec["n_words"])
            text = generate(client, prompt)
            if not text:
                print(f"  id={rec['id']} kihagyva")
                continue

            out.write(json.dumps({
                "id": rec["id"],
                "title": rec["title"],
                "text": text,
                "n_words": len(text.split()),
                "source": MODEL,
                "label": "llm",
            }, ensure_ascii=False) + "\n")
            out.flush()

            if (i + 1) % 25 == 0:
                print(f"  {len(done) + i + 1}/{len(human)} generálva")

            time.sleep(SLEEP)

    print(f"Kész -> {OUT_FILE}")


if __name__ == "__main__":
    main()
