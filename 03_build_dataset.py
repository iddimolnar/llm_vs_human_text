"""
03 - Adathalmaz összeállítása és felosztása.

Összefűzi az ember-írta (human.jsonl) és az LLM-generált (llm.jsonl)
szövegeket, megkeveri, majd rétegzett (stratified) módon train/val/test
részekre osztja, hogy mindkét osztály aránya megmaradjon.

Használat:
    python 03_build_dataset.py

Kimenet:
    data/train.csv, data/val.csv, data/test.csv  (oszlopok: text, label)
    label: 0 = ember, 1 = LLM
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"


def load_jsonl(name):
    path = DATA_DIR / name
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    rows = load_jsonl("human.jsonl") + load_jsonl("llm.jsonl")
    df = pd.DataFrame(rows)[["text", "label"]]
    df["label"] = df["label"].map({"human": 0, "llm": 1})
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)

    # 70 / 15 / 15 felosztás, rétegzve a címkére.
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=42
    )

    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = DATA_DIR / f"{name}.csv"
        part.to_csv(out, index=False)
        human_n = int((part["label"] == 0).sum())
        llm_n = int((part["label"] == 1).sum())
        print(f"{name:5s}: {len(part):4d} minta  (ember={human_n}, llm={llm_n})")

    print(f"\nÖsszes: {len(df)} minta -> {DATA_DIR}")


if __name__ == "__main__":
    main()
