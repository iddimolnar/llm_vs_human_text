"""
04 - Baseline modell: TF-IDF + logisztikus regresszió.

Egyszerű, átlátható viszonyítási pont a Transformer-modellhez.
A tanító halmazon tanul, a teszt halmazon értékel, és kiírja az
accuracy / precision / recall / F1 metrikákat, valamint elmenti a
confusion matrixot ábraként.

Használat:
    python 04_baseline.py

Előfeltétel:
    data/train.csv és data/test.csv (lásd 03_build_dataset.py)

Kimenet:
    results/baseline_metrics.json
    results/baseline_confusion_matrix.png
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(train["text"], train["label"])

    pred = model.predict(test["text"])
    y_true = test["label"]

    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", pos_label=1
    )

    metrics = {
        "model": "TF-IDF + LogisticRegression",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    (RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_confusion_matrix(y_true, pred, RESULTS_DIR / "baseline_confusion_matrix.png")
    print(f"\nEredmények elmentve -> {RESULTS_DIR}")


def save_confusion_matrix(y_true, pred, out_path):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["ember", "LLM"])
    ax.set_yticks([0, 1], ["ember", "LLM"])
    ax.set_xlabel("Jósolt")
    ax.set_ylabel("Valós")
    ax.set_title("Baseline – Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
