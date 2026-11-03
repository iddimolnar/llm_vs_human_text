import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
MODEL_DIR = BASE / "model"

MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
MAX_LENGTH = 256
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    splits = {name: load_split(name).map(tokenize, batched=True)
              for name in ["train", "val", "test"]}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_out = trainer.predict(splits["test"])
    preds = np.argmax(test_out.predictions, axis=-1)
    labels = test_out.label_ids

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    metrics = {
        "model": MODEL_NAME,
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    (RESULTS_DIR / "transformer_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_confusion_matrix(labels, preds, RESULTS_DIR / "transformer_confusion_matrix.png")

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nKész. Eredmények -> {RESULTS_DIR}, modell -> {MODEL_DIR}")


def save_confusion_matrix(labels, preds, out_path):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["ember", "LLM"])
    ax.set_yticks([0, 1], ["ember", "LLM"])
    ax.set_xlabel("Jósolt")
    ax.set_ylabel("Valós")
    ax.set_title("Transformer – Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
