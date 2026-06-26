# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.45.0",
#   "datasets",
#   "evaluate",
#   "seqeval",
#   "scikit-learn",
#   "accelerate"
# ]
# ///

import os
import re
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    set_seed,
)

from transformers.modeling_outputs import TokenClassifierOutput
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

TARGET_TEST_FILE = "data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv"
LABEL_COLUMN = "NE-COARSE-LIT"


# -------------------------
# TSV reading
# -------------------------


def parse_year(date_str: Optional[str]) -> int:
    if not date_str:
        return 1800
    m = re.match(r"(\d{4})", date_str)
    if not m:
        return 1800
    return int(m.group(1))


def normalize_year(year: int) -> float:
    """
    Rough normalization for historical documents.
    Example:
    1700 -> -1
    1800 -> 0
    1900 -> 1
    2000 -> 2
    """
    return (year - 1800) / 100.0


def read_hipe_tsv(path: Path, label_column: str = LABEL_COLUMN) -> List[Dict]:
    """
    Reads one HIPE-style TSV file.

    Returns sentence-level examples:
    {
        "tokens": [...],
        "labels": [...],
        "year": 1798,
        "source_file": ...
    }
    """

    examples = []

    current_tokens = []
    current_labels = []
    current_date = None
    current_year = 1800

    header = None
    label_idx = None
    misc_idx = None

    def flush_sentence():
        nonlocal current_tokens, current_labels, current_year
        if current_tokens:
            examples.append(
                {
                    "tokens": current_tokens,
                    "labels": current_labels,
                    "year": current_year,
                    "source_file": str(path),
                }
            )
        current_tokens = []
        current_labels = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                flush_sentence()
                continue

            if line.startswith("#"):
                if line.startswith("# hipe2022:date ="):
                    current_date = line.split("=", 1)[1].strip()
                    current_year = parse_year(current_date)
                continue

            cols = line.split("\t")

            if header is None:
                header = cols
                if label_column not in header:
                    raise ValueError(f"{label_column} not found in header of {path}")
                label_idx = header.index(label_column)
                misc_idx = header.index("MISC") if "MISC" in header else None
                continue

            if len(cols) <= label_idx:
                continue

            token = cols[0]
            label = cols[label_idx]
            misc = (
                cols[misc_idx] if misc_idx is not None and len(cols) > misc_idx else "_"
            )

            current_tokens.append(token)
            current_labels.append(label)

            if "EndOfSentence" in misc:
                flush_sentence()

    flush_sentence()
    return examples


def load_all_data(data_dir: str, target_test_file: str = TARGET_TEST_FILE):
    data_dir = Path(data_dir)

    all_tsvs = sorted(data_dir.rglob("*.tsv"))

    test_paths = [p for p in all_tsvs if p.name == target_test_file]
    if not test_paths:
        raise FileNotFoundError(f"Could not find target test file: {target_test_file}")

    test_path = test_paths[0]

    train_paths = [p for p in all_tsvs if p != test_path]

    print(f"Target test file: {test_path}")
    print(f"Number of training TSV files: {len(train_paths)}")

    train_examples = []
    for p in train_paths:
        train_examples.extend(read_hipe_tsv(p))

    test_examples = read_hipe_tsv(test_path)

    print(f"Train sentences: {len(train_examples)}")
    print(f"Test sentences: {len(test_examples)}")

    return train_examples, test_examples


# -------------------------
# Label handling
# -------------------------


def build_label_maps(train_examples, test_examples):
    labels = set()
    for ex in train_examples + test_examples:
        labels.update(ex["labels"])

    labels = sorted(labels)

    # Put O first if present
    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    return labels, label2id, id2label


class HipeTokenDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        label2id,
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        tokens = ex["tokens"]
        labels = ex["labels"]
        year = ex["year"]
        year_value = normalize_year(year)

        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )

        word_ids = encoded.word_ids()

        aligned_labels = []
        time_values = []

        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
                time_values.append(0.0)
            else:
                # Label first subtoken, ignore remaining subtokens
                if word_idx != previous_word_idx:
                    aligned_labels.append(self.label2id[labels[word_idx]])
                else:
                    aligned_labels.append(-100)

                time_values.append(year_value)

            previous_word_idx = word_idx

        encoded["labels"] = aligned_labels
        encoded["time_values"] = time_values

        return encoded


# -------------------------
# Model
# -------------------------


class HistoricalBertNER(PreTrainedModel):
    """
    Three possible variants:

    baseline:
        historical BERT -> classifier

    stacked:
        historical BERT -> 2 TransformerEncoder layers -> classifier

    time:
        historical BERT + date/time embedding -> 2 TransformerEncoder layers -> classifier
    """

    config_class = AutoConfig

    def __init__(
        self,
        config,
        model_name: str,
        num_labels: int,
        variant: str = "baseline",
        extra_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(config)

        self.num_labels = num_labels
        self.variant = variant
        self.extra_layers = extra_layers if variant in ["stacked", "time"] else 0
        self.use_time = variant == "time"

        self.bert = AutoModel.from_pretrained(model_name, config=config)

        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)

        if self.use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
            )

        if self.extra_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.extra_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.extra_layers,
            )

        self.classifier = nn.Linear(hidden_size, num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        time_values=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        if self.use_time:
            if time_values is None:
                raise ValueError("time_values must be provided for variant='time'")

            time_values = time_values.to(sequence_output.device).float()
            time_values = time_values.unsqueeze(-1)

            time_emb = self.time_mlp(time_values)
            sequence_output = sequence_output + time_emb

        if self.extra_layers > 0:
            key_padding_mask = attention_mask == 0
            sequence_output = self.extra_encoder(
                sequence_output,
                src_key_padding_mask=key_padding_mask,
            )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


# -------------------------
# Metrics
# -------------------------


def make_compute_metrics(label_list):
    def compute_metrics(pred):
        logits, labels = pred
        predictions = logits.argmax(axis=-1)

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            sent_preds = []
            sent_labels = []

            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                sent_preds.append(label_list[p])
                sent_labels.append(label_list[l])

            true_predictions.append(sent_preds)
            true_labels.append(sent_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


# -------------------------
# Training
# -------------------------


def train_one_variant(args, variant: str):
    print("=" * 80)
    print(f"Training variant: {variant}")
    print("=" * 80)

    train_examples, test_examples = load_all_data(
        args.data_dir,
        target_test_file=args.test_file,
    )

    label_list, label2id, id2label = build_label_maps(train_examples, test_examples)

    print("Labels:")
    for label in label_list:
        print(label)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = HipeTokenDataset(
        train_examples,
        tokenizer,
        label2id,
        max_length=args.max_length,
    )

    test_dataset = HipeTokenDataset(
        test_examples,
        tokenizer,
        label2id,
        max_length=args.max_length,
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    model = HistoricalBertNER(
        config=config,
        model_name=args.model_name,
        num_labels=len(label_list),
        variant=variant,
        extra_layers=2,
        dropout=args.dropout,
    )

    output_dir = os.path.join(args.output_dir, variant)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=args.fp16,
        save_total_limit=2,
        seed=args.seed,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(label_list),
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final test metrics for {variant}:")
    print(metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_file", type=str, default=TARGET_TEST_FILE)

    parser.add_argument(
        "--model_name",
        type=str,
        default="dbmdz/bert-base-historic-multilingual-cased",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["baseline", "stacked", "time", "all"],
    )

    parser.add_argument("--output_dir", type=str, default="outputs/ner_hipe")
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.variant == "all":
        variants = ["baseline", "stacked", "time"]
    else:
        variants = [args.variant]

    all_metrics = {}

    for variant in variants:
        metrics = train_one_variant(args, variant)
        all_metrics[variant] = metrics

    print("=" * 80)
    print("ALL RESULTS")
    print("=" * 80)
    for variant, metrics in all_metrics.items():
        print(variant, metrics)


if __name__ == "__main__":
    main()
