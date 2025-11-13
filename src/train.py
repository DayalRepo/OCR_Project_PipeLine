# src/train.py

import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer
)
import evaluate

# ----------------------------------------------------
# LABEL SET (COMPLETE)
# ----------------------------------------------------
LABEL_LIST = [
    "O",

    "B-NAME","I-NAME",

    "B-ADDRESS","I-ADDRESS",

    "B-CITY","I-CITY",

    "B-STATE","I-STATE",

    "B-ZIP","I-ZIP",

    "B-TRACKING","I-TRACKING",

    "B-COMPANY","I-COMPANY"
]

num_labels = len(LABEL_LIST)


# ----------------------------------------------------
# ALIGN LABELS
# ----------------------------------------------------
def align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=256
    )

    all_labels = []

    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []

        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            else:
                tag = LABEL_LIST[labels[w]]

                # convert B- to I- for same word continuation
                if prev == w and tag.startswith("B-"):
                    tag = "I-" + tag[2:]

                label_ids.append(LABEL_LIST.index(tag))

            prev = w

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)

    metric = evaluate.load("seqeval")

    true_preds = []
    true_labels = []

    for pred_row, label_row in zip(preds, labels):
        p_clean = []
        l_clean = []
        for p_i, l_i in zip(pred_row, label_row):
            if l_i != -100:
                p_clean.append(LABEL_LIST[p_i])
                l_clean.append(LABEL_LIST[l_i])
        true_preds.append(p_clean)
        true_labels.append(l_clean)

    return metric.compute(predictions=true_preds, references=true_labels)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    print("Loading dataset...")
    ds = load_from_disk("data/hf_dataset")

    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base",
        add_prefix_space=True
    )

    print("Tokenizing training set...")
    train_ds = ds["train"].map(
        lambda ex: align_labels(ex, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names
    )

    print("Tokenizing validation set...")
    val_ds = ds["test"].map(
        lambda ex: align_labels(ex, tokenizer),
        batched=True,
        remove_columns=ds["test"].column_names
    )

    print("Loading model...")
    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    # ----------------------------------------------------
    # UNIVERSAL TrainingArguments (WORKS ON ALL VERSIONS)
    # ----------------------------------------------------
    args = TrainingArguments(
        output_dir="models/roberta_ner",
        do_train=True,
        do_eval=True,

        num_train_epochs=3,     # ← 3 EPOCHS HERE
        learning_rate=3e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        weight_decay=0.01,
        logging_steps=50,
        save_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("models/roberta_ner")
    tokenizer.save_pretrained("models/roberta_ner")

    print("TRAINING COMPLETE ✔")


if __name__ == "__main__":
    main()
