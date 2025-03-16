import os
import time
import copy
import json
import pickle
from multiprocessing import Pool

import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    AdamW,
    get_scheduler,
)

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from scripts.attr_parser import (
    parse_attr_to_str,
    parse_attr_keys_to_str,
    parse_attr_selected_to_str,
)

from modeling.trainer import (
    FocalTrainer,
    WeightedCETrainer,
    HierarchicalFocalTrainer,
)
from modeling.callback import TimeLimitCallback
from modeling.metrics import compute_metrics

tqdm.pandas()

# --- CONFIG ---

WORKDIR = "/kaggle/input/data-fusion-2025/"
HEAD_COEFF = 1000
N_PROC = 4
TEST_SIZE = 0.1
SEED = 42
ATTR_MODE = "selected"
DIFF_DIST = True
TRAINER = "hierarchical"
VER = "2epoch-diffhierarchy-tokens-groupby1000-selected-name-and-keys-focal-ruropebert-1024"

os.environ["WANDB_API_KEY"] = "..."
os.environ["WANDB_PROJECT"] = "data fusion ruropebert"
os.environ["WANDB_NOTES"] = f"data fusion ruropebert VER-{VER}"
os.environ["WANDB_NAME"] = f"data-fusion-ruropebert-{VER}"


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "Tochka-AI/ruRoPEBert-e5-base-2k"
    hdim: int = 312
    num_labels: int = 776
    max_length: int = 1024
    optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 8
    n_epochs: int = 2
    lr: float = 5e-5
    warmup_ratio: float = 0.05
    seed: int = SEED


def load_data(workdir: str):
    labeled_train = pd.read_parquet(os.path.join(workdir, "labeled_train.parquet"))
    unlabeled_train = pd.read_parquet(os.path.join(workdir, "unlabeled_train.parquet"))
    category_tree = pd.read_csv(os.path.join(workdir, "category_tree.csv"))
    result_df = pd.read_csv(os.path.join(workdir, "keys_and_values.csv"))
    return labeled_train, unlabeled_train, category_tree, result_df


def parallel_apply(series: pd.Series, func, n_proc: int = N_PROC):
    with Pool(n_proc) as pool:
        result = pool.map(func, series)
    return result


def prepare_train_data(train_df: pd.DataFrame, attr_mode: str) -> pd.DataFrame:
    train_df = train_df.groupby("cat_id").head(HEAD_COEFF)

    attr_parsers = {
        "selected": parse_attr_selected_to_str,
        "keys": parse_attr_keys_to_str,
        "all": parse_attr_to_str,
    }

    if attr_mode not in attr_parsers:
        raise ValueError(f"Unknown ATTR_MODE: {attr_mode}")

    train_df["attrs"] = parallel_apply(train_df["attributes"], attr_parsers[attr_mode])

    train_df["source_name"] = train_df["source_name"].fillna("нет названия")
    train_df["attrs"] = train_df["attrs"].fillna("нет атрибутов")
    train_df["text"] = (
        "<name_start>" +
        train_df["source_name"] +
        "<name_end><attrs_start>" +
        train_df["attrs"] +
        "<attrs_end>"
    )

    train_df = train_df[["text", "cat_id"]].rename(columns={"cat_id": "label"})
    return train_df


def create_label_mapping(labels: pd.Series):
    unique_labels = labels.unique()
    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_labels))}
    reverse_mapping = {v: k for k, v in mapping.items()}
    return mapping, reverse_mapping


def main():
    config = Config()

    labeled_train, _, category_tree, result_df = load_data(WORKDIR)

    keys_with_good_values = result_df[result_df["is_value_needed"] == True]["key"].tolist()
    only_good_keys = result_df[
        (result_df["is_key_needed"] == True) & pd.isna(result_df["is_value_needed"])
    ]["key"].tolist()

    train_labeled = prepare_train_data(labeled_train, ATTR_MODE)

    mapping, reverse_mapping = create_label_mapping(train_labeled["label"])
    
    with open("mapping.pickle", "wb") as handle:
        pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    train_labeled["label"] = train_labeled["label"].map(mapping)

    training_args = TrainingArguments(
        output_dir=f"output-{VER}",
        overwrite_output_dir=True,
        report_to="wandb",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=21232,
        optim=config.optim_type,
        learning_rate=config.lr,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=config.num_labels,
        device_map="cuda",
    )

    special_tokens = [
        "<some_value>",
        "<name_start>", "<name_end>",
        "<attrs_start>", "<attrs_end>",
    ]
    num_added_toks = tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    counts = train_labeled["label"].value_counts().to_dict()
    total_samples = sum(counts.values())
    num_classes = len(counts)
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in counts.items()}

    train_df, test_df = train_test_split(
        train_labeled,
        test_size=TEST_SIZE,
        # stratify=train_labeled["cat_id"],
        random_state=SEED,
    )

    ds_train = Dataset.from_pandas(train_df)
    ds_test = Dataset.from_pandas(test_df)

    def encode(batch):
        return tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=config.max_length
        )

    ds_train = ds_train.map(encode, batched=True)
    ds_test = ds_test.map(encode, batched=True)

    trainer_mapping = {
        "hierarchical": HierarchicalFocalTrainer,
        "focal": FocalTrainer,
        "ce": WeightedCETrainer,
    }

    if TRAINER not in trainer_mapping:
        raise ValueError(f"Unknown TRAINER type: {TRAINER}")

    trainer_class = trainer_mapping[TRAINER]
    common_params = {
        "args": training_args,
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": ds_train,
        "eval_dataset": ds_test,
        "compute_metrics": compute_metrics,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "class_weights": list(class_weights.values()),
        "callbacks": [TimeLimitCallback(time_limit_hours=11.15)],
    }

    if TRAINER == "hierarchical":
        trainer = trainer_class(
            **common_params,
            diff_dist=DIFF_DIST,
            category_tree_df=category_tree,
            mapping=reverse_mapping,
        )
    else:
        trainer = trainer_class(**common_params)

    trainer.train()


if __name__ == "__main__":
    main()
