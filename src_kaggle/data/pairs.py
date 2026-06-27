"""Pair dataset construction for Kaggle War Mode."""

from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from src_kaggle.data.contracts import TableKind, validate_dataframe
from src_kaggle.data.io import merge_items, merge_terms
from src_kaggle.data.schema import SCHEMA


def normalize_positive_pairs(training_pairs: pd.DataFrame) -> pd.DataFrame:
    training_pairs = validate_dataframe(training_pairs, TableKind.TRAINING_PAIRS, positive_only=True)
    return training_pairs[[SCHEMA.id, SCHEMA.term_id, SCHEMA.item_id, SCHEMA.label]].copy()


def sample_negative_pairs(
    positives: pd.DataFrame,
    all_item_ids: list,
    negatives_per_positive: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample local validation negatives from items not positive for the same term.

    Official training_pairs.csv contains only label=1. These sampled negatives
    are for local binary training/validation only; official submission pairs are
    never sampled here.
    """
    rng = random.Random(seed)
    pos_by_term: dict[object, set] = defaultdict(set)
    for row in positives[[SCHEMA.term_id, SCHEMA.item_id]].itertuples(index=False):
        pos_by_term[row[0]].add(row[1])

    records = []
    item_ids = list(dict.fromkeys(all_item_ids))
    synthetic_id = -1
    for term_id, _item_id in positives[[SCHEMA.term_id, SCHEMA.item_id]].itertuples(index=False):
        used = set(pos_by_term[term_id])
        tries = 0
        made = 0
        max_tries = max(100, 20 * negatives_per_positive)
        while made < negatives_per_positive and tries < max_tries:
            tries += 1
            candidate = rng.choice(item_ids)
            if candidate in used:
                continue
            records.append({
                SCHEMA.id: synthetic_id,
                SCHEMA.term_id: term_id,
                SCHEMA.item_id: candidate,
                SCHEMA.label: 0,
            })
            synthetic_id -= 1
            used.add(candidate)
            made += 1
    return pd.DataFrame.from_records(records, columns=[SCHEMA.id, SCHEMA.term_id, SCHEMA.item_id, SCHEMA.label])


def build_train_pair_dataset(
    training_pairs: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
    negatives_per_positive: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    terms = validate_dataframe(terms, TableKind.TERMS)
    items = validate_dataframe(items, TableKind.ITEMS)
    positives = normalize_positive_pairs(training_pairs)
    negatives = sample_negative_pairs(
        positives,
        all_item_ids=items[SCHEMA.item_id].dropna().tolist(),
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    pairs = pd.concat([positives, negatives], ignore_index=True)
    pairs = merge_terms(pairs, terms)
    pairs = merge_items(pairs, items)
    pairs = validate_dataframe(pairs, TableKind.TRAIN_FEATURES)
    return pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_test_pair_dataset(submission_pairs: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    terms = validate_dataframe(terms, TableKind.TERMS)
    items = validate_dataframe(items, TableKind.ITEMS)
    pairs = validate_dataframe(submission_pairs, TableKind.SUBMISSION_PAIRS)
    pairs = pairs[[SCHEMA.id, SCHEMA.term_id, SCHEMA.item_id]].copy()
    pairs = merge_terms(pairs, terms)
    pairs = merge_items(pairs, items)
    pairs = validate_dataframe(pairs, TableKind.TEST_FEATURES)
    return pairs.reset_index(drop=True)
