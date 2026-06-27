"""Canonical TEKNOFEST 2026 Kaggle data schema.

This is the single source of truth for Kaggle War Mode column names.
Do not use legacy names such as product_id, product_name, search_query,
product_color or product_material in `src_kaggle` / `scripts_kaggle`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KaggleSchema:
    # Shared identifiers
    id: str = "id"
    term_id: str = "term_id"
    item_id: str = "item_id"

    # Terms table
    query: str = "query"

    # Items table
    title: str = "title"
    category: str = "category"
    brand: str = "brand"
    gender: str = "gender"
    age_group: str = "age_group"
    attributes: str = "attributes"
    attribute_dict: str = "attribute_dict"
    attribute_keys: str = "attribute_keys"
    attribute_values: str = "attribute_values"
    normalized_attribute_text: str = "normalized_attribute_text"
    color_value: str = "color_value"
    material_value: str = "material_value"
    style_value: str = "style_value"
    full_item_text: str = "full_item_text"
    retrieval_text: str = "retrieval_text"
    dense_text: str = "dense_text"

    # Labels / predictions
    label: str = "label"
    prediction: str = "prediction"


SCHEMA = KaggleSchema()

ITEM_REQUIRED_COLUMNS: tuple[str, ...] = (
    SCHEMA.item_id,
    SCHEMA.title,
    SCHEMA.category,
    SCHEMA.brand,
    SCHEMA.gender,
    SCHEMA.age_group,
    SCHEMA.attributes,
)

TERMS_REQUIRED_COLUMNS: tuple[str, ...] = (
    SCHEMA.term_id,
    SCHEMA.query,
)

TRAINING_PAIRS_REQUIRED_COLUMNS: tuple[str, ...] = (
    SCHEMA.id,
    SCHEMA.term_id,
    SCHEMA.item_id,
    SCHEMA.label,
)

SUBMISSION_PAIRS_REQUIRED_COLUMNS: tuple[str, ...] = (
    SCHEMA.id,
    SCHEMA.term_id,
    SCHEMA.item_id,
)

SAMPLE_SUBMISSION_REQUIRED_COLUMNS: tuple[str, ...] = (
    SCHEMA.id,
    SCHEMA.prediction,
)

CANONICAL_FIELD_MAP: dict[str, str] = {
    "id": SCHEMA.id,
    "term_id": SCHEMA.term_id,
    "item_id": SCHEMA.item_id,
    "query": SCHEMA.query,
    "title": SCHEMA.title,
    "category": SCHEMA.category,
    "brand": SCHEMA.brand,
    "gender": SCHEMA.gender,
    "age_group": SCHEMA.age_group,
    "attributes": SCHEMA.attributes,
    "attribute_dict": SCHEMA.attribute_dict,
    "attribute_keys": SCHEMA.attribute_keys,
    "attribute_values": SCHEMA.attribute_values,
    "normalized_attribute_text": SCHEMA.normalized_attribute_text,
    "color_value": SCHEMA.color_value,
    "material_value": SCHEMA.material_value,
    "style_value": SCHEMA.style_value,
    "full_item_text": SCHEMA.full_item_text,
    "retrieval_text": SCHEMA.retrieval_text,
    "dense_text": SCHEMA.dense_text,
    "label": SCHEMA.label,
    "prediction": SCHEMA.prediction,
}

TRAIN_MERGED_COLUMNS: tuple[str, ...] = (
    SCHEMA.id,
    SCHEMA.term_id,
    SCHEMA.item_id,
    SCHEMA.label,
    SCHEMA.query,
    SCHEMA.title,
    SCHEMA.category,
    SCHEMA.brand,
    SCHEMA.gender,
    SCHEMA.age_group,
    SCHEMA.attributes,
    SCHEMA.attribute_dict,
    SCHEMA.attribute_keys,
    SCHEMA.attribute_values,
    SCHEMA.normalized_attribute_text,
    SCHEMA.color_value,
    SCHEMA.material_value,
    SCHEMA.style_value,
    SCHEMA.full_item_text,
)

TEST_MERGED_COLUMNS: tuple[str, ...] = (
    SCHEMA.id,
    SCHEMA.term_id,
    SCHEMA.item_id,
    SCHEMA.query,
    SCHEMA.title,
    SCHEMA.category,
    SCHEMA.brand,
    SCHEMA.gender,
    SCHEMA.age_group,
    SCHEMA.attributes,
    SCHEMA.attribute_dict,
    SCHEMA.attribute_keys,
    SCHEMA.attribute_values,
    SCHEMA.normalized_attribute_text,
    SCHEMA.color_value,
    SCHEMA.material_value,
    SCHEMA.style_value,
    SCHEMA.full_item_text,
)
