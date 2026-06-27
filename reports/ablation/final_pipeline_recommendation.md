# Final Pipeline Recommendation

Official OOF evidence is not available in this workspace; recommendations below are guarded defaults and must be replaced by completed ablation metrics.

## Tutulacak bileşenler

- Pair-centric schema/data contract
- OOF-first validation + threshold tuning
- Submission safety layer
- Attribute/gender-age conflict features
- Negative mining with false-negative safety

## Opsiyonel ama riskli bileşenler

| ablation_id | category | changed_component | risk_flag |
| --- | --- | --- | --- |
| feat_all_features | feature | all_features | needs_fold_safe_recheck |
| feat_no_lexical | feature | use_lexical_features | needs_fold_safe_recheck |
| feat_no_category | feature | use_category_features | needs_fold_safe_recheck |
| feat_no_brand | feature | use_brand_features | needs_fold_safe_recheck |
| feat_no_attribute | feature | use_attribute_features | needs_fold_safe_recheck |
| feat_no_gender_age | feature | use_gender_age_features | needs_fold_safe_recheck |
| feat_no_query_features | feature | use_query_features | needs_fold_safe_recheck |
| feat_no_retrieval | feature | use_retrieval_features | needs_fold_safe_recheck |
| feat_no_semantic | feature | use_semantic_features | needs_fold_safe_recheck |
| feat_no_metadata | feature | use_metadata_features | needs_fold_safe_recheck |
| feat_only_core_features | feature | only_core_features | needs_fold_safe_recheck |
| feat_only_lexical_category_brand | feature | only_lexical_category_brand | needs_fold_safe_recheck |
| feat_only_attribute_gender_age | feature | only_attribute_gender_age | needs_fold_safe_recheck |
| feat_lexical_plus_retrieval | feature | lexical_plus_retrieval | needs_fold_safe_recheck |
| feat_attribute_plus_semantic | feature | attribute_plus_semantic | needs_fold_safe_recheck |
| neg_easy_only | negative | easy | needs_fold_safe_recheck |
| neg_easy_plus_same_category | negative | same_category | needs_fold_safe_recheck |
| neg_easy_plus_same_brand | negative | same_brand | needs_fold_safe_recheck |
| neg_easy_plus_attribute_conflict | negative | attribute_conflict | needs_fold_safe_recheck |
| neg_easy_plus_lexical_confusing | negative | lexical_confusing | needs_fold_safe_recheck |

## Çıkarılacak / sadeleştirilecek bileşenler

- No component can be dropped without official OOF evidence yet.

## Model family önerisi

Default final candidate: tabular strong baseline + CE OOF if CE adds class-0/class-1 complementary lift. Transformer is helper until real OOF proves it should be core.

## Dense önerisi

Dense should first be used as an auxiliary signal and hard-negative source. It becomes core only if dense_features_plus_dense_negatives beats no_dense_anywhere with stable class0 F1 and no fold-safety alarm.

## Negative mix önerisi

Start with full mix without unsafe dense; add dense hard negatives only with strict false-negative filtering and fold-aware rebuild.

## Threshold önerisi

Use OOF global best or stable-midpoint. Segment thresholds remain analysis-only unless validated across seeds and private-LB simulation.
