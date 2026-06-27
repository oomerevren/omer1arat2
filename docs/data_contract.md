# Data Contract

Official schema:

- `items.csv`: `item_id,title,category,brand,gender,age_group,attributes`
- `terms.csv`: `term_id,query`
- `training_pairs.csv`: `id,term_id,item_id,label`
- `submission_pairs.csv`: `id,term_id,item_id`
- `sample_submission.csv`: `id,prediction`

Strict contract implementation: `src_kaggle/data/schema.py`, `src_kaggle/data/contracts.py`.
