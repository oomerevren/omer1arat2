# Submission Process

Submission format strictly:

```text
id,prediction
```

Builder/validator:

- `src_kaggle/submission/submission_builder.py`
- `src_kaggle/submission/submission_validator.py`
- `scripts_kaggle/make_submission.py`

Validation checks:

- exact columns
- row count
- id set/order
- duplicate/missing id
- binary prediction
- null prediction
- positive rate warning
- previous submission drift warning

Registry: `reports/submissions/submission_registry.csv`.
