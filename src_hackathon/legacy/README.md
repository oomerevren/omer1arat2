# Legacy final-mode source map

The following existing folders are considered Hackathon/Final or legacy until
explicitly ported:

- `src/deployment/` -> `src_hackathon/deployment/`
- `src/xai/` -> `src_hackathon/explainability/`
- `src/inference/onnx_export.py`, `src/inference/quantizer.py` -> `src_hackathon/speed/`
- `src/models/`, `src/training/`, `src/evaluation/` with `num_labels=3` -> `src_hackathon/multiclass/`
- `scripts/run_dashboard.py`, `scripts/run_xai_demo.py`, Docker/monitoring scripts -> `scripts_hackathon/`

They are intentionally not imported by `scripts_kaggle/`.
