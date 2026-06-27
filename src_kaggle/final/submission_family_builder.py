"""Build three final submission families and artefact metadata."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src_kaggle.submission.submission_builder import build_submission_from_proba, save_safe_submission, append_submission_registry
from src_kaggle.data.io import read_table

FAMILY_DEFS = [
    {"family_name": "family_A_balanced", "display_name": "Balanced / Safest", "risk_label": "safest_balanced", "purpose": "Default private-safe final candidate", "threshold_policy": "OOF best or stable midpoint", "weight_policy": "OOF-weighted, penalize risky candidates"},
    {"family_name": "family_B_defensive", "display_name": "Class0 / Precision / Defensive", "risk_label": "private_defensive", "purpose": "Protect irrelevant class and reduce false positives", "threshold_policy": "class0-protective threshold", "weight_policy": "increase tabular/conflict-signal weight"},
    {"family_name": "family_C_aggressive", "display_name": "Semantic / Recall / Aggressive", "risk_label": "semantic_aggressive", "purpose": "Semantic/CE/dense challenger if risk flags are low", "threshold_policy": "slightly recall-friendly threshold", "weight_policy": "increase CE/dense semantic weight"},
]


def _norm_weights(weights: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s <= 0:
        n = len(weights) or 1
        return {k: 1/n for k in weights}
    return {k: max(0.0, float(v))/s for k, v in weights.items()}


def choose_family_models(pool: pd.DataFrame, blend_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ready = pool[pool.get("selection_status", "").eq("ready")].copy() if not pool.empty else pd.DataFrame()
    families: dict[str, dict[str, Any]] = {}
    for fd in FAMILY_DEFS:
        fam = dict(fd)
        if ready.empty:
            fam.update({"models": [], "weights": {}, "threshold": None, "OOF macro-F1": pd.NA, "class0 F1": pd.NA, "class1 F1": pd.NA, "seed_std": pd.NA, "threshold_fragility": "unknown", "status": "not_ready_no_oof_or_test_predictions", "notes": "Official OOF/test prediction artefacts are required to materialize submission.csv."})
        else:
            if fd["family_name"] == "family_A_balanced":
                candidates = ready[~ready["risk_label"].astype(str).str.contains("PUBLIC_UP|PRIVATE_UNSAFE", na=False)].sort_values(["selection_score"], ascending=False).head(3)
            elif fd["family_name"] == "family_B_defensive":
                candidates = ready.sort_values(["class0 F1", "selection_score"], ascending=False).head(3)
            else:
                sem = ready[ready["model_family"].astype(str).str.contains("dense|ce|text|ensemble|semantic", case=False, na=False)]
                candidates = (sem if not sem.empty else ready).sort_values(["selection_score"], ascending=False).head(3)
            weights = _norm_weights({str(r["candidate_id"]): float(r.get("selection_score", 0) if pd.notna(r.get("selection_score", pd.NA)) else 0.1) for _, r in candidates.iterrows()})
            thr = candidates["threshold"].dropna().median() if "threshold" in candidates else 0.5
            if fd["family_name"] == "family_B_defensive" and pd.notna(thr): thr = min(0.95, float(thr) + 0.04)
            if fd["family_name"] == "family_C_aggressive" and pd.notna(thr): thr = max(0.05, float(thr) - 0.03)
            fam.update({"models": list(weights), "weights": weights, "threshold": float(thr) if pd.notna(thr) else 0.5, "OOF macro-F1": candidates["OOF macro-F1"].max(), "class0 F1": candidates["class0 F1"].max(), "class1 F1": candidates["class1 F1"].max(), "seed_std": candidates["seed_std"].dropna().mean() if "seed_std" in candidates else pd.NA, "threshold_fragility": "monitor", "status": "ready_metadata_only", "notes": "Family selected from ready OOF candidates; submission materialization requires test_pred_path per model."})
        families[fd["family_name"]] = fam
    return families


def _load_test_pred(path: str | Path) -> pd.DataFrame | None:
    p = Path(str(path))
    if not str(path) or not p.exists(): return None
    df = pd.read_csv(p)
    if "proba" not in df.columns: return None
    if "id" not in df.columns: return None
    return df[["id", "proba"]].copy()


def materialize_family_artifacts(
    families: dict[str, dict[str, Any]],
    pool: pd.DataFrame,
    submission_pairs_path: str | Path | None = None,
    artifact_root: str | Path = "artifacts/final_submissions",
    configs_root: str | Path = "configs/kaggle/final",
    registry_path: str | Path = "reports/submissions/submission_registry.csv",
) -> pd.DataFrame:
    root = Path(artifact_root); cfg_root = Path(configs_root); root.mkdir(parents=True, exist_ok=True); cfg_root.mkdir(parents=True, exist_ok=True)
    ref = read_table(submission_pairs_path) if submission_pairs_path and Path(str(submission_pairs_path)).exists() else None
    pool_by_id = {str(r["candidate_id"]): r.to_dict() for _, r in pool.iterrows()} if not pool.empty else {}
    rows = []
    for name, fam in families.items():
        d = root / name; d.mkdir(parents=True, exist_ok=True)
        metadata = {
            "family_name": name, "display_name": fam.get("display_name"), "risk_label": fam.get("risk_label"),
            "purpose": fam.get("purpose"), "models_used": fam.get("models", []), "blend_weights": fam.get("weights", {}),
            "threshold": fam.get("threshold"), "source_oof_reports": [pool_by_id.get(m, {}).get("oof_path", "") for m in fam.get("models", [])],
            "experiment_ids": [pool_by_id.get(m, {}).get("experiment_id", "") for m in fam.get("models", [])],
            "created_at": datetime.now(timezone.utc).isoformat(), "status": fam.get("status"), "notes": fam.get("notes"),
        }
        # Try to build a real submission only if all test prediction files and official reference exist.
        can_build = ref is not None and fam.get("models") and fam.get("threshold") is not None
        probs = None
        if can_build:
            probs = np.zeros(len(ref), dtype=float)
            for mid, w in fam.get("weights", {}).items():
                pred = _load_test_pred(pool_by_id.get(mid, {}).get("test_pred_path", ""))
                if pred is None:
                    can_build = False; metadata["status"] = "not_materialized_missing_test_predictions"; break
                aligned = ref[["id"]].merge(pred, on="id", how="left")
                if aligned["proba"].isna().any():
                    can_build = False; metadata["status"] = "not_materialized_prediction_id_mismatch"; break
                probs += float(w) * aligned["proba"].values
        if can_build and probs is not None:
            sub = build_submission_from_proba(ref, probs, float(fam["threshold"]))
            report_path = d / "submission_validation_report.json"
            save_safe_submission(sub, ref, d / "submission.csv", report_path=report_path)
            metadata["submission_path"] = str(d / "submission.csv")
            metadata["validation_report"] = str(report_path)
            metadata["status"] = "submission_ready_validated"
            append_submission_registry({"experiment_id": name, "models": "|".join(fam.get("models", [])), "threshold": fam.get("threshold"), "validation_score": fam.get("OOF macro-F1", ""), "class0_f1": fam.get("class0 F1", ""), "class1_f1": fam.get("class1 F1", ""), "file_path": str(d / "submission.csv"), "validation_report": str(report_path), "note": fam.get("purpose", "")}, registry_path=registry_path)
        else:
            metadata.setdefault("submission_path", "")
            metadata["status"] = metadata.get("status") or "not_materialized_missing_reference_or_predictions"
        (d / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (cfg_root / f"{name}.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
        rows.append({
            "family_name": name, "used_models": "|".join(fam.get("models", [])), "blend_method": "weighted_average",
            "blend_weights": json.dumps(fam.get("weights", {}), ensure_ascii=False), "threshold": fam.get("threshold"),
            "OOF macro-F1": fam.get("OOF macro-F1"), "class0 F1": fam.get("class0 F1"), "class1 F1": fam.get("class1 F1"),
            "seed_std": fam.get("seed_std"), "splitter_alignment": "term_group_default_private_safe", "threshold_fragility": fam.get("threshold_fragility"),
            "public_private_risk_label": fam.get("risk_label"), "segment_strengths": "see final_family_segment_scores.csv",
            "segment_weaknesses": "see final_family_segment_scores.csv", "notes": fam.get("notes"), "status": metadata["status"],
            "artifact_dir": str(d), "config_path": str(cfg_root / f"{name}.yaml"),
        })
    return pd.DataFrame(rows)
