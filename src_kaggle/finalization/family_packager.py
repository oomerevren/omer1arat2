"""Package final families into standardized artifacts/final layout."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil

import pandas as pd

FAMILY_NAMES = ["family_A_balanced", "family_B_defensive", "family_C_aggressive"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def package_final_families(
    source_root: str | Path = "artifacts/final_submissions",
    final_root: str | Path = "artifacts/final",
) -> list[dict]:
    source_root = Path(source_root); final_root = Path(final_root)
    (final_root / "families").mkdir(parents=True, exist_ok=True)
    (final_root / "submissions").mkdir(parents=True, exist_ok=True)
    records = []
    for fam in FAMILY_NAMES:
        src_dir = source_root / fam
        dst_dir = final_root / "families" / fam
        dst_dir.mkdir(parents=True, exist_ok=True)
        meta = _read_json(src_dir / "metadata.json")
        meta.setdefault("family_name", fam)
        meta["standardized_at"] = datetime.now(timezone.utc).isoformat()
        meta["standard_artifact_dir"] = str(dst_dir)
        meta["standard_submission_path"] = str(final_root / "submissions" / f"{fam}_submission.csv")
        (dst_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (dst_dir / "selected_models.json").write_text(json.dumps({"family_name": fam, "models_used": meta.get("models_used", []), "experiment_ids": meta.get("experiment_ids", []), "source_oof_reports": meta.get("source_oof_reports", [])}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (dst_dir / "threshold.json").write_text(json.dumps({"family_name": fam, "threshold": meta.get("threshold"), "threshold_policy": meta.get("threshold_policy", "see final config")}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        sub_src = src_dir / "submission.csv"
        sub_dst = final_root / "submissions" / f"{fam}_submission.csv"
        if sub_src.exists():
            shutil.copy2(sub_src, sub_dst)
            try:
                pd.read_csv(sub_dst).head(20).to_csv(dst_dir / "submission_preview.csv", index=False)
            except Exception:
                pass
            sub_exists = True
        else:
            sub_exists = False
        records.append({"family_name": fam, "source_dir": str(src_dir), "standard_dir": str(dst_dir), "metadata_path": str(dst_dir / "metadata.json"), "submission_path": str(sub_dst), "submission_exists": sub_exists, "status": meta.get("status", "metadata_missing")})
    return records
