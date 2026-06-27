"""End-to-end championship readiness audit."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any
import pandas as pd

from src_kaggle.audit.evidence_collector import collect_evidence
from src_kaggle.audit.readiness_scorer import component_row
from src_kaggle.audit.risk_prioritizer import build_gap_list, top_risks
from src_kaggle.audit.go_no_go_decider import decide_go_no_go
from src_kaggle.audit.family_audit import audit_families


def build_component_table(e: dict[str, Any]) -> pd.DataFrame:
    f=e["files"]; rel=e.get("release_validation", {})
    rows=[]
    add=rows.append
    has=lambda k: bool(f.get(k,{}).get("exists"))
    tests=e.get("tests_kaggle_count",0)
    ab_rows=e.get("ablation_rows",0); ab_done=e.get("ablation_completed_rows",0)
    pub=e.get("leaderboard_public_points",0)
    release_ready=bool(rel.get("release_ready")); metadata_ready=bool(rel.get("metadata_lock_ready"))
    submissions_exist=all(e.get("submission_paths",{}).values()) if e.get("submission_paths") else False

    add(component_row("data contract",5,"düşük","yüksek","orta","yüksek",["src_kaggle/data/contracts.py","configs/kaggle/war_mode.yaml data_contract"],[],"Maintain strict contract checks before every data build.","P1"))
    add(component_row("pair pipeline",5,"düşük","yüksek","orta","yüksek",["src_kaggle/data/pair_builder.py","build_pair_dataset.py"],[],"Run pair builder on official data and inspect data quality reports.","P1"))
    add(component_row("attribute parser",5,"düşük","orta","düşük","orta-yüksek",["src_kaggle/data/attribute_parser.py","tests_kaggle/test_attribute_parser_examples.py"],[],"Keep parser tests in smoke suite; inspect rare attribute formats on official data.","P2"))
    add(component_row("query intent",5,"düşük","orta","düşük","orta-yüksek",["src_kaggle/features/query_intent.py","tests_kaggle/test_query_intent_examples.py"],[],"Monitor segment distributions after official data join.","P2"))
    add(component_row("retrieval",4,"orta","orta","düşük","orta",["src_kaggle/retrieval/hybrid_retriever.py","reports/retrieval design docs"],["official retrieval evaluation not run in this workspace"],"Run build_retrieval_index + evaluate_dense_retrieval on official items/queries.","P2"))
    add(component_row("dense retrieval",3,"orta-yüksek","yüksek","düşük","orta",["real_dense backend guard implemented","reports/leaderboard dense risk flags"],["real dense model/index not materialized in workspace","public/OOF dense lift unknown"],"Build real dense index with official items; compare dense_v1/v2 and BM25 overlap before final use.","P1"))
    add(component_row("negative mining",4,"orta","yüksek","düşük","orta",["src_kaggle/data/negative_mining.py","false-negative safety statuses"],["fold-aware negative rebuild not fully proven on official OOF"],"Use fold-aware negative builds for final validation; inspect uncertain/skipped rates.","P1"))
    add(component_row("semantic hard negatives",3,"yüksek","yüksek","düşük","orta",["dense hard negative subtype/report code"],["semantic hard negative OOF gain unproven","false negative risk remains"],"Keep dense hard negatives out of default unless ablation shows class0-safe lift.","P0"))
    add(component_row("feature engineering",5,"düşük-orta","yüksek","düşük","yüksek",["feature pipeline tests",f"tests_kaggle_count={tests}"],[],"Run feature catalog on official data; drop constant/leaky features if flagged.","P1"))
    add(component_row("tabular modeling",4,"orta","yüksek","düşük","orta",["src_kaggle/models/tabular_model.py","validation runner"],["no official OOF metrics in workspace"],"Run official OOF; keep tabular as baseline/family A backbone until CE proves lift.","P0"))
    add(component_row("transformer cross-encoder",3,"yüksek","yüksek","orta","orta",["real transformers path implemented with no silent fallback"],["real transformer fine-tuning not run here","operational GPU/dependency risk"],"Run GPU OOF with backend=transformers; compare against sklearn_text and tabular before final weighting.","P0"))
    add(component_row("ensemble",3,"orta-yüksek","yüksek","orta","orta",["src_kaggle/final/ensemble_optimizer.py","final blend comparison"],["no ready OOF candidates, no prediction-correlation evidence"],"Materialize OOF/test preds; reject highly correlated redundant blends.","P0"))
    add(component_row("validation framework",4,"orta","çok yüksek","düşük","orta",["term_group/query_group/tail_aware splitters","threshold/segment reports"],["public/OOF correlation has 0 public points"],"Run splitter ablation on official data and track public/OOF correlation over submissions.","P0"))
    add(component_row("OOF engine",4,"orta","çok yüksek","düşük","orta",["run_experiment.py","ablation runner"],[f"completed ablation rows={ab_done}/{ab_rows}"],"Complete official OOF ablations for selected final candidates.","P0"))
    add(component_row("threshold optimization",4,"orta","yüksek","orta","orta",["threshold_optimizer.py","threshold reliability report"],["threshold stability not measured on real final OOF"],"Use OOF global/stable midpoint; never tune directly to public LB.","P1"))
    add(component_row("error analysis",4,"düşük-orta","orta","düşük","orta",["src_kaggle/analysis","error reports scripts"],["requires real OOF errors to populate final insights"],"Run after first official OOF; inspect FP/FN class0-heavy segments.","P2"))
    add(component_row("pseudo labeling",4,"orta","orta-yüksek","düşük","orta",["controlled pseudo label module disabled by default"],["not validated as final lift"],"Keep disabled unless late ablation proves safe; avoid final unproven pseudo labels.","P2"))
    add(component_row("experiment tracking",5,"düşük","orta","orta","yüksek",["experiment registry","master logs","ablation master table"],[],"Continue recording all final submissions/public scores.","P1"))
    add(component_row("public/OOF/private-LB strategy",3 if pub==0 else 4,"orta","çok yüksek","orta","orta",["leaderboard analysis reports",f"public_points={pub}"],["no public LB history yet" if pub==0 else "limited public sample"],"Add public LB entries after every upload; do not over-infer under n<3.","P0"))
    add(component_row("submission safety",5 if has("submission_registry") or has("manifest") else 4,"düşük","orta","çok yüksek","yüksek",["submission_validator.py","submission checklist"],[],"Hard gate every final CSV through validator before upload.","P0"))
    add(component_row("final family configs",5 if metadata_ready else 3,"düşük","orta","yüksek","yüksek",["configs/kaggle/final/*.yaml","final_mode/frozen guards"],[] if metadata_ready else ["metadata lock not ready"],"Do not edit final configs directly; use experiments/promote workflow.","P0"))
    add(component_row("artefact manifest",5 if has("manifest") else 2,"orta","orta","çok yüksek","yüksek",["reports/final/final_artifact_manifest.json"],[] if has("manifest") else ["manifest missing"],"Rebuild manifest after any final candidate/family change.","P0"))
    add(component_row("reproducibility",4,"orta","orta","yüksek","orta-yüksek",["docs/competition_freeze.md","package/validate scripts"],["full dry-run impossible without official data/test predictions"],"Run final dry-run after official artefacts arrive; archive command outputs.","P1"))
    add(component_row("competition freeze",5 if metadata_ready else 3,"düşük","orta","yüksek","yüksek",["final config freeze index","submission day checklist"],[] if metadata_ready else ["freeze validator errors"],"Use validate_final_release as race-day gate.","P0"))
    add(component_row("docs / jury readiness",5,"düşük","düşük","orta","yüksek",["README.md","docs/*", "final reports"],[],"Keep README pointing to Kaggle final path; avoid legacy confusion.","P2"))
    add(component_row("actual final submissions",5 if submissions_exist and release_ready else 2,"orta","yüksek","çok yüksek","yüksek",["final release validation report"],[] if submissions_exist else ["family submission.csv files not materialized","validator reports pending"],"Generate official family submissions from real test predictions and rerun release validation until release_ready=true.","P0"))
    return pd.DataFrame(rows)


def write_reports(out_dir: str | Path = "reports/final") -> dict[str, Any]:
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    e=collect_evidence()
    comp=build_component_table(e)
    fam=audit_families(e)
    gaps=build_gap_list(comp)
    risks=top_risks(comp, 10)
    decision=decide_go_no_go(comp, e)
    comp.to_csv(out/"championship_component_status.csv", index=False)
    gaps.to_csv(out/"championship_gap_list.csv", index=False)
    fam.to_csv(out/"family_readiness_table.csv", index=False)
    # private risk matrix: components with private impact high/very high
    comp[comp["private_lb_impact"].isin(["yüksek","çok yüksek"])].to_csv(out/"private_lb_risk_matrix.csv", index=False)
    (out/"championship_go_no_go.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown(out, comp, gaps, risks, fam, decision, e)
    return {"component_table": comp, "gap_list": gaps, "family_table": fam, "decision": decision, "evidence": e}


def write_markdown(out: Path, comp: pd.DataFrame, gaps: pd.DataFrame, risks: pd.DataFrame, fam: pd.DataFrame, decision: dict, e: dict) -> None:
    counts=comp["status"].value_counts().to_dict()
    top5_risks = risks.head(5)
    top_fixes = gaps[gaps["status"].eq("open")].head(5)
    lines=[
        "# Championship Readiness Report", "",
        "## 1. Genel özet", "",
        f"Final decision: **{decision['final_decision']}**", "",
        decision["rationale"], "",
        f"Component status counts: `{counts}`", "",
        f"Evidence: ablation_rows={e.get('ablation_rows')}, completed_ablation_rows={e.get('ablation_completed_rows')}, public_points={e.get('leaderboard_public_points')}, tests_kaggle={e.get('tests_kaggle_count')}", "",
        "## 2. Teknik readiness tablosu", "", comp[comp["component_name"].isin(["data contract","pair pipeline","attribute parser","query intent","retrieval","dense retrieval","negative mining","semantic hard negatives","feature engineering","tabular modeling","transformer cross-encoder","ensemble","validation framework","OOF engine","threshold optimization"])][["component_name","status","technical_risk","private_lb_impact","confidence_level","open_gaps","priority"]].to_csv(index=False), "",
        "## 3. Operasyonel readiness tablosu", "", comp[comp["component_name"].isin(["submission safety","final family configs","artefact manifest","reproducibility","competition freeze","docs / jury readiness","actual final submissions"])][["component_name","status","submission_day_impact","confidence_level","open_gaps","priority"]].to_csv(index=False), "",
        "## 4. Private LB risk özeti", "", "En yüksek private risk alanları: semantic hard negatives, transformer CE, ensemble, validation/public-OOF strategy, actual OOF evidence eksikliği.", "", comp[comp["private_lb_impact"].isin(["yüksek","çok yüksek"])][["component_name","status","open_gaps","recommended_action"]].to_csv(index=False), "",
        "## 5. Submission-day risk özeti", "", "Submission safety kodu ve manifest hazır; fakat gerçek family submission.csv dosyaları official test predictions olmadığı için materialize değil. Bu nedenle immediate upload için NO-GO.", "",
        "## 6. Family A/B/C mini audit", "", fam.to_csv(index=False), "",
        "## 7. Top 5 remaining risks", "", top5_risks[["component_name","status","private_lb_impact","submission_day_impact","open_gaps","recommended_action"]].to_csv(index=False), "",
        "## 8. Top 5 highest-leverage final fixes", "", top_fixes[["gap_id","component","description","severity","recommended_fix"]].to_csv(index=False), "",
        "## 9. GO / GO WITH RISKS / NO-GO kararı", "", f"**{decision['final_decision']}**", "", decision["rationale"], "",
        "Blocking issues:", "", json.dumps(decision.get("blocking_issues", []), indent=2, ensure_ascii=False), "",
        "## 10. Son hafta aksiyon planı", "", "1. Official data ve test prediction artefact'larını yerleştir.", "2. Real OOF ablationları tamamla: tabular baseline, CE, dense features, dense negatives, family blends.", "3. Family A/B/C submission.csv dosyalarını üret ve validator'dan geçir.", "4. package_final_families.py + validate_final_release.py çalıştır; release_ready=true olana kadar upload yapma.", "5. Public skorları tracking table'a işle, OOF/class0/risk flag sırasını koru.",
    ]
    (out/"championship_readiness_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (out/"championship_top_risks.md").write_text("# Championship Top Risks\n\n"+top5_risks[["component_name","status","open_gaps","recommended_action"]].to_csv(index=False), encoding="utf-8")
    (out/"championship_top_fixes.md").write_text("# Championship Top Fixes\n\n"+top_fixes[["gap_id","component","recommended_fix"]].to_csv(index=False), encoding="utf-8")
