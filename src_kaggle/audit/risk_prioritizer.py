from __future__ import annotations
import pandas as pd


def build_gap_list(component_table: pd.DataFrame) -> pd.DataFrame:
    rows=[]; gid=1
    sev_map={"red":"critical","yellow":"major","green":"minor"}
    for _, r in component_table.iterrows():
        gaps = str(r.get("open_gaps", "none"))
        if gaps == "none" and r.get("status") == "green":
            continue
        rows.append({
            "gap_id": f"GAP-{gid:03d}",
            "component": r["component_name"],
            "description": gaps,
            "severity": sev_map.get(r["status"], "major"),
            "private_lb_impact": r["private_lb_impact"],
            "submission_day_impact": r["submission_day_impact"],
            "effort_estimate": "S" if r["component_name"] in {"submission safety", "final family configs", "artefact manifest", "competition freeze"} else "M/L",
            "owner": "modeling/ops team",
            "status": "open" if r["status"] != "green" else "monitor",
            "recommended_fix": r["recommended_action"],
        }); gid += 1
    return pd.DataFrame(rows)


def top_risks(component_table: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    pri = {"P0":0,"P1":1,"P2":2,"P3":3}
    st = {"red":0,"yellow":1,"green":2}
    df=component_table.copy()
    df["_p"] = df["priority"].map(pri).fillna(9)
    df["_s"] = df["status"].map(st).fillna(9)
    return df.sort_values(["_s","_p"]).head(n).drop(columns=["_p","_s"])
