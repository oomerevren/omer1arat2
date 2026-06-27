from __future__ import annotations
import pandas as pd


def audit_families(evidence: dict) -> pd.DataFrame:
    rows=[]
    statuses={r.get("family_name"): r for r in evidence.get("family_statuses", [])}
    submissions=evidence.get("submission_paths", {})
    for fam, role in [("family_A_balanced","default balanced"),("family_B_defensive","class0 defensive"),("family_C_aggressive","semantic aggressive")]:
        st=statuses.get(fam, {})
        sub_exists = submissions.get(f"artifacts/final/submissions/{fam}_submission.csv", False)
        if not st:
            status="red"; gap="family row missing"
        elif not sub_exists:
            status="yellow" if st.get("status") else "red"; gap="metadata exists but validated submission.csv pending"
        else:
            status="green"; gap="none"
        rows.append({"family_name":fam,"role":role,"status":status,"risk_label":st.get("public_private_risk_label", ""),"artifact_status":st.get("status", "missing"),"submission_exists":sub_exists,"open_gap":gap})
    return pd.DataFrame(rows)
