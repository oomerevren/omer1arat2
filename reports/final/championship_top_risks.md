# Championship Top Risks

component_name,status,open_gaps,recommended_action
actual final submissions,red,family submission.csv files not materialized | validator reports pending,Generate official family submissions from real test predictions and rerun release validation until release_ready=true.
semantic hard negatives,yellow,semantic hard negative OOF gain unproven | false negative risk remains,Keep dense hard negatives out of default unless ablation shows class0-safe lift.
tabular modeling,yellow,no official OOF metrics in workspace,Run official OOF; keep tabular as baseline/family A backbone until CE proves lift.
transformer cross-encoder,yellow,real transformer fine-tuning not run here | operational GPU/dependency risk,Run GPU OOF with backend=transformers; compare against sklearn_text and tabular before final weighting.
ensemble,yellow,"no ready OOF candidates, no prediction-correlation evidence",Materialize OOF/test preds; reject highly correlated redundant blends.
