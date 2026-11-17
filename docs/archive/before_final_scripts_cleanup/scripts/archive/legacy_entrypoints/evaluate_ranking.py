"""Secondary next-step ranking of frozen scores, without fitting."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from scripts.freeze_warning_methods import predict

def ranking_metrics(data,artifact):
    # First-failure next-step task; no failed/recovered steps or final horizon step.
    mask=data.eligible_fit.to_numpy(dtype=bool)
    eligible=data.loc[mask]
    y=eligible.fails_next_step.to_numpy(dtype=int)
    summaries,predictions,rocs,prs=[],[],[],[]
    for spec in artifact['methods']:
        score=predict(spec,data)[mask]
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite frozen warning score.')
        both=np.unique(y).size==2
        summaries.append(dict(classifier=artifact['classifier'],model=spec['name'],n_steps=len(y),
            n_images=eligible.image_group.nunique(),n_next_step_failures=int(y.sum()),
            failure_prevalence=float(y.mean()) if len(y) else np.nan,
            auroc=roc_auc_score(y,score) if both else np.nan,
            aupr=average_precision_score(y,score) if both else np.nan,
            status='ok' if both else 'undefined: requires both failure and nonfailure labels'))
        frame=eligible[['source','image_id','corruption','severity','first_failure','fails_next_step']].copy()
        frame['model'],frame['score'],frame['classifier']=spec['name'],score,artifact['classifier']
        predictions.append(frame)
        if both:
            fpr,tpr,threshold=roc_curve(y,score)
            rocs.append(pd.DataFrame(dict(classifier=artifact['classifier'],model=spec['name'],fpr=fpr,tpr=tpr,threshold=threshold)))
            precision,recall,threshold=precision_recall_curve(y,score)
            prs.append(pd.DataFrame(dict(classifier=artifact['classifier'],model=spec['name'],precision=precision,recall=recall,threshold=np.r_[threshold,np.nan])))
    return pd.DataFrame(summaries),pd.concat(predictions,ignore_index=True),pd.concat(rocs,ignore_index=True) if rocs else pd.DataFrame(),pd.concat(prs,ignore_index=True) if prs else pd.DataFrame()

