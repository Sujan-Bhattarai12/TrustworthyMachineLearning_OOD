"""Development pilot warning sensitivity versus trajectory false alarms."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.evaluate_early_warnings import TRAJECTORY, calibrate_threshold

ROOT=Path('results/local_sparse_map_pilot')
BUDGETS=[0.01,0.02,0.05,0.10,0.20]

def summarize(frame):
    records=[]
    for identity,g in frame.groupby(TRAJECTORY):
        failure=g.first_failure.iloc[0]
        eligible=g[(g.severity>0)&((g.severity<failure) if np.isfinite(failure) else True)]
        records.append(dict(zip(TRAJECTORY,identity),first_failure=failure,
            maximum=float(eligible.score.max()) if len(eligible) else -np.inf))
    return pd.DataFrame(records)

def rates(frame,threshold):
    failing=frame.first_failure.notna();early=frame.loc[failing,'maximum']>threshold
    false=frame.loc[~failing,'maximum']>threshold
    return dict(early_warning_rate=early.mean(),false_alarm_rate=false.mean(),n_early=int(early.sum()),n_false=int(false.sum()),n_failures=int(failing.sum()),n_nonfailures=int((~failing).sum()))

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    scores=pd.read_csv(ROOT/'observation_scores.csv');calibrated=[];matched=[];curves=[]
    for (method,representation),part in scores.groupby(['method','representation']):
        cal=summarize(part[part.image_group.isin(protocol['calibration_images'])])
        val=summarize(part[part.image_group.isin(protocol['validation_images'])])
        negative_cal=cal.loc[cal.first_failure.isna(),'maximum'].to_numpy()
        negative_val=val.loc[val.first_failure.isna(),'maximum'].to_numpy()
        for budget in BUDGETS:
            threshold=calibrate_threshold(negative_cal,budget)
            calibrated.append(dict(method=method,representation=representation,target_calibration_fpr=budget,
                calibration_fpr=float(np.mean(negative_cal>threshold)),threshold=threshold,**rates(val,threshold)))
            # Lowest threshold satisfying the validation FPR cap; descriptive only.
            threshold=calibrate_threshold(negative_val,budget)
            matched.append(dict(method=method,representation=representation,validation_fpr_cap=budget,threshold=threshold,**rates(val,threshold)))
        for threshold in np.r_[-np.inf,np.unique(val.maximum[np.isfinite(val.maximum)])]:
            curves.append(dict(method=method,representation=representation,threshold=threshold,**rates(val,threshold)))
    pd.DataFrame(calibrated).to_csv(ROOT/'calibrated_operating_points.csv',index=False)
    matched=pd.DataFrame(matched);matched.to_csv(ROOT/'descriptive_matched_fpr.csv',index=False)
    curves=pd.DataFrame(curves);curves.to_csv(ROOT/'descriptive_warning_curves.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,method in zip(axes,['ig','gradcam']):
        for group in [method,'baseline']:
            for name,g in curves[curves.method==group].groupby('representation'):
                g=g.sort_values(['false_alarm_rate','early_warning_rate'])
                ax.step(g.false_alarm_rate*100,g.early_warning_rate*100,where='post',label=name)
        ax.set(xlim=(0,20),ylim=(0,100),title=method.upper(),xlabel='Nonfailing trajectories with an alarm (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_ylabel('Failing trajectories warned before first error (%)')
    fig.suptitle('Exploratory development holdout: threshold-swept curves, not frozen test results',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'warning_false_alarm_tradeoff.png',dpi=180);fig.savefig(ROOT/'warning_false_alarm_tradeoff.pdf');plt.close(fig)
    print(matched[matched.validation_fpr_cap.isin([.02,.05,.1])].to_string(index=False))

if __name__=='__main__':main()
