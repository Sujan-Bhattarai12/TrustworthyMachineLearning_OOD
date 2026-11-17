"""Supplement frozen fresh-image timing results with next-step AUROC/AP and curves.

Post-hoc descriptive metrics only: no fitting, recalibration, or model selection.
AUPR is reported as sklearn average precision, not trapezoidal PR area.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from scripts.archive.legacy_entrypoints.freeze_warning_methods import predict
from scripts.archive.fresh_trend_validation import sha,ARCHITECTURES
from src.attribution_features import add_features


from scripts.archive.legacy_entrypoints.evaluate_ranking import ranking_metrics


def run(root,output):
    if output.exists():
        raise ValueError('Use a new output directory; existing results are preserved.')
    protocol=json.loads((root/'locked_protocol.json').read_text())
    if sha(root/'fresh_test_manifest.csv')!=protocol['manifest_sha256']:
        raise ValueError('Locked manifest changed.')
    all_summaries,all_rocs,all_prs=[],[],[]
    for name in ARCHITECTURES:
        artifact_path=root/'frozen'/f'{name}.json'
        if sha(artifact_path)!=protocol['artifact_sha256'][name]:
            raise ValueError('Frozen artifact changed.')
        artifact=json.loads(artifact_path.read_text())
        if not (root/name/'evaluation/summary.csv').is_file():
            raise ValueError('Wait for validated fresh evaluation to complete.')
        df=pd.read_csv(root/name/'images.csv')
        if set(df.source)!={'test'} or set(df.corruption)!={'blur'} or set(df.classifier)!={name}:
            raise ValueError('Unexpected fresh cohort input.')
        expected=pd.read_csv(root/name/'selection_audit.csv')
        ids=set(expected.loc[expected.clean_correct==1,'image_id'])
        manifest=pd.read_csv(root/'fresh_test_manifest.csv')
        if set(expected.image_id)!=set(manifest.image_id) or set(df.image_id)!=ids or len(df)!=12*len(ids):
            raise ValueError('Fresh collection is incomplete or mismatched.')
        for key,value in artifact['settings'].items():
            if not np.allclose(df[key],value):
                raise ValueError('Frozen settings mismatch.')
        data,excluded=add_features(df)
        summary,predictions,roc,pr=ranking_metrics(data,artifact)
        folder=output/name
        folder.mkdir(parents=True)
        for label,frame in [('summary',summary),('predictions',predictions),('roc_curve',roc),('pr_curve',pr),('excluded_trajectories',excluded)]:
            frame.to_csv(folder/f'{label}.csv',index=False)
        all_summaries.append(summary); all_rocs.append(roc); all_prs.append(pr)
        print(summary[['classifier','model','n_steps','n_next_step_failures','auroc','aupr']].to_string(index=False))
    combined=pd.concat(all_summaries,ignore_index=True)
    combined.to_csv(output/'summary.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(task='next-severity first failure among currently correct steps at severities 1-4',
        positive='first failure at current severity + 1',score='unchanged frozen warning scores; no thresholds applied for ranking',
        population='same complete valid trajectories as primary evaluation; ranking excludes failures at severity 1 because no preceding monitored step exists',
        aupr_definition='average_precision_score (AP), not trapezoidal integration',
        no_test_fitting=True,no_threshold_changes=True,status='post-hoc descriptive secondary metrics',
        caveat='severity steps within an image are correlated; no independence-based significance claim'),indent=2))
    plot_curves(combined,pd.concat(all_rocs),pd.concat(all_prs),output)


def plot_curves(summary,roc,pr,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['MSP','ODIN','Confidence + trend','GRADCAM attribution + trends','IG+GRADCAM attribution/trends']
    short=['MSP','ODIN','Confidence trend','Attribution-only Grad-CAM','Attribution-only fusion']
    fig,axes=plt.subplots(3,2,figsize=(12,13))
    for row,name in enumerate(ARCHITECTURES):
        for method,label in zip(methods,short):
            r=roc[(roc.classifier==name)&(roc.model==method)]
            p=pr[(pr.classifier==name)&(pr.model==method)]
            subset=summary[(summary.classifier==name)&(summary.model==method)]
            if subset.empty:
                continue
            s=subset.iloc[0]
            if len(r):axes[row,0].plot(r.fpr,r.tpr,label=f'{label} ({s.auroc:.3f})')
            if len(p):axes[row,1].step(p.recall,p.precision,where='post',label=f'{label} (AP {s.aupr:.3f})')
        prevalence=summary[summary.classifier==name].failure_prevalence.iloc[0]
        axes[row,0].plot([0,1],[0,1],'k--',alpha=.4)
        axes[row,1].axhline(prevalence,color='k',linestyle='--',alpha=.4,label='Failure prevalence')
        for col in range(2):
            axes[row,col].set(xlim=(0,1),ylim=(0,1.02),title=f'{name}: '+('ROC' if col==0 else 'Precision–recall'))
            axes[row,col].legend(fontsize=8,loc='best'); axes[row,col].grid(alpha=.2)
        axes[row,0].set(xlabel='False-positive rate (next-step examples)',ylabel='True-positive rate')
        axes[row,1].set(xlabel='Recall',ylabel='Precision')
    fig.suptitle('Frozen fresh-image scores: secondary next-step failure ranking',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(output/'roc_pr_curves.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/archive/historical_hybrids/fresh_trend_validation'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/archive/historical_hybrids/fresh_trend_ranking'))
    args=parser.parse_args()
    run(args.root,args.output_dir)


if __name__=='__main__':main()
