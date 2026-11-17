"""Descriptive development-only feature diagnosis, without fitting or test access."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.attribution_features import add_features


def diagnose(frame):
    if set(frame.source)!={'train'} or set(frame.corruption)!={'gaussian_noise','brightness'}:
        raise ValueError('Only development noise/brightness observations are allowed.')
    data,excluded=add_features(frame)
    # Common opportunity window: 1..4, still correct; no post-failure observations.
    rows=data[data.eligible_fit].copy()
    rows['outcome_group']=np.where(rows.fails_next_step==1,'fails_next',
        np.where(rows.first_failure.notna(),'fails_later','stays_correct'))
    records=[];separation=[];association=[]
    for method in ['ig','gradcam']:
        for metric in ['loss','gain','weighted_wasserstein']:
            column=f'{method}_{metric}'
            for (corruption,severity,outcome),g in rows.groupby(['corruption','severity','outcome_group']):
                values=g[column]
                records.append(dict(method=method,metric=metric,corruption=corruption,severity=int(severity),
                    outcome_group=outcome,n=len(g),median=values.median(),q25=values.quantile(.25),q75=values.quantile(.75)))
            for corruption,g in rows.groupby('corruption'):
                for comparator in ['stays_correct','fails_later','all_not_next']:
                    pair=g[g.outcome_group.isin(['fails_next',comparator])] if comparator!='all_not_next' else g
                    if pair.fails_next_step.nunique()<2:continue
                    auc=roc_auc_score(pair.fails_next_step,pair[column])
                    strata=[]
                    for severity,h in pair.groupby('severity'):
                        if h.fails_next_step.nunique()!=2:continue
                        positive=int(h.fails_next_step.sum());negative=len(h)-positive
                        strata.append((roc_auc_score(h.fails_next_step,h[column]),positive*negative))
                    separation.append(dict(method=method,metric=metric,corruption=corruption,comparator=comparator,
                        n=len(pair),n_next=int(pair.fails_next_step.sum()),pooled_auc=auc,
                        within_severity_auc=np.average([a for a,w in strata],weights=[w for a,w in strata]) if strata else np.nan,
                        n_severity_strata=len(strata)))
                never=g[g.outcome_group=='stays_correct']
                rho=spearmanr(never.severity,never[column]).statistic if never.severity.nunique()>1 else np.nan
                association.append(dict(method=method,metric=metric,corruption=corruption,
                    n_nonfailing_steps=len(never),severity_spearman_nonfailing=rho))
    return rows,pd.DataFrame(records),pd.DataFrame(separation),pd.DataFrame(association),excluded


def run(root,output):
    if output.exists():raise ValueError('Use a new output directory.')
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_sep=[];all_assoc=[]
    for classifier in ['resnet50','vit_b_16','swin_t']:
        rows,groups,separation,association,excluded=diagnose(pd.read_csv(root/classifier/'images.csv'))
        folder=output/classifier;folder.mkdir()
        for name,df in [('eligible_observations',rows),('group_summaries',groups),('separation',separation),
                        ('severity_association',association),('excluded',excluded)]:df.to_csv(folder/f'{name}.csv',index=False)
        all_sep.append(separation.assign(classifier=classifier));all_assoc.append(association.assign(classifier=classifier))
        colors={'fails_next':'#d62728','fails_later':'#ff7f0e','stays_correct':'#1f77b4'}
        for corruption in ['gaussian_noise','brightness']:
            fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
            for i,method in enumerate(['ig','gradcam']):
                for j,metric in enumerate(['loss','gain','weighted_wasserstein']):
                    ax=axes[i,j]
                    for outcome,color in colors.items():
                        g=groups[(groups.method==method)&(groups.metric==metric)&(groups.corruption==corruption)&(groups.outcome_group==outcome)].sort_values('severity')
                        x=g.severity.to_numpy();ax.plot(x,g['median'].to_numpy(),marker='o',color=color,label=outcome)
                        ax.fill_between(x,g.q25.to_numpy(),g.q75.to_numpy(),color=color,alpha=.12)
                    ax.set_title(f'{method.upper()} {metric}');ax.set_xticks([1,2,3,4]);ax.set_xlabel('Severity (still-correct observations)')
            axes[0,0].legend(fontsize=8)
            fig.suptitle(f'{classifier}: {corruption} — median and interquartile range')
            fig.tight_layout();fig.savefig(folder/f'{corruption}.png',dpi=160);plt.close(fig)
    pd.concat(all_sep).to_csv(output/'separation.csv',index=False)
    pd.concat(all_assoc).to_csv(output/'severity_association.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(scope='descriptive development feature diagnosis; no fitting or test access',
        cohort='common valid, clean-correct trajectories; severity 1..4 strictly before first failure',
        groups='next-step failure; later observed first failure; no observed failure through severity 5',
        D='weighted_wasserstein = retention-weighted Wasserstein distance',
        auc='raw increasing-score orientation fixed; within-severity AUC weights positive-negative pairs within each severity',
        caveats=['Repeated observations are dependent; no inferential significance claims.',
                 'Outcome grouping uses future labels for diagnosis only, never as predictor inputs.',
                 'Still correct through severity 5 does not imply never fails.',
                 'Observed separation does not establish causal effects or unseen-domain generalization.']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path('results/classifier_benchmarks_500'))
    p.add_argument('--output-dir',type=Path,default=Path('results/archive/exploratory/attribution_feature_diagnosis'))
    args=p.parse_args();run(args.input_root,args.output_dir)
