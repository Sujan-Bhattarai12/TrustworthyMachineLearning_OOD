"""Development figures: attribution summaries versus severity and current errors.

Descriptive same-step association, not next-step warning evaluation. No fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.attribution_features import add_features

NAMES={'resnet50':'ResNet-50','vit_b_16':'ViT-B/16','swin_t':'Swin-T'}
METRICS={'loss':'Attribution loss (L)','gain':'Attribution gain (G)','weighted_wasserstein':'Retention-weighted distance (D)'}

def run(root,output):
    if output.exists():raise ValueError('Choose a new output folder to preserve existing figures.')
    output.mkdir(parents=True)
    summaries=[];associations=[];rates=[];cohorts=[];sources=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for classifier,label in NAMES.items():
        path=root/classifier/'images.csv';raw=pd.read_csv(path)
        if set(raw.source)!={'train'} or set(raw.corruption)!={'gaussian_noise','brightness'}:raise ValueError('Development only.')
        data,excluded=add_features(raw)
        data['error']=(data.predicted_class!=data.true_label).astype(int)
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        cohorts.append(dict(classifier=classifier,raw_images=raw.image_id.nunique(),
            valid_trajectories=len(data[['source','image_id','corruption']].drop_duplicates()),
            excluded_trajectories=len(excluded)))
        for corruption in ['gaussian_noise','brightness']:
            subset=data[data.corruption==corruption]
            for severity,g in subset.groupby('severity'):
                rates.append(dict(classifier=classifier,corruption=corruption,severity=int(severity),
                    n_images=len(g),n_correct=int((g.error==0).sum()),n_incorrect=int(g.error.sum()),error_rate=g.error.mean()))
            fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
            for i,method in enumerate(['ig','gradcam']):
                for j,(metric,title) in enumerate(METRICS.items()):
                    col=f'{method}_{metric}';ax=axes[i,j]
                    for error,color,name in [(0,'#0072B2','Correct'),(1,'#D55E00','Incorrect')]:
                        line=[]
                        for severity in range(6):
                            g=subset[(subset.severity==severity)&(subset.error==error)];v=g[col]
                            row=dict(classifier=classifier,corruption=corruption,method=method,metric=metric,
                                severity=severity,prediction_group=name,n=len(g),median=v.median(),q25=v.quantile(.25),q75=v.quantile(.75))
                            summaries.append(row);line.append(row)
                        g=pd.DataFrame(line)
                        ax.plot(g.severity,g['median'],marker='o',markersize=4,color=color,label=name)
                        ax.fill_between(g.severity.to_numpy(),g.q25.to_numpy(),g.q75.to_numpy(),color=color,alpha=.15)
                    for severity,g in subset.groupby('severity'):
                        valid=g.error.nunique()==2 and g[col].nunique()>1
                        associations.append(dict(classifier=classifier,corruption=corruption,method=method,metric=metric,
                            severity=int(severity),n=len(g),n_errors=int(g.error.sum()),
                            point_biserial_r=float(pointbiserialr(g.error,g[col]).statistic) if valid else np.nan,
                            error_auc=roc_auc_score(g.error,g[col]) if g.error.nunique()==2 else np.nan,
                            status='ok' if valid else 'undefined correlation: one outcome group or constant metric'))
                    ax.set_title(f'{method.upper()}: {title}',fontsize=11)
                    ax.set_xticks(range(6));ax.set_xlabel('Corruption severity');ax.set_ylabel('Attribution metric')
                    if i==0 and j==0:ax.legend()
            fig.suptitle(f'{label} — {corruption.replace("_"," ")}\nCurrent prediction groups; median and interquartile range',fontsize=13)
            counts=[]
            for severity in range(6):
                g=subset[subset.severity==severity];counts.append(f'{severity}: {int((g.error==0).sum())}/{int(g.error.sum())}')
            fig.text(.5,.012,'Severity: correct/incorrect counts   '+ '   |   '.join(counts),ha='center',fontsize=9)
            fig.tight_layout(rect=[0,.045,1,.93])
            stem=output/f'{classifier}_{corruption}'
            fig.savefig(str(stem)+'.png',dpi=200);fig.savefig(str(stem)+'.pdf');plt.close(fig)
    summary=pd.DataFrame(summaries);association=pd.DataFrame(associations);rates=pd.DataFrame(rates)
    for name,frame in [('group_summaries',summary),('within_severity_associations',association),('error_rates_and_counts',rates),('cohorts',pd.DataFrame(cohorts))]:
        frame.to_csv(output/f'{name}.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
    for ax,(classifier,label) in zip(axes,NAMES.items()):
        for corruption,color in [('gaussian_noise','#009E73'),('brightness','#CC79A7')]:
            g=rates[(rates.classifier==classifier)&(rates.corruption==corruption)].sort_values('severity')
            ax.plot(g.severity,g.error_rate*100,marker='o',label=corruption.replace('_',' '),color=color)
        ax.set_title(label);ax.set_xlabel('Severity');ax.set_xticks(range(6));ax.set_ylim(0,100);ax.legend(fontsize=8)
    axes[0].set_ylabel('Current classification error (%)');fig.tight_layout()
    fig.savefig(output/'classification_error_by_severity.png',dpi=200);fig.savefig(output/'classification_error_by_severity.pdf');plt.close(fig)
    aggregate=[]
    for keys,g in association[association.severity>0].groupby(['classifier','corruption','method','metric']):
        valid=g[g.error_auc.notna()];weights=valid.n_errors*(valid.n-valid.n_errors)
        aggregate.append(dict(zip(['classifier','corruption','method','metric'],keys),
            within_severity_error_auc=np.average(valid.error_auc,weights=weights) if weights.sum() else np.nan,
            n_valid_strata=len(valid)))
    agg=pd.DataFrame(aggregate);agg.to_csv(output/'association_overview.csv',index=False)
    pivot=agg.pivot(index=['classifier','corruption','method'],columns='metric',values='within_severity_error_auc').reset_index()
    def table(frame):
        return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    for c in METRICS:pivot[c]=pivot[c].map(lambda x:f'{x:.3f}')
    report='''# Figure 1: attribution changes and current classification errors

Six development figures cover IG and Grad-CAM L/G/D against severity for ResNet-50, ViT-B/16 and Swin-T, separately for noise and brightness. Curves show medians and shaded interquartile ranges for currently correct versus incorrect predictions. Counts below each figure show changes in group composition. Missing groups are left blank, not zero-filled. Clean images are correct by selection.

**Suggested caption:** Attribution-derived loss (L), gain (G) and retention-weighted Wasserstein distance (D) across corruption severity. Rows correspond to IG and Grad-CAM, and colors distinguish current prediction correctness. Lines show group medians; shading shows interquartile ranges rather than uncertainty intervals. Each image is evaluated repeatedly and may change prediction group across severity levels. Differences describe concurrent errors, not proof of an early-warning signal or causal effects.

## Association table

The following AUCs compare incorrect versus correct observations **within the same corruption and severity**, then average across levels 1–5 using positive-negative pair counts as weights. The raw higher-value-means-error direction is fixed. Values below 0.5 can indicate a reverse association, rather than absence of information. These are descriptive, reused-development estimates, without significance tests or classifier fitting. Classifier/corruption cohorts and prevalence differ.

'''+table(pivot)+'''\n
The detailed table provides point-biserial correlations and raw-score AUC for each severity, along with counts. Correlation is undefined for one-group or constant-score strata. AUC can remain defined at 0.5 for constant scores. No p-values are reported because repeated images and exploratory comparisons require an explicit inferential design.

## Important distinctions

- L is attribution loss, not classification cross-entropy. The response is current classification error.
- All maps target the clean predicted class, including after an error. Post-failure/recovered observations remain in these descriptive plots; this differs from the prefailure-only early-warning evaluation.
- Comparisons use common valid IG/Grad-CAM trajectories from the original saved 500-image collections. Empty-clean/invalid-map trajectories are excluded consistently; check cohort and count tables.
- Apparent changes across severity combine corruption effects with changing group membership. Within-severity associations help avoid pooling this severity effect, but do not establish causal independence.
- Classification loss could be reconstructed using stored target confidence because these trajectories start clean-correct, but these figures intentionally use observed correctness, without log-clipping choices.

PNG and PDF versions are provided. `classification_error_by_severity` supplies context for increasing error prevalence. All other tables and source hashes are saved alongside the figures. No test/blur results, new predictions, model fitting or threshold changes are involved.
'''
    for classifier,label in NAMES.items():
        report+=f'\n## {label}\n\n'
        for corruption in ['gaussian_noise','brightness']:
            report+=f'![{label} {corruption}]({classifier}_{corruption}.png)\n\n'
    (output/'README.md').write_text(report)
    (output/'sources.json').write_text(json.dumps(dict(scope='development descriptive same-step error association',sources=sources),indent=2))
    print('Saved figures and association tables:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path('results/classifier_benchmarks_500'))
    p.add_argument('--output-dir',type=Path,default=Path('results/reporting/attribution_error_relationships'))
    args=p.parse_args();run(args.input_root,args.output_dir)
