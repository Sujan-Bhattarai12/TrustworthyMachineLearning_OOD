"""Consolidate existing frozen evaluations; no fitting, calibration or collection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.attribution_features import add_features
from scripts.archive.legacy_entrypoints.evaluate_ranking import ranking_metrics
from scripts.freeze_warning_methods import predict
from scripts.evaluate_early_warnings import trajectory_outcomes

NAMES={'resnet50':'ResNet-50','vit_b_16':'ViT-B/16','swin_t':'Swin-T'}
COLORS={'Attribution':'#0072B2','MSP':'#E69F00','ODIN':'#009E73'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(frame):
    cols=list(frame.columns)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def run():
    out=Path('results/reporting/week4')
    if out.exists():raise ValueError('Report outputs exist; preserve or explicitly remove generated outputs before rebuilding.')
    out.mkdir(parents=True)
    sources=[];timing=[];rankings=[];cohorts=[];specs=[];rocs=[];prs=[]
    for classifier in NAMES:
        root=Path('results/final_blur_evaluation')/classifier
        ap=root/'evaluation/frozen_artifact_used.json'
        record=json.loads(ap.read_text())
        artifact=record['artifact']
        assert record['sha256']==sha(Path('results/frozen_warning_methods')/f'{classifier}.json')
        assert artifact==json.loads((Path('results/frozen_warning_methods')/f'{classifier}.json').read_text())
        dp=Path(artifact['development_csv']);assert sha(dp)==artifact['development_sha256']
        raw=pd.read_csv(root/'images.csv');assert set(raw.source)=={'test'} and set(raw.corruption)=={'blur'}
        data,excluded=add_features(raw)
        saved=pd.read_csv(root/'evaluation/summary.csv')
        for spec in artifact['methods']:
            if spec['kind'] not in ['msp','odin']:
                assert all(c.startswith(('ig_','gradcam_')) for c in spec['columns'])
            label=spec['name'] if spec['kind'] in ['msp','odin'] else 'Attribution'
            row=saved[saved.model==spec['name']].iloc[0].to_dict()
            computed=trajectory_outcomes(data,predict(spec,data),spec['threshold'])
            assert len(computed)==row['n_trajectories']
            assert computed.early_warning.sum()==row['n_early_warnings']
            assert np.isclose(computed.loc[computed.first_failure.isna(),'false_alarm'].mean(),row['false_alarm_rate'])
            timing.append(dict(row,display_model=label))
            specs.append(dict(classifier=classifier,model=spec['name'],kind=spec['kind'],threshold=spec['threshold'],
                columns=','.join(spec.get('columns',[])),scales=','.join(map(str,spec.get('scale',[]))),
                calibration_negatives=spec['calibration_negatives'],calibration_fpr=spec['calibration_fpr']))
        rank,_,roc,pr=ranking_metrics(data,artifact)
        rankings.append(rank);rocs.append(roc);prs.append(pr)
        audit=pd.read_csv(root/'selection_audit.csv')
        cohorts.append(dict(classifier=classifier,candidates=len(audit),clean_correct=raw.image_id.nunique(),
            valid_trajectories=data.image_group.nunique(),excluded_trajectories=len(excluded),
            failures=int(saved.iloc[0].n_failures),nonfailures=int(saved.iloc[0].n_nonfailures),
            fit_images=len(artifact['fit_image_groups']),calibration_images=len(artifact['calibration_image_groups'])))
        for p in [ap,dp,root/'images.csv',root/'selection_audit.csv',root/'evaluation/summary.csv']:
            sources.append(dict(path=str(p),sha256=sha(p)))
    timing=pd.DataFrame(timing);ranking=pd.concat(rankings,ignore_index=True);cohorts=pd.DataFrame(cohorts);specs=pd.DataFrame(specs)
    for name,frame in [('timing',timing),('ranking',ranking),('cohorts',cohorts),('frozen_specifications',specs)]:frame.to_csv(out/f'{name}.csv',index=False)
    pd.concat(rocs).to_csv(out/'roc_curves.csv',index=False);pd.concat(prs).to_csv(out/'pr_curves.csv',index=False)
    (out/'sources.json').write_text(json.dumps(dict(scope='report consolidation; no fitting or threshold changes; secondary ranking computed retrospectively on existing frozen scores',sources=sources),indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(metric,title,mult) in zip(axes,[('early_warning_rate','Early warnings (% of failures)',100),('false_alarm_rate','False alarms (% of nonfailures)',100),('mean_lead_all_failures','Mean lead (missed warnings = 0)',1)]):
        for j,label in enumerate(COLORS):
            vals=[timing[(timing.classifier==c)&(timing.display_model==label)][metric].iloc[0]*mult for c in NAMES]
            bars=ax.bar(np.arange(3)+(j-1)*.24,vals,.23,label=label,color=COLORS[label])
            ax.bar_label(bars,fmt='%.1f' if mult==100 else '%.2f',fontsize=8,padding=3)
        ax.set_xticks(range(3),NAMES.values());ax.set_title(title,fontsize=11);ax.margins(y=.22)
        if metric=='false_alarm_rate':ax.axhline(5,color='gray',linestyle='--',linewidth=1,label='Development target: 5%')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.suptitle('Completed frozen blur evaluation — cohorts differ across classifiers')
    fig.tight_layout();fig.savefig(out/'timing_and_false_alarms.png',dpi=180);fig.savefig(out/'timing_and_false_alarms.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for j,c in enumerate(NAMES):
        art=json.loads((Path('results/frozen_warning_methods')/f'{c}.json').read_text())
        for spec in art['methods']:
            label=spec['name'] if spec['kind'] in ['msp','odin'] else 'Attribution'
            roc=pd.concat(rocs);roc=roc[(roc.classifier==c)&(roc.model==spec['name'])]
            pr=pd.concat(prs);pr=pr[(pr.classifier==c)&(pr.model==spec['name'])]
            r=ranking[(ranking.classifier==c)&(ranking.model==spec['name'])].iloc[0]
            axes[0,j].plot(roc.fpr,roc.tpr,color=COLORS[label],label=f'{label}: {r.auroc:.3f}')
            axes[1,j].plot(pr.recall,pr.precision,color=COLORS[label],label=f'{label}: {r.aupr:.3f}')
        axes[0,j].plot([0,1],[0,1],':',color='gray');axes[0,j].set_title(NAMES[c]);axes[0,j].set_xlabel('False-positive rate');axes[0,j].set_ylabel('True-positive rate')
        prevalence=ranking[ranking.classifier==c].failure_prevalence.iloc[0]
        axes[1,j].axhline(prevalence,linestyle=':',color='gray',label=f'Prevalence: {prevalence:.3f}')
        axes[1,j].set_xlabel('Recall');axes[1,j].set_ylabel('Precision')
        for ax in axes[:,j]:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(fontsize=8)
    fig.suptitle('Secondary next-step ranking — legend shows AUROC (top), AP (bottom)')
    fig.tight_layout();fig.savefig(out/'roc_pr.png',dpi=180);fig.savefig(out/'roc_pr.pdf');plt.close(fig)
    rows=[]
    for _,r in timing.iterrows():
        rank=ranking[(ranking.classifier==r.classifier)&(ranking.model==r.model)].iloc[0]
        rows.append({'Classifier':NAMES[r.classifier],'Score':r.model,'Early / failures':f'{r.n_early_warnings} / {r.n_failures}',
            'Early %':f'{100*r.early_warning_rate:.1f}','False alarm %':f'{100*r.false_alarm_rate:.1f}',
            'Mean lead':f'{r.mean_lead_all_failures:.3f}','AUROC':f'{rank.auroc:.3f}','AP':f'{rank.aupr:.3f}'})
    report='''# Week 4: completed warning-method evaluation

## Main finding

The completed frozen blur evaluation does not demonstrate consistent attribution superiority over MSP. The selected attribution score detects fewer failures early than MSP on all three classifiers. Development calibration targeting 5% trajectory false alarms does not reliably transfer to blur. These are completed evaluation outcomes, not new experiments or a newly selected winner.

## What was actually evaluated

This primary table uses only the original saved frozen attribution-only specifications and their exact blur cohorts. ResNet and ViT use fixed Grad-CAM L+D; Swin uses raw Grad-CAM L. These are not the later learned trend, fusion, interaction, DRO or V-REx scores. A logistic warning framework exists, but it would be inaccurate to describe these particular final results as logistic models.

Development fitting/scaling and threshold calibration used disjoint training-source images under noise and brightness. Attribution combinations were selected from completed development summaries by credited lead subject to a development false-alarm budget. Selection details and coefficients/scales are retained in the frozen artifacts. Classifiers were not retrained. MSP and ODIN were calibrated separately with the same policy. The original frozen suite did not include confidence trends; its absence here does not imply it was evaluated or defeated.

## Cohorts

'''+table(cohorts.rename(columns={'classifier':'Classifier'}))+'''

Cohorts are shared across methods within each classifier, but differ across classifiers. Clean-incorrect candidates and invalid attribution trajectories are excluded; results are conditional on this selection. Failure means at least one incorrect prediction by blur severity 5, even if the classifier later recovers.

## Primary timing and secondary ranking results

'''+table(pd.DataFrame(rows))+'''

Early warnings occur strictly before first failure. Mean lead is measured in severity steps and includes missed failures as zero. False-alarm denominators are nonfailing trajectories, not all images. AUROC and AP use still-correct, nonzero-severity observations with an observable next step (levels 1–4 before first failure); they assess a different task from trajectory timing. First-level failures cannot receive a preceding nonzero-level warning. AP is average precision, not trapezoidal PR area. Secondary ranking was computed retrospectively from frozen scores without fitting or calibration.

![Timing and false alarms](../results/reporting/week4/timing_and_false_alarms.png)

The dashed line is a development calibration target, not guaranteed test control. Higher early-warning counts must be interpreted alongside achieved false alarms. No test thresholds were adjusted to force equal false-alarm rates.

![ROC and precision–recall](../results/reporting/week4/roc_pr.png)

PR prevalence and eligible observation counts differ across classifiers. Do not compare AP across cohorts as if the tasks were identical. Curves and bars are descriptive point estimates without confidence intervals; they do not establish statistical significance.

## Supporting evidence, kept separate

- [Feature diagnosis](../results/archive/exploratory/attribution_feature_diagnosis/FINDINGS.md): L/G/D contain heterogeneous signals depending on classifier, corruption and comparison population.
- [Interaction ablation](../results/archive/exploratory/resnet_attribution_interactions_v2/FINDINGS.md): Grad-CAM with L/G/D trends and interactions improved mixed-corruption ResNet development detection (161/515 versus 137 additive), with 5.57% versus 5.98% false alarms. This is exploratory reused-development evidence.
- [Corruption transfer](../results/archive/exploratory/resnet_corruption_transfer/FINDINGS.md): the interaction gains did not reliably transfer. Noise→brightness detected 4/56 early with 4.50% false alarms; brightness→noise detected 90/459 with 80.49% false alarms.
- DRO/V-REx, historical hybrids and early pilots are [archived](../results/README.md). They are not pooled into the main comparison or used to claim attribution-only superiority.

## Limitations and report conclusion

Blur was originally held out but its outcomes have since been inspected during the broader research process. It is now a previously examined condition, not an untouched corruption for future tuning. Published classifiers may have seen development images during original training; external training exposure is not independently established. Maps require a clean reference and a fixed clean predicted class; trends require ordered observations. Severity steps are not real time, so lead time is not seconds. Small nonfailing groups make false-alarm estimates sensitive to individual images.

The defensible conclusion is that attribution changes carry useful predictive information, while the present warning scores do not establish consistent superiority over confidence or robust false-alarm control under corruption shift. Week 4 evaluation is consolidated; report discussion and failure analysis remain. Method expansion remains closed.

## Reproducibility

[CSV tables, curves, figures and source hashes](../results/reporting/week4/) are generated by `scripts/build_week4_report.py`. It verifies frozen artifacts against those used for evaluation, development hashes and reconstructed warning counts before rendering. No source result file is overwritten, and no collection, fitting or threshold optimization is performed. Generated output directory must be absent to rebuild; preserve it before regenerating.
'''
    specification_table=specs[['classifier','model','threshold','calibration_negatives','calibration_fpr']].copy()
    specification_table['threshold']=specification_table.threshold.map(lambda x:f'{x:.6f}')
    specification_table['calibration_fpr']=specification_table.calibration_fpr.map(lambda x:f'{100*x:.2f}%')
    report=report.replace('## Cohorts', '## Frozen thresholds\n\n'+table(specification_table)+'\n\nFixed L+D averages L and D after dividing by fitting-only 95th-percentile scales; raw L uses no such averaging. These scales are normalization choices, not probability thresholds. Exact scales and input columns are in `frozen_specifications.csv`.\n\n## Cohorts')
    eligibility=ranking.drop_duplicates('classifier')[['classifier','n_images','n_steps','n_next_step_failures','failure_prevalence']].copy()
    eligibility['failure_prevalence']=eligibility.failure_prevalence.map(lambda x:f'{100*x:.2f}%')
    report=report.replace('## Supporting evidence, kept separate', '## Secondary ranking denominators\n\n'+table(eligibility)+'\n\n## Supporting evidence, kept separate')
    Path('docs/WEEK4_EVALUATION.md').write_text(report)
    print('Created docs/WEEK4_EVALUATION.md and',out)

if __name__=='__main__':run()
