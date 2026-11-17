"""Exploratory development-only feature experiments after final test was viewed.

Uses saved development CSVs only. Does not modify frozen scores or read blur/test
results. Consecutive-severity features require sequential observations, unlike
clean-reference-only features. Differences are per severity step, not per second.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.warning_features import validate_attribution_columns
from scripts.evaluate_early_warnings import prepare, evaluate, KEYS, TRAJECTORY, COMPONENTS


from src.attribution_features import add_features


def candidates(fit,calibration,test):
    scores={}
    training=fit[fit.eligible_fit]
    def raw(name,col):
        scores[name]=(calibration[col].to_numpy(),test[col].to_numpy())
    def learned(name,cols):
        if not name.startswith('MSP'):
            validate_attribution_columns(cols)
        if training.fails_next_step.nunique()!=2:
            return
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
        pipe.fit(training[cols],training.fails_next_step)
        scores[name]=(pipe.predict_proba(calibration[cols])[:,1],pipe.predict_proba(test[cols])[:,1])
    raw('MSP trend raw','delta_unconfidence')
    learned('MSP learned',['unconfidence'])
    learned('MSP + trend learned',['unconfidence','delta_unconfidence'])
    for method in ['ig','gradcam']:
        prefix=method.upper()
        original=[f'{method}_{c}' for c in COMPONENTS]
        separated=[f'{method}_{c}' for c in ['loss','gain','wasserstein','mass_loss','low_retention']]
        trends=[f'delta_{method}_{c}' for c in COMPONENTS]
        raw(f'{prefix} unweighted W raw',f'{method}_wasserstein')
        raw(f'{prefix} mass loss raw',f'{method}_mass_loss')
        for label,col in zip('LGD',trends):
            raw(f'{prefix} delta {label} raw',col)
        learned(f'{prefix} L/G/W/retention learned',separated)
        learned(f'{prefix} L/G/D + trends learned',original+trends)
        learned(f'{prefix} separated + trends learned',separated+trends)

    return scores


def run(root,output,targets):
    if output.exists():
        raise ValueError('Use a new output directory to preserve experimental results.')
    output.mkdir(parents=True)
    all_summary=[]
    for name in ['resnet50','vit_b_16','swin_t']:
        path=root/name/'images.csv'
        df=pd.read_csv(path)
        if set(df.source)!={'train'} or set(df.corruption)!={'gaussian_noise','brightness'}:
            raise ValueError('Exploratory runner accepts development train-source noise/brightness only.')
        data,excluded=add_features(df)
        for target in targets:
            summary,trajectories,thresholds,splits,paired=evaluate(data,target_fpr=target,extra_scores=candidates)
            folder=output/name/f'fpr_{target:g}'
            folder.mkdir(parents=True)
            for label,frame in [('summary',summary),('trajectories',trajectories),('thresholds',thresholds),
                                ('splits',splits),('paired_comparisons',paired),('excluded_trajectories',excluded)]:
                frame.assign(classifier=name).to_csv(folder/f'{label}.csv',index=False)
            all_summary.append(summary.assign(classifier=name))
            pooled=summary[summary.corruption=='all']
            print(name,'target',target,flush=True)
            print(pooled.sort_values('mean_lead_all_failures',ascending=False).head(6)[
                ['model','early_warning_rate','mean_lead_all_failures','false_alarm_rate']].to_string(index=False),flush=True)
    pd.concat(all_summary,ignore_index=True).to_csv(output/'all_summary.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(status='exploratory; designed after final blur results were viewed',
        data='development noise/brightness only; no reserved blur data read',
        targets=targets,split='same nested image-grouped fit/calibration/evaluation as original timing evaluator',
        fit='next-step first-failure labels; StandardScaler and logistic regression fit only on fitting images',
        sequence_assumption='delta scores use only current and previous severity; not future observations',
        distance_caveat='unweighted W retains historical zero convention for vanished maps; low-retention proxy is separate',
        fairness='report achieved evaluation FPR; shared nominal targets do not guarantee matched evaluation FPR',
        excluded='whole shared trajectory excluded for invalid component/additional features'),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-root',type=Path,default=Path('results/classifier_benchmarks_500'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/exploratory_attribution_only_v2'))
    parser.add_argument('--targets',type=float,nargs='+',default=[.01,.05,.10])
    args=parser.parse_args()
    if any(not 0<=target<1 for target in args.targets):
        parser.error('False-alarm targets must be in [0,1).')
    run(args.development_root,args.output_dir,args.targets)


if __name__=='__main__':
    main()
