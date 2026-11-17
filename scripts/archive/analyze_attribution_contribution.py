"""HISTORICAL HYBRID ANALYSIS ONLY: paired bootstrap of completed OOF outcomes.

These confidence+attribution comparisons do not establish standalone attribution superiority.

No model fitting, score selection, threshold tuning or final-test data access.
Confidence intervals condition on existing fitted folds and calibrated thresholds;
they do not account for refitting variability or correct multiple comparisons.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=['source','image_id','corruption','fold']
CONTROL='MSP + trend learned'


from src.paired_analysis import grouped_effects


def run(root,output,repetitions):
    if output.exists():
        raise ValueError('Use a new analysis directory; existing results will not be overwritten.')
    output.mkdir(parents=True)
    records=[]
    for classifier in ['resnet50','vit_b_16','swin_t']:
        for folder in sorted((root/classifier).glob('fpr_*')):
            frame=pd.read_csv(folder/'trajectories.csv')
            if set(frame.source)!={'train'} or set(frame.corruption)!={'gaussian_noise','brightness'}:
                raise ValueError('Only existing development trajectories are accepted.')
            target=float(folder.name.split('_',1)[1])
            for method in ['IG','GRADCAM']:
                comparisons=[('attribution_vs_confidence_trends',f'{method} separated + trends learned',CONTROL),
                             ('combined_vs_confidence_trends',f'MSP + {method} separated/trends learned',CONTROL),
                             ('combined_vs_attribution',f'MSP + {method} separated/trends learned',f'{method} separated + trends learned')]
                for comparison,candidate,control in comparisons:
                    for corruption in ['all','gaussian_noise','brightness']:
                        cohort=frame if corruption=='all' else frame[frame.corruption==corruption]
                        result=grouped_effects(cohort[cohort.model==candidate],cohort[cohort.model==control],repetitions)
                        records.append(dict(classifier=classifier,target_fpr=target,corruption=corruption,
                                            attribution_generator=method,comparison=comparison,candidate=candidate,control=control,**result))
    results=pd.DataFrame(records)
    results.to_csv(output/'paired_effects.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(status='exploratory development analysis after original test was viewed',
        unit='image-group bootstrap; both corruptions resampled together',repetitions=repetitions,seed=42,
        ci='percentile 95% paired differences; positive warning/lead difference favors candidate; positive false-alarm difference is worse',
        caveat='conditional on existing fitted folds/thresholds; no refitting and no multiple-comparison correction',
        fairness='compare actual false-alarm differences and intervals; nominal targets alone do not ensure matched false alarms'),indent=2))
    lines=['# Does attribution add value beyond confidence trends?','',
           'Exploratory development-only analysis. Both corruption trajectories of each image are resampled together (2,000 paired image-cluster bootstrap draws). Intervals condition on the existing fitted models and thresholds and do not correct for trying multiple candidates.','',
           'The table below compares confidence + attribution/trends with confidence + confidence trend at the 5% calibration target. Positive warning-rate/lead differences favor the combined score; positive false-alarm differences are worse.','',
           '| Classifier | Maps | Early-warning difference (95% CI), percentage points | Mean lead difference (95% CI), severity steps | False-alarm difference (95% CI), percentage points |',
           '|---|---|---:|---:|---:|']
    selected=results[(results.comparison=='combined_vs_confidence_trends')&(results.corruption=='all')&(results.target_fpr==.05)]
    for _,r in selected.iterrows():
        fmt=lambda name,mult: f"{mult*r[name+'_difference']:+.2f} [{mult*r[name+'_ci_low']:+.2f}, {mult*r[name+'_ci_high']:+.2f}]"
        lines.append(f"| {r.classifier} | {r.attribution_generator} | {fmt('early_warning_rate',100)} | {fmt('mean_lead',1)} | {fmt('false_alarm_rate',100)} |")
    lines+=['','Full results include attribution-only vs confidence trends, combined vs attribution-only, separate corruption types, and 1%, 5%, 10% targets in `paired_effects.csv`. These comparisons identify development candidates; they do not establish untouched-test improvement.','',
            'Do not interpret an interval above zero for warning lead as an equal-false-alarm win when the candidate also increases false alarms. Scaling to more data is a separate decision; no new image/map collection or reserved-test evaluation is performed by this script.']
    (output/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(selected[['classifier','attribution_generator','early_warning_rate_difference','early_warning_rate_ci_low','early_warning_rate_ci_high','mean_lead_difference','false_alarm_rate_difference']].to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/archive/historical_hybrids/exploratory_warning_improvements'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/archive/historical_hybrids/attribution_contribution_analysis'))
    parser.add_argument('--bootstrap-repetitions',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap_repetitions<100:
        parser.error('Use at least 100 bootstrap draws.')
    run(args.input_dir,args.output_dir,args.bootstrap_repetitions)


if __name__=='__main__':
    main()
