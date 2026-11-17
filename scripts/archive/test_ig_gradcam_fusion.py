"""Exploratory ResNet development test: learn IG+Grad-CAM feature fusion.

Uses saved development maps' features, not pixelwise heatmap averaging.
Same nested image splits and calibration as prior exploratory experiments.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.warning_features import validate_attribution_columns
from src.attribution_features import add_features
from scripts.archive.test_warning_improvements import candidates
from scripts.evaluate_early_warnings import evaluate,COMPONENTS
from src.paired_analysis import grouped_effects


def fusion_scores(fit,calibration,test):
    scores=candidates(fit,calibration,test)
    training=fit[fit.eligible_fit]
    if training.fails_next_step.nunique()!=2:
        return scores
    static=[f'{m}_{c}' for m in ['ig','gradcam'] for c in ['loss','gain','wasserstein','mass_loss','low_retention']]
    trends=[f'delta_{m}_{c}' for m in ['ig','gradcam'] for c in COMPONENTS]
    features={'IG+GradCAM fusion static':static,
              'IG+GradCAM fusion + trends':static+trends}
    for name,cols in features.items():
        validate_attribution_columns(cols)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
        model.fit(training[cols],training.fails_next_step)
        scores[name]=(model.predict_proba(calibration[cols])[:,1],model.predict_proba(test[cols])[:,1])
    return scores


def run(input_path,output):
    if output.exists():raise ValueError('Use a new output folder.')
    df=pd.read_csv(input_path)
    if set(df.source)!={'train'} or set(df.corruption)!={'gaussian_noise','brightness'}:
        raise ValueError('Development noise/brightness only.')
    data,excluded=add_features(df)
    summary,trajectories,thresholds,splits,paired=evaluate(data,target_fpr=.05,extra_scores=fusion_scores)
    comparisons=[]
    for candidate in ['IG+GradCAM fusion static','IG+GradCAM fusion + trends']:
        for control in ['MSP + trend learned','IG separated + trends learned','GRADCAM separated + trends learned','MSP','ODIN']:
            effects=grouped_effects(trajectories[trajectories.model==candidate],trajectories[trajectories.model==control])
            comparisons.append(dict(candidate=candidate,control=control,**effects))
    output.mkdir(parents=True)
    for label,frame in [('summary',summary),('trajectories',trajectories),('thresholds',thresholds),('splits',splits),
                        ('excluded_trajectories',excluded),('baseline_paired_comparisons',paired),('fusion_paired_effects',pd.DataFrame(comparisons))]:
        frame.to_csv(output/f'{label}.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(status='exploratory development sample experiment',
        input=str(input_path),fusion='concatenate IG and Grad-CAM L/G/W/retention features and trends; learn logistic weights',
        model='StandardScaler + LogisticRegression C=1; fit only on inner fitting images',
        target_fpr=.05,split='same outer five image folds and inner fit/calibration groups as prior experiments',
        no_test_data=True,no_frozen_changes=True,warning='actual false-alarm rates must be compared; no multiple-comparison correction'),indent=2))
    names=['MSP','ODIN','MSP + trend learned','IG separated + trends learned','GRADCAM separated + trends learned',
           'IG+GradCAM fusion static','IG+GradCAM fusion + trends']
    print(summary[(summary.corruption=='all')&summary.model.isin(names)][['model','n_failures','n_early_warnings','mean_lead_all_failures','false_alarm_rate']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/classifier_benchmarks_500/resnet50/images.csv'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/archive/exploratory/resnet_attribution_fusion_v2'))
    args=parser.parse_args()
    run(args.input,args.output_dir)
