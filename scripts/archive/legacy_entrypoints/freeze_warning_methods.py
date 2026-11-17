"""Select on completed DEVELOPMENT summaries and freeze scores/thresholds as JSON.

No reserved images or blur scores are read. Selection: greatest all-failure
credited lead among attribution candidates with development-CV false alarms
<= 5%; ties prefer early-warning rate, then model name. If none qualifies,
stop rather than relax the budget after observing test results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.warning_features import validate_attribution_columns
from scripts.evaluate_early_warnings import prepare, calibrate_threshold, COMPONENTS, TRAJECTORY


def predict(spec, data):
    if spec.get('role') == 'attribution':
        validate_attribution_columns(spec['columns'])
    if spec['kind'] == 'msp':
        return 1-data.msp.to_numpy()
    if spec['kind'] == 'odin':
        return data.odin.to_numpy()
    x=data[spec['columns']].to_numpy()
    if spec['kind'] == 'raw':
        return x[:,0]
    if spec['kind'] == 'fixed':
        return (x/np.asarray(spec['scale'])).mean(axis=1)
    z=(x-np.asarray(spec['mean']))/np.asarray(spec['scale'])
    logit=z@np.asarray(spec['weights'])+spec['intercept']
    return 1/(1+np.exp(-np.clip(logit,-700,700)))


def fit_spec(name, fit):
    method, labels, kind=name.split()
    cols=[f'{method.lower()}_{dict(L="loss",G="gain",D="weighted_wasserstein")[c]}' for c in labels.split('+')]
    validate_attribution_columns(cols)
    spec=dict(name=name,kind=kind,role='attribution',columns=cols)
    training=fit[fit.eligible_fit]
    if training.empty:
        raise ValueError('No fitting examples.')
    if kind=='fixed':
        scale=training[cols].quantile(.95).to_numpy()
        spec['scale']=np.where(scale>1e-12,scale,1.).tolist()
    elif kind=='learned':
        if training.fails_next_step.nunique()!=2:
            raise ValueError('Cannot fit selected score with one training label class.')
        pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
        pipe.fit(training[cols],training.fails_next_step)
        scaler,lr=pipe.steps[0][1],pipe.steps[1][1]
        spec.update(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                    weights=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]))
    return spec


def freeze(root, output):
    output.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for architecture in ['resnet50','vit_b_16','swin_t']:
        destination=output/f'{architecture}.json'
        if destination.exists():
            raise ValueError(f'Refusing to overwrite frozen artifact: {destination}')
        folder=root/architecture
        table=pd.read_csv(folder/'comparison/early_warning/summary.csv')
        choices=table[(table.corruption=='all')&(table.model.str.startswith(('IG ', 'GRADCAM ')))&(table.false_alarm_rate<=.05)].copy()
        if choices.empty:
            raise ValueError(f'{architecture}: no attribution score meets the development false-alarm budget.')
        selected=choices.sort_values(['mean_lead_all_failures','early_warning_rate','model'],ascending=[False,False,True]).iloc[0]
        csv_path=folder/'images.csv'
        df=pd.read_csv(csv_path)
        if not set(df.source)=={'train'} or set(df.corruption)!={'gaussian_noise','brightness'}:
            raise ValueError('Freeze input must be development training-source images and development corruptions only.')
        data,excluded=prepare(df)
        fit_ids,cal_ids=train_test_split(sorted(data.image_group.unique()),test_size=.25,random_state=42)
        fit=data[data.image_group.isin(fit_ids)]
        calibration=data[data.image_group.isin(cal_ids)]
        methods=[dict(name='MSP',kind='msp'),dict(name='ODIN',kind='odin'),fit_spec(selected.model,fit)]
        for spec in methods:
            cal=calibration.assign(score=predict(spec,calibration))
            maxima=cal[cal.first_failure.isna()&(cal.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
            threshold=calibrate_threshold(maxima,.05)
            if not np.isfinite(threshold):
                raise ValueError('No nonfailing calibration trajectories.')
            spec.update(threshold=float(threshold),calibration_negatives=len(maxima),
                        calibration_fpr=float(np.mean(maxima>threshold)))
        settings={}
        for key in ['ig_steps','odin_temperature','odin_epsilon']:
            if df[key].nunique()!=1:
                raise ValueError(f'Inconsistent {key}')
            settings[key]=float(df[key].iloc[0])
        provenance=json.loads((Path('checkpoints/huggingface')/architecture/'provenance.json').read_text())
        artifact=dict(classifier=architecture,methods=methods,settings=settings,provenance=provenance,
                      development_csv=str(csv_path),development_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                      fit_image_groups=fit_ids,calibration_image_groups=cal_ids,
                      selection_policy=__doc__,selection_development_metrics=selected.to_dict(),
                      target_fpr=.05,alarm_rule='score > threshold',monitor_severities=[1,2,3,4,5],
                      test_split='test',test_corruption='blur',excluded_development_trajectories=len(excluded))
        # Only write after fitting and calibration succeed for this classifier.
        destination.write_text(json.dumps(artifact,indent=2,allow_nan=False))
        artifacts.append(artifact)
        print(architecture,'frozen:',selected.model,flush=True)
    return artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-root',type=Path,default=Path('results/classifier_benchmarks_500'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/frozen_warning_methods'))
    args=parser.parse_args()
    freeze(args.development_root,args.output_dir)


if __name__=='__main__':
    main()
