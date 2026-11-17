"""Freeze revised scores and evaluate once on fresh CIFAR-10 images under blur.

Primary: attribution-only Grad-CAM features/trends vs confidence trends.
IG variants are prespecified secondary comparisons. Blur is previously examined;
this confirms fresh-image generalization, not unseen-corruption generalization.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.warning_features import validate_attribution_columns,is_attribution_method
from src.paired_analysis import grouped_effects
from scripts.evaluate_early_warnings import calibrate_threshold,trajectory_outcomes,TRAJECTORY,COMPONENTS
from scripts.freeze_warning_methods import predict
from src.attribution_features import add_features

ARCHITECTURES=['resnet50','vit_b_16','swin_t']
CONTROL='Confidence + trend'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_model(name,columns,training):
    if is_attribution_method(name):
        validate_attribution_columns(columns)
    if training.fails_next_step.nunique()!=2:
        raise ValueError('Need both future failure labels in fitting data.')
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
    pipe.fit(training[columns],training.fails_next_step)
    scaler,lr=pipe.steps[0][1],pipe.steps[1][1]
    return dict(name=name,kind='learned',role='attribution' if is_attribution_method(name) else 'baseline',columns=columns,mean=scaler.mean_.tolist(),
                scale=scaler.scale_.tolist(),weights=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]))


def freeze_all(root,development):
    folder=root/'frozen'
    folder.mkdir()
    for architecture in ARCHITECTURES:
        path=development/architecture/'images.csv'
        df=pd.read_csv(path)
        if set(df.source)!={'train'} or set(df.corruption)!={'gaussian_noise','brightness'}:
            raise ValueError('Fitting/calibration must use development data only.')
        data,excluded=add_features(df)
        fit_ids,cal_ids=train_test_split(sorted(data.image_group.unique()),test_size=.25,random_state=42)
        fit=data[data.image_group.isin(fit_ids)]
        training=fit[fit.eligible_fit]
        calibration=data[data.image_group.isin(cal_ids)]
        confidence=['unconfidence','delta_unconfidence']
        methods=[dict(name='MSP',kind='msp'),dict(name='ODIN',kind='odin'),fit_model(CONTROL,confidence,training)]
        for method in ['ig','gradcam']:
            separate=[f'{method}_{c}' for c in ['loss','gain','wasserstein','mass_loss','low_retention']]
            trends=[f'delta_{method}_{c}' for c in COMPONENTS]
            methods.append(fit_model(f'{method.upper()} attribution + trends',separate+trends,training))

        for spec in methods:
            scored=calibration.assign(score=predict(spec,calibration))
            negatives=scored[scored.first_failure.isna()&(scored.severity>0)]
            maxima=negatives.groupby(TRAJECTORY).score.max().to_numpy()
            threshold=calibrate_threshold(maxima,.05)
            if not np.isfinite(threshold):
                raise ValueError('No nonfailure calibration trajectories.')
            spec.update(threshold=threshold,calibration_negatives=len(maxima),calibration_fpr=float(np.mean(maxima>threshold)))
        settings={key:float(df[key].iloc[0]) for key in ['ig_steps','odin_temperature','odin_epsilon']}
        if any(df[key].nunique()!=1 for key in settings):
            raise ValueError('Development settings differ within classifier.')
        artifact=dict(classifier=architecture,methods=methods,settings=settings,
            provenance=json.loads((Path('checkpoints/huggingface')/architecture/'provenance.json').read_text()),
            fit_image_groups=fit_ids,calibration_image_groups=cal_ids,
            development_path=str(path),development_sha256=sha(path),excluded_development_trajectories=len(excluded),
            research_scope='attribution_only_vs_confidence',primary='GRADCAM attribution + trends versus Confidence + trend',
            secondary='IG combined and attribution-only scores; MSP and ODIN reference baselines',
            target_fpr=.05,alarm_rule='score > threshold',
            temporal_assumption='current and immediately previous severity only; clean map available',
            note='Revised after original blur evaluation; fresh images under a previously examined corruption.')
        (folder/f'{architecture}.json').write_text(json.dumps(artifact,indent=2,allow_nan=False))
        print('Frozen revised scores:',architecture,flush=True)


def reserve_fresh(root,count=500,seed=20261101):
    excluded=set()
    evidence=[]
    # Reserve against prior manifests even when some candidates were clean-incorrect.
    paths=list(Path('splits').glob('*.csv'))+list(Path('results').rglob('*.csv'))
    for path in sorted(paths):
        if root==path or root in path.parents:
            continue
        try:
            header=pd.read_csv(path,nrows=0).columns
        except pd.errors.EmptyDataError:
            continue
        if 'image_id' not in header:
            continue
        columns=['image_id']+(['source'] if 'source' in header else [])
        frame=pd.read_csv(path,usecols=columns)
        # Source-less older pilots are conservatively treated as possible exposure.
        if 'source' in frame:
            frame=frame[frame.source=='test']
        ids=set(pd.to_numeric(frame.image_id,errors='raise').dropna().astype(int))
        ids={i for i in ids if 0<=i<10000}
        if ids:
            excluded.update(ids)
            evidence.append(dict(path=str(path),sha256=sha(path),n_possible_test_ids=len(ids)))
    with Path('data/cifar-10-batches-py/test_batch').open('rb') as stream:
        labels=np.asarray(pickle.load(stream,encoding='bytes')[b'labels'])
    if count%10 or count<=0:
        raise ValueError('Fresh cohort must be a positive multiple of ten.')
    rng=np.random.default_rng(seed)
    rows=[]
    for label in range(10):
        available=[i for i in np.flatnonzero(labels==label) if int(i) not in excluded]
        if len(available)<count//10:
            raise ValueError('Insufficient unused test candidates.')
        for image in rng.choice(available,count//10,replace=False):
            rows.append(dict(image_id=int(image),class_id=label,source='test',split='test'))
    manifest=pd.DataFrame(rows).sort_values('image_id')
    assert not (set(manifest.image_id)&excluded)
    path=root/'fresh_test_manifest.csv'
    manifest.to_csv(path,index=False)
    audit=dict(seed=seed,n_candidates=len(manifest),per_class=count//10,
               excluded_possible_prior_test_ids=sorted(excluded),prior_evidence=evidence,
               manifest_sha256=sha(path),selection='label-stratified without classifier predictions or warning scores',
               scope='fresh to recorded warning experiments; external classifier training exposure is not established')
    (root/'reservation_audit.json').write_text(json.dumps(audit,indent=2))
    return path


def evaluate_fresh(root,architecture):
    path=root/architecture/'images.csv'
    artifact_path=root/'frozen'/f'{architecture}.json'
    artifact=json.loads(artifact_path.read_text())
    protocol=json.loads((root/'locked_protocol.json').read_text())
    manifest_path=root/'fresh_test_manifest.csv'
    if sha(manifest_path)!=protocol['manifest_sha256'] or sha(artifact_path)!=protocol['artifact_sha256'][architecture]:
        raise ValueError('Frozen protocol files changed.')
    df=pd.read_csv(path)
    if set(df.source)!={'test'} or set(df.corruption)!={'blur'} or set(df.classifier)!={architecture}:
        raise ValueError('Evaluation accepts only the fresh blur cohort for this classifier.')
    manifest=pd.read_csv(manifest_path)
    audit=pd.read_csv(root/architecture/'selection_audit.csv')
    if set(audit.image_id)!=set(manifest.image_id):
        raise ValueError('Selection audit must include every reserved candidate.')
    expected=set(audit.loc[audit.clean_correct==1,'image_id'])
    if set(df.image_id)!=expected:
        raise ValueError('Collection is incomplete or contains nonselected images.')
    if len(df)!=12*len(expected):
        raise ValueError('Incomplete or duplicate blur collection.')
    for key,value in artifact['settings'].items():
        if not np.allclose(df[key],value):
            raise ValueError('Frozen collection settings mismatch.')
    data,excluded=add_features(df)
    outcomes=[]
    for spec in artifact['methods']:
        result=trajectory_outcomes(data,predict(spec,data),spec['threshold'])
        result['model'],result['threshold'],result['fold']=spec['name'],spec['threshold'],0
        outcomes.append(result)
    trajectories=pd.concat(outcomes,ignore_index=True)
    summary=[]
    for model,group in trajectories.groupby('model'):
        failing=group[group.first_failure.notna()]
        negatives=group[group.first_failure.isna()]
        summary.append(dict(classifier=architecture,model=model,n_trajectories=len(group),
            n_failures=len(failing),n_nonfailures=len(negatives),n_early_warnings=int(failing.early_warning.sum()),
            n_missed_warnings=int(failing.missed_warning.sum()),
            early_warning_rate=float(failing.early_warning.mean()) if len(failing) else np.nan,
            mean_lead_all_failures=float(failing.credited_lead_steps.mean()) if len(failing) else np.nan,
            mean_lead_detected_only=float(failing.detected_lead_steps.mean()),
            false_alarm_rate=float(negatives.false_alarm.mean()) if len(negatives) else np.nan))
    pairs=[]
    control=trajectories[trajectories.model==CONTROL]
    for name in [m['name'] for m in artifact['methods'] if 'attribution' in m['name']]:
        candidate=trajectories[trajectories.model==name]
        for baseline in protocol.get('comparison_controls',[CONTROL,'MSP','ODIN']):
            if baseline == name:
                continue
            paired=grouped_effects(candidate,trajectories[trajectories.model==baseline],repetitions=2000)
            pairs.append(dict(candidate=name,control=baseline,primary=name==protocol.get('primary_candidate','GRADCAM attribution + trends') and baseline==protocol.get('primary_control',CONTROL),**paired))
    output=root/architecture/'evaluation'
    if output.exists():
        raise ValueError('Refusing to overwrite fresh evaluation results.')
    output.mkdir()
    for name,frame in [('summary',pd.DataFrame(summary)),('trajectories',trajectories),('paired_effects',pd.DataFrame(pairs)),('excluded_trajectories',excluded)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    (output/'artifact_used.json').write_text(json.dumps(artifact,indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def setup(root,development):
    if root.exists():
        raise ValueError('Use a new root; existing experiments are never overwritten.')
    root.mkdir(parents=True)
    freeze_all(root,development)
    manifest=reserve_fresh(root)
    protocol=dict(research_scope='attribution_only_vs_confidence',primary_candidate='GRADCAM attribution + trends',primary_control=CONTROL,primary='attribution-only Grad-CAM trends versus confidence trends, separately for each classifier',
        secondary='IG variants and attribution-only ablations; MSP and ODIN references',
        sample_size=500,corruption='blur (previously examined); fresh-image confirmation',
        target_calibration_trajectory_fpr=.05,monitor_severities=[1,2,3,4,5],
        no_test_refitting=True,no_test_threshold_adjustment=True,
        manifest_sha256=sha(manifest),artifact_sha256={name:sha(root/'frozen'/f'{name}.json') for name in ARCHITECTURES},
        interpretation='Compare achieved false alarms; improvements at different alarm rates are trade-offs. No further selection after fresh results.',
        uncertainty='2000 paired image bootstrap draws conditional on frozen methods; IG secondary; no multiplicity-adjusted claim across classifiers')
    (root/'locked_protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Fresh cohort locked:',manifest,flush=True)


def collect(root):
    protocol=json.loads((root/'locked_protocol.json').read_text())
    if sha(root/'fresh_test_manifest.csv')!=protocol['manifest_sha256']:
        raise ValueError('Fresh manifest changed.')
    if protocol.get('research_scope')!='attribution_only_vs_confidence':
        raise ValueError('Historical hybrid protocol: do not launch as an attribution-only experiment.')
    for name in ARCHITECTURES:
        artifact_path=root/'frozen'/f'{name}.json'
        if sha(artifact_path)!=protocol['artifact_sha256'][name]:
            raise ValueError('Frozen artifact changed.')
        artifact=json.loads(artifact_path.read_text())
        provenance=json.loads((Path('checkpoints/huggingface')/name/'provenance.json').read_text())
        if artifact['provenance']!=provenance or artifact['settings']['ig_steps'] not in [1,2,4,8,16,32,50,64,128]:
            raise ValueError('Checkpoint/settings changed.')
        output=root/name
        if output.exists():
            raise ValueError('Refusing to overwrite a fresh collection.')
        subprocess.run([sys.executable,'-u','-m','scripts.collect_attribution_metrics','--classifier',name,
            '--checkpoint',str(Path('checkpoints/huggingface')/name),'--manifest',str(root/'fresh_test_manifest.csv'),
            '--split','test','--corruptions','blur','--all-clean-correct','--output-dir',str(output),
            '--odin-temperature',str(artifact['settings']['odin_temperature']),
            '--odin-epsilon',str(artifact['settings']['odin_epsilon'])]+(['--legacy-ig-steps-50'] if artifact['settings']['ig_steps']==50 else ['--ig-steps',str(int(artifact['settings']['ig_steps']))]),check=True)
        evaluate_fresh(root,name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','run'])
    parser.add_argument('--root',type=Path,default=Path('results/fresh_attribution_only_v2'))
    parser.add_argument('--development-root',type=Path,default=Path('results/classifier_benchmarks_500'))
    args=parser.parse_args()
    if args.action=='setup':
        setup(args.root,args.development_root)
    else:
        collect(args.root)


if __name__=='__main__':
    main()
