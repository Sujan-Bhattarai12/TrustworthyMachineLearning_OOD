"""Expanded CIFAR-10-C study: 5000 development candidates, 2000 fresh test candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from src.attribution_features import add_features
from src.warning_features import validate_attribution_columns
from src.data_reservation import reserve_fresh
from src.cifar10c_loader import CIFAR10CLoader
from scripts.evaluate_early_warnings import COMPONENTS,TRAJECTORY,calibrate_threshold,trajectory_outcomes

CLASSIFIERS=['resnet50','vit_b_16','swin_t']
SEED=20261114

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(p)

def recipes():
    result=[]
    for method in ['ig','gradcam']:
        base=[f'{method}_{c}' for c in COMPONENTS]
        for variant in ['static','dynamic','interactions']:
            result.append(dict(name=f'{method.upper()} {variant}',role='attribution',primary=variant=='dynamic',
                columns=base+([f'delta_{method}_{c}' for c in COMPONENTS] if variant!='static' else []),interactions=variant=='interactions'))
    result.append(dict(name='Confidence trends',role='baseline',primary=False,columns=['unconfidence','delta_unconfidence'],interactions=False))
    return result

def make_model(interactions):
    steps=[StandardScaler()]
    if interactions:steps.extend([PolynomialFeatures(2,interaction_only=True,include_bias=False),StandardScaler()])
    return make_pipeline(*steps,LogisticRegression(C=1,max_iter=2000,random_state=42))

def fit_models(data):
    ids=sorted(data.image_group.unique())
    fit_ids,cal_ids=train_test_split(ids,test_size=.25,random_state=42)
    training=data[data.image_group.isin(fit_ids)&data.eligible_fit]
    cal=data[data.image_group.isin(cal_ids)]
    if training.fails_next_step.nunique()!=2:raise ValueError('Need both failure labels in fitting data.')
    fitted={};specs=[]
    for recipe in recipes():
        if recipe['role']=='attribution':validate_attribution_columns(recipe['columns'])
        model=make_model(recipe['interactions']).fit(training[recipe['columns']],training.fails_next_step)
        fitted[recipe['name']]=model;specs.append(recipe)
    specs.extend([dict(name='MSP',role='baseline',primary=False),dict(name='ODIN',role='baseline',primary=False)])
    for spec in specs:
        score=score_model(spec,fitted,cal)
        g=cal.assign(score=score)
        maxima=g[g.first_failure.isna()&(g.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
        if len(maxima)==0:raise ValueError('No nonfailing calibration trajectories.')
        threshold=calibrate_threshold(maxima,.05)
        spec.update(threshold=float(threshold),calibration_negatives=len(maxima),calibration_fpr=float(np.mean(maxima>threshold)))
    return fitted,dict(methods=specs,fit_image_groups=fit_ids,calibration_image_groups=cal_ids)

def score_model(spec,models,data):
    if spec['name']=='MSP':return 1-data.msp.to_numpy()
    if spec['name']=='ODIN':return data.odin.to_numpy()
    if spec['role']=='attribution':validate_attribution_columns(spec['columns'])
    return models[spec['name']].predict_proba(data[spec['columns']])[:,1]

def setup(root):
    if root.exists():raise ValueError('Choose a new expanded run root.')
    root.mkdir(parents=True)
    reservation=root/'reservation';reservation.mkdir()
    fresh=pd.read_csv(reserve_fresh(reservation,count=6000,seed=20261116))
    original=pd.read_csv('results/final_cifar10c/development_manifest.csv')
    if len(original)!=1000:raise ValueError('Expected original 1000 development candidates.')
    rng=np.random.default_rng(20261116);new_dev=[];new_test=[]
    for label,group in fresh.groupby('class_id'):
        shuffled=group.iloc[rng.permutation(len(group))].copy()
        new_dev.append(shuffled.iloc[:400].assign(split='development'))
        new_test.append(shuffled.iloc[400:].assign(split='test'))
    development=pd.concat([original,*new_dev]).sort_values('image_id')
    test=pd.concat(new_test).sort_values('image_id')
    if len(development)!=5000 or len(test)!=2000 or set(development.image_id)&set(test.image_id):
        raise ValueError('Invalid expanded cohort sizes/overlap.')
    development.to_csv(root/'development_manifest.csv',index=False)
    test.to_csv(root/'fresh_test_manifest.csv',index=False)
    original_protocol=json.loads(Path('results/final_cifar10c/locked_protocol.json').read_text())
    protocol=dict(original_protocol)
    paths=list(Path('src').glob('*.py'))+list(Path('scripts').glob('*.py'))
    protocol.update(development_images_per_classifier=None,development_candidates_per_classifier=5000,
        test_candidates_per_classifier=2000,
        development_sampling='All clean-correct among 5000 label-stratified candidates (500/class); original 1000 development pool plus4000 unused candidates; actual counts architecture-dependent',
        test_sampling='2000 additional original-test candidates, 200/class, seed20261116, excluded from all locally recorded experiment image IDs; clean-correct subset retained',
        extension='Larger-sample follow-up after observing the original official study; same methods and corruption families, not new-corruption confirmation',
        original_protocol_sha256=sha('results/final_cifar10c/locked_protocol.json'),
        development_manifest_sha256=sha(root/'development_manifest.csv'),test_manifest_sha256=sha(root/'fresh_test_manifest.csv'),
        code_sha256={str(p):sha(p) for p in paths},
        test_interpretation='Additional image IDs fresh to recorded warning experiments; Gaussian blur previously examined; external classifier exposure unverified',
        requested_slurm_hours=48,internal_budget_hours=47.5)
    save(root/'locked_protocol.json',protocol)
    check(root)
    print('Locked expanded protocol:',root/'locked_protocol.json',flush=True)

def check(root):
    p=json.loads((root/'locked_protocol.json').read_text())
    if p.get('dataset')!='CIFAR-10-C':raise ValueError('Official study requires a CIFAR-10-C protocol.')
    if sha(root/'development_manifest.csv')!=p['development_manifest_sha256'] or sha(root/'fresh_test_manifest.csv')!=p['test_manifest_sha256']:raise ValueError('Manifest changed.')
    for c in CLASSIFIERS:
        if sha(Path('checkpoints/huggingface')/c/'provenance.json')!=p['checkpoint_provenance_sha256'][c]:raise ValueError('Checkpoint provenance changed.')
    for file,digest in p['code_sha256'].items():
        if sha(file)!=digest:raise ValueError(f'Locked code changed: {file}')
    for name,digest in p['official_file_sha256'].items():
        if sha(Path('data/CIFAR-10-C')/f'{name}.npy')!=digest:raise ValueError('Official corruption file changed.')
    return p

def collect(root,c,split,deadline,smoke=False):
    out=root/('preflight' if smoke else split)/c
    if (out/'summary.csv').exists():return
    args=[sys.executable,'-u','-m','scripts.collect_attribution_metrics','--classifier',c,'--checkpoint',str(Path('checkpoints/huggingface')/c),
        '--manifest',str(root/('development_manifest.csv' if split=='development' else 'fresh_test_manifest.csv')),
        '--dataset','cifar10c','--split','development' if split=='development' else 'test','--ig-steps','64','--output-dir',str(out),
        '--corruptions',*(['gaussian_noise','brightness'] if split=='development' else ['gaussian_blur']),
        '--deadline',str(deadline),'--resume','--save-maps']
    if split=='test' or not smoke:args.append('--all-clean-correct')
    else:args.extend(['--n-per-class','1' if smoke else '50'])
    if smoke:args.extend(['--max-images','1'])
    out.mkdir(parents=True,exist_ok=True)
    before=time.time()
    with (out/'run.log').open('a') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode==75:raise TimeoutError('Collection paused at image boundary.')
    if result.returncode:raise RuntimeError(f'Collection failed ({result.returncode}); see {out}/run.log')
    if smoke:print(f'Preflight {c}: {time.time()-before:.1f}s including model load/selection.',flush=True)

def freeze(root,c):
    folder=root/'frozen'/c
    if (folder/'lock.json').exists():return
    path=root/'development'/c/'images.csv';raw=pd.read_csv(path)
    if set(raw.source)!={'test'} or set(raw.dataset)!={'cifar10c'} or set(raw.ig_steps)!={64} or set(raw.corruption)!={'gaussian_noise','brightness'}:raise ValueError('Unexpected development settings.')
    audit=pd.read_csv(root/'development'/c/'selection_audit.csv')
    manifest=pd.read_csv(root/'development_manifest.csv')
    selected=set(audit.loc[audit.clean_correct==1,'image_id'])
    if set(audit.image_id)!=set(manifest.image_id) or set(raw.image_id)!=selected or len(raw)!=24*len(selected):
        raise ValueError('Expanded development collection incomplete.')
    data,excluded=add_features(raw)
    models,artifact=fit_models(data)
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(models,folder/'warning_models.joblib')
    # Human-readable fitted weights, intercepts and preprocessing parameters alongside serialized pipelines.
    for spec in artifact['methods']:
        if spec['name'] not in models:continue
        pipe=models[spec['name']]
        spec.update(feature_names=pipe[:-1].get_feature_names_out(spec['columns']).tolist(),weights=pipe[-1].coef_[0].tolist(),intercept=float(pipe[-1].intercept_[0]))
        spec['preprocessing']=[dict(step=name,mean=step.mean_.tolist(),scale=step.scale_.tolist()) for name,step in pipe.steps if isinstance(step,StandardScaler)]
    artifact.update(classifier=c,development_sha256=sha(path),protocol_sha256=sha(root/'locked_protocol.json'),ig_steps=64)
    save(folder/'specifications.json',artifact)
    excluded.to_csv(folder/'excluded_development.csv',index=False)
    save(folder/'lock.json',dict(models_sha256=sha(folder/'warning_models.joblib'),specifications_sha256=sha(folder/'specifications.json')))
    print('Frozen warning models and thresholds:',c,flush=True)

def evaluate(root,c):
    folder=root/'frozen'/c;lock=json.loads((folder/'lock.json').read_text())
    if sha(folder/'warning_models.joblib')!=lock['models_sha256'] or sha(folder/'specifications.json')!=lock['specifications_sha256']:raise ValueError('Frozen artifact changed.')
    models=joblib.load(folder/'warning_models.joblib');artifact=json.loads((folder/'specifications.json').read_text())
    if sha(root/'locked_protocol.json')!=artifact['protocol_sha256']:raise ValueError('Protocol changed.')
    raw=pd.read_csv(root/'test'/c/'images.csv');audit=pd.read_csv(root/'test'/c/'selection_audit.csv');manifest=pd.read_csv(root/'fresh_test_manifest.csv')
    if set(audit.image_id)!=set(manifest.image_id):raise ValueError('Test audit incomplete.')
    selected=set(audit.loc[audit.clean_correct==1,'image_id'])
    if set(raw.image_id)!=selected or len(raw)!=12*len(selected):raise ValueError('Test collection incomplete.')
    if set(raw.source)!={'test'} or set(raw.corruption)!={'gaussian_blur'} or set(raw.dataset)!={'cifar10c'} or set(raw.ig_steps)!={64}:raise ValueError('Unexpected test settings.')
    data,excluded=add_features(raw);summaries=[];trajectories=[];ranking=[];curves=[]
    for spec in artifact['methods']:
        scores=score_model(spec,models,data)
        result=trajectory_outcomes(data,scores,spec['threshold']);result['model']=spec['name'];result['classifier']=c
        trajectories.append(result);failing=result[result.first_failure.notna()];nonfail=result[result.first_failure.isna()]
        summaries.append(dict(classifier=c,model=spec['name'],primary=spec['primary'],n_trajectories=len(result),n_failures=len(failing),n_nonfailures=len(nonfail),
            n_early=int(failing.early_warning.sum()),early_warning_rate=failing.early_warning.mean(),missed_warning_rate=failing.missed_warning.mean(),
            mean_lead_all_failures=failing.credited_lead_steps.mean(),false_alarm_rate=nonfail.false_alarm.mean()))
        mask=data.eligible_fit.to_numpy();y=data.loc[mask,'fails_next_step'];s=scores[mask]
        ranking.append(dict(classifier=c,model=spec['name'],n_steps=len(y),n_next_failures=int(y.sum()),prevalence=y.mean(),
            auroc=roc_auc_score(y,s) if y.nunique()==2 else np.nan,aupr=average_precision_score(y,s) if y.nunique()==2 else np.nan))
        if y.nunique()==2:
            fpr,tpr,_=roc_curve(y,s);precision,recall,_=precision_recall_curve(y,s)
            curves.append(pd.DataFrame(dict(model=spec['name'],kind='roc',x=fpr,y=tpr)))
            curves.append(pd.DataFrame(dict(model=spec['name'],kind='pr',x=recall,y=precision)))
    out=root/'evaluation'/c;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(out/'timing.csv',index=False);pd.DataFrame(ranking).to_csv(out/'ranking.csv',index=False)
    pd.concat(trajectories).to_csv(out/'trajectories.csv',index=False);excluded.to_csv(out/'excluded.csv',index=False)
    if curves:pd.concat(curves).to_csv(out/'curves.csv',index=False)
    save(out/'cohort.json',dict(candidates=len(manifest),clean_correct=len(selected),valid_trajectories=data.image_group.nunique(),excluded_trajectories=len(excluded)))
    # Standalone presentation exports for each completed classifier, including partial budgeted runs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    timing=pd.DataFrame(summaries);rank=pd.DataFrame(ranking)
    display=timing.merge(rank,on=['classifier','model'])
    display.to_csv(out/'comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    names=timing.model.tolist();positions=np.arange(len(names))
    colors=['#0072B2' if value else '#999999' for value in timing.primary]
    for ax,(column,label,mult) in zip(axes,[('early_warning_rate','Early warning (%)',100),('false_alarm_rate','False alarms (%)',100),('mean_lead_all_failures','Mean lead, misses=0',1)]):
        ax.barh(positions,timing[column]*mult,color=colors);ax.set_yticks(positions,names);ax.invert_yaxis();ax.set_xlabel(label)
        if column=='false_alarm_rate':ax.axvline(5,linestyle='--',color='red',linewidth=1)
    fig.suptitle(f'{c}: frozen 64-step fresh-image CIFAR-10-C gaussian-blur confirmation (blue: primary)')
    fig.tight_layout();fig.savefig(out/'timing.png',dpi=170);fig.savefig(out/'timing.pdf');plt.close(fig)
    lines=['# Final rerun: '+c,'','Official CIFAR-10-C subset evaluation; related custom blur was examined previously. Models and thresholds were frozen using development data only.','',
        '| Score | Role | Early / failures | False alarms | AUROC | AP |','|---|---|---:|---:|---:|---:|']
    for _,row in display.iterrows():
        lines.append(f"| {row.model} | {'Primary' if row.primary else 'Baseline/ablation'} | {row.n_early}/{row.n_failures} | {100*row.false_alarm_rate:.2f}% | {row.auroc:.3f} | {row.aupr:.3f} |")
    lines.extend(['','A 5% development calibration target does not guarantee 5% false alarms under official Gaussian blur. All methods share the same valid cohort within this classifier; achieved false alarms differ.','',
        'Static and interaction methods are predeclared ablations; the primary IG and Grad-CAM models both use original L/G/D plus adjacent changes. Do not select a winner using these outcomes.','',
        'See comparison.csv, ranking.csv, trajectories.csv and cohort.json for exact denominators. Frozen specifications and fitted preprocessing/weights are under ../../frozen/'+c+'/.'])
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    save(out/'provenance.json',dict(test_csv_sha256=sha(root/'test'/c/'images.csv'),protocol_sha256=sha(root/'locked_protocol.json'),frozen_lock=lock))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

def run(root,hours):
    protocol=check(root);deadline=time.time()+hours*3600
    save(root/'status.json',dict(status='running',started_unix=time.time(),deadline_unix=deadline))
    try:
        # Prior completed A100 IG64 preflights cover the unchanged checkpoints and attribution settings.
        for c in CLASSIFIERS:
            if time.time()>=deadline:raise TimeoutError('Budget reached before next classifier.')
            print('Starting development:',c,flush=True);collect(root,c,'development',deadline)
            freeze(root,c)
            print('Starting test:',c,flush=True);collect(root,c,'test',deadline)
            evaluate(root,c)
        summaries=[pd.read_csv(root/'evaluation'/c/'timing.csv') for c in CLASSIFIERS]
        pd.concat(summaries).to_csv(root/'timing_all_classifiers.csv',index=False)
        pd.concat([pd.read_csv(root/'evaluation'/c/'ranking.csv') for c in CLASSIFIERS]).to_csv(root/'ranking_all_classifiers.csv',index=False)
        save(root/'status.json',dict(status='complete',finished_unix=time.time()))
    except TimeoutError as e:
        save(root/'status.json',dict(status='budget_paused',message=str(e),paused_unix=time.time()))
        print(str(e),'Resume with the same run command and a new time budget.',flush=True)
    except Exception as e:
        save(root/'status.json',dict(status='failed',message=str(e),failed_unix=time.time()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['setup','run']);p.add_argument('--root',type=Path,default=Path('results/expanded_cifar10c_5000dev_2000test'))
    p.add_argument('--hours',type=float,default=47.5)
    a=p.parse_args()
    if not 0<a.hours<=48:p.error('Budget must be greater than zero and at most48 hours per invocation.')
    setup(a.root) if a.action=='setup' else run(a.root,a.hours)
