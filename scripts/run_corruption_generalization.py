"""Fixed ResNet-50 multi-corruption panel; no detector fitting or threshold tuning."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from src.attribution_features import add_features
from src.cifar10c_loader import CIFAR10CLoader
from src.classifiers import load_published_classifier
from src.supplementary_scores import supplementary_scores
from scripts.run_expanded_cifar10c_study import sha,save,check,score_model
from scripts.evaluate_early_warnings import TRAJECTORY,trajectory_outcomes

PRIMARY=Path('results/expanded_cifar10c_5000dev_2000test')
ROOT=Path('results/additional_corruption_smoke')
CORRUPTIONS=['shot_noise','motion_blur','fog','jpeg_compression']


def setup():
    if ROOT.exists():raise ValueError('Panel folder already exists.')
    ROOT.mkdir(parents=True)
    source=pd.read_csv(PRIMARY/'fresh_test_manifest.csv');rng=np.random.default_rng(20261117)
    selected=pd.concat([g.iloc[rng.permutation(len(g))[:20]] for _,g in source.groupby('class_id')]).sort_values('image_id')
    selected.to_csv(ROOT/'manifest.csv',index=False)
    files=[PRIMARY/'locked_protocol.json',PRIMARY/'frozen/resnet50/warning_models.joblib',PRIMARY/'frozen/resnet50/specifications.json',
        PRIMARY/'supplementary_entropy_energy/resnet50/frozen_thresholds.json',PRIMARY/'individual_lgd_ablations/resnet50/models.joblib',
        PRIMARY/'individual_lgd_ablations/resnet50/frozen_specifications.json']
    save(ROOT/'protocol.json',dict(scope='Exploratory frozen-model generalization across additional official corruptions; previously examined Gaussian blur reference',
        classifier='resnet50',candidate_images=200,sampling='20/class random subset of expanded2000 test candidates, seed20261117; no outcome selection',
        new_corruptions=CORRUPTIONS,reference='gaussian_blur from existing expanded test CSV',ig_steps=64,severities=list(range(6)),
        fitting=False,threshold_tuning=False,report_all=True,manifest_sha256=sha(ROOT/'manifest.csv'),
        frozen_files_sha256={str(p):sha(p) for p in files},
        official_files_sha256={c:sha(Path('data/CIFAR-10-C')/f'{c}.npy') for c in CORRUPTIONS},
        code_sha256={'scripts/run_corruption_generalization.py':sha('scripts/run_corruption_generalization.py')},
        cohort_policy='Common valid maps for every method within corruption; report valid counts per corruption, plus a cross-corruption common-image comparison'))


def summarize(data,scores,threshold):
    outcomes=trajectory_outcomes(data,scores,threshold)
    fail=outcomes[outcomes.first_failure.notna()];nonfail=outcomes[outcomes.first_failure.isna()]
    mask=data.eligible_fit.to_numpy();y=data.loc[mask,'fails_next_step'];s=scores[mask]
    return dict(n_trajectories=len(outcomes),n_failures=len(fail),n_nonfailures=len(nonfail),n_early=int(fail.early_warning.sum()),n_false=int(nonfail.false_alarm.sum()),
        early_warning_rate=fail.early_warning.mean(),false_alarm_rate=nonfail.false_alarm.mean(),mean_lead_all_failures=fail.credited_lead_steps.mean(),
        auroc=roc_auc_score(y,s) if y.nunique()==2 else np.nan,aupr=average_precision_score(y,s) if y.nunique()==2 else np.nan)


def run():
    check(PRIMARY);protocol=json.loads((ROOT/'protocol.json').read_text())
    for file,digest in {**protocol['frozen_files_sha256'],**protocol['code_sha256'],str(ROOT/'manifest.csv'):protocol['manifest_sha256']}.items():
        if sha(file)!=digest:raise ValueError(f'Locked panel input changed: {file}')
    for c,h in protocol['official_files_sha256'].items():
        if sha(Path('data/CIFAR-10-C')/f'{c}.npy')!=h:raise ValueError('Official corruption changed.')
    save(ROOT/'status.json',dict(status='running',started_unix=time.time()))
    out=ROOT/'collection'
    if not (out/'summary.csv').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.collect_attribution_metrics','--dataset','cifar10c','--classifier','resnet50',
            '--checkpoint','checkpoints/huggingface/resnet50','--manifest',str(ROOT/'manifest.csv'),'--split','test','--all-clean-correct',
            '--corruptions',*CORRUPTIONS,'--ig-steps','64','--save-maps','--resume','--output-dir',str(out)],check=True)
    fresh=pd.read_csv(out/'images.csv')
    selected=set(pd.read_csv(ROOT/'manifest.csv').image_id)
    original=pd.read_csv(PRIMARY/'test/resnet50/images.csv');original=original[original.image_id.isin(selected)]
    data,excluded=add_features(pd.concat([fresh,original],ignore_index=True));data=data.reset_index(drop=True)
    excluded.to_csv(ROOT/'excluded.csv',index=False)
    # Additional output-only baselines: forward images directly, preserving collector tensor layout.
    loader=CIFAR10CLoader(ROOT/'manifest.csv');device='cuda' if torch.cuda.is_available() else 'cpu'
    net=load_published_classifier(Path('checkpoints/huggingface/resnet50'),'resnet50',device)
    entropy=[];energy=[]
    with torch.no_grad():
        for r in data.itertuples():
            image=loader.get_corrupted_image(int(r.image_id),r.source,r.corruption,int(r.severity),device=device)
            logits=net(image);p=logits.softmax(1)
            if int(p.argmax())!=r.predicted_class or not np.isclose(float(p.max()),r.msp,atol=1e-4,rtol=1e-4):
                raise ValueError('Panel baseline forward consistency failed.')
            scores=supplementary_scores(logits)
            entropy.append(float(scores['Entropy'][0]));energy.append(float(scores['Energy'][0]))
    data['Entropy']=entropy;data['Energy']=energy
    folder=PRIMARY/'frozen/resnet50';models=joblib.load(folder/'warning_models.joblib');specs=json.loads((folder/'specifications.json').read_text())['methods']
    single=PRIMARY/'individual_lgd_ablations/resnet50';single_models=joblib.load(single/'models.joblib');single_specs=json.loads((single/'frozen_specifications.json').read_text())['methods']
    output_specs=json.loads((PRIMARY/'supplementary_entropy_energy/resnet50/frozen_thresholds.json').read_text())['methods']
    sets=[set(g.image_id) for _,g in data.groupby('corruption')];common=set.intersection(*sets)
    rows=[]
    for policy in ['per-corruption valid','cross-corruption common images']:
        frame=data if policy=='per-corruption valid' else data[data.image_id.isin(common)]
        for corruption,g in frame.groupby('corruption'):
            for spec in specs:
                rows.append(dict(cohort_policy=policy,corruption=corruption,model=spec['name'],**summarize(g,score_model(spec,models,g),spec['threshold'])))
            for spec in single_specs:
                scores=single_models[spec['name']].predict_proba(g[[spec['column']]])[:,1]
                rows.append(dict(cohort_policy=policy,corruption=corruption,model=spec['name'],**summarize(g,scores,spec['threshold'])))
            for spec in output_specs:
                rows.append(dict(cohort_policy=policy,corruption=corruption,model=spec['model'],**summarize(g,g[spec['model']].to_numpy(),spec['threshold'])))
    result=pd.DataFrame(rows);result.to_csv(ROOT/'comparison.csv',index=False)
    save(ROOT/'cohort.json',dict(candidate_images=200,cross_corruption_common_images=len(common),valid_per_corruption={c:int(g.image_id.nunique()) for c,g in data.groupby('corruption')}))
    save(ROOT/'status.json',dict(status='complete',finished_unix=time.time()))
    print(result[(result.cohort_policy=='per-corruption valid')&result.model.isin(['IG dynamic','GRADCAM dynamic','MSP','ODIN','Entropy','Energy'])].to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['setup','run']);args=parser.parse_args()
    setup() if args.action=='setup' else run()
