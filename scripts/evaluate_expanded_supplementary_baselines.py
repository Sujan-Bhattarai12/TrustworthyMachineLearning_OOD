"""Forward-only supplementary baselines; never changes the primary study."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from src.supplementary_scores import supplementary_scores
from src.classifiers import load_published_classifier
from src.cifar10c_loader import CIFAR10CLoader
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import TRAJECTORY, calibrate_threshold, trajectory_outcomes
from scripts.run_expanded_cifar10c_study import CLASSIFIERS, sha, save, check


def setup(root):
    out = root / 'supplementary_entropy_energy'
    path = out / 'protocol.json'
    if path.exists():
        raise ValueError('Supplementary protocol already exists; do not overwrite it.')
    save(path, dict(primary_protocol_sha256=sha(root/'locked_protocol.json'),
        repair_of='supplementary_entropy_energy/protocol.json', execution='Direct loader tensor, preserving original collector strides/memory layout; single-image torch.no_grad; unchanged prediction/MSP tolerances', added_unix=time.time(), status='supplementary addition after primary study started',
        methods={'Entropy': '-sum softmax(z)_k * ln softmax(z)_k',
                 'Energy': '-T * logsumexp(z/T)'}, energy_temperature=1.0,
        entropy_temperature=1.0, target_calibration_fpr=0.05,
        calibration='Original frozen calibration image groups; maximum over severity 1–5 of nonfailing trajectories; strict exceedance.',
        cohort='Same common-valid attribution cohort as primary evaluation; no test tuning.',
        references={'Entropy': 'https://doi.org/10.1002/j.1538-7305.1948.tb01338.x',
                    'Energy': 'https://arxiv.org/abs/2010.03759'},
        code_sha256={p:sha(p) for p in ['src/supplementary_scores.py','scripts/evaluate_expanded_supplementary_baselines.py']}))


def collect(root, out, classifier, phase, model, device):
    # Summary marks completed collection, so partially written source CSVs are never consumed.
    source = root/phase/classifier
    if not (source/'summary.csv').exists():
        raise ValueError(f'Primary collection not complete: {source}')
    data, _ = add_features(pd.read_csv(source/'images.csv'))
    cache = out/classifier/f'{phase}_scores.csv'
    provenance = cache.with_suffix('.json')
    expected = dict(source_sha256=sha(source/'images.csv'), rows=len(data))
    if cache.exists() and provenance.exists():
        if json.loads(provenance.read_text()) != expected:
            raise ValueError('Cached supplementary cohort changed.')
        scores = pd.read_csv(cache)
    else:
        loader = CIFAR10CLoader(root/('development_manifest.csv' if phase=='development' else 'fresh_test_manifest.csv'))
        records = []
        with torch.no_grad():
            for start in range(0, len(data), 1):
                batch = data.iloc[start:start+1]
                r = next(batch.itertuples())
                images = loader.get_corrupted_image(int(r.image_id), r.source, r.corruption, int(r.severity), device=device)
                logits = model(images)
                values = supplementary_scores(logits)
                probabilities = logits.softmax(1)
                for i, r in enumerate(batch.itertuples()):
                    if int(logits[i].argmax()) != int(r.predicted_class) or not np.isclose(float(probabilities[i].max()), r.msp, atol=1e-4, rtol=1e-4):
                        raise ValueError(f'Prediction mismatch {classifier} {phase} image={r.image_id} corruption={r.corruption} severity={r.severity}: expected label={r.predicted_class} msp={r.msp}; observed label={int(logits[i].argmax())} msp={float(probabilities[i].max())}')
                    record = {k:getattr(r,k) for k in TRAJECTORY+['severity']}
                    record.update({name:float(value[i]) for name,value in values.items()})
                    record.update({f'logit_{k}':float(v) for k,v in enumerate(logits[i])})
                    records.append(record)
        scores = pd.DataFrame(records)
        cache.parent.mkdir(parents=True,exist_ok=True)
        scores.to_csv(cache,index=False)
        save(provenance,expected)
    merged = data.merge(scores,on=TRAJECTORY+['severity'],validate='one_to_one')
    if len(merged)!=len(data) or not np.isfinite(merged[['Entropy','Energy']]).all().all():
        raise ValueError('Incomplete supplementary score coverage.')
    return merged


def run(root):
    out = root/'supplementary_entropy_energy'
    protocol = json.loads((out/'protocol.json').read_text())
    check(root)
    if json.loads((root/'status.json').read_text()).get('status')!='complete':
        raise ValueError('Expanded primary run must finish before supplementary evaluation.')
    if protocol['primary_protocol_sha256'] != sha(root/'locked_protocol.json'):
        raise ValueError('Primary protocol changed.')
    for path,digest in protocol['code_sha256'].items():
        if sha(path)!=digest: raise ValueError('Supplementary code changed.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for classifier in CLASSIFIERS:
        frozen = root/'frozen'/classifier
        artifact = json.loads((frozen/'specifications.json').read_text())
        lock = json.loads((frozen/'lock.json').read_text())
        if sha(frozen/'specifications.json') != lock['specifications_sha256']:
            raise ValueError('Primary frozen specification changed.')
        model = load_published_classifier(Path('checkpoints/huggingface')/classifier,classifier,device)
        development = collect(root,out,classifier,'development',model,device)
        cal = development[development.image_group.isin(artifact['calibration_image_groups'])]
        specs = []
        for name in ['Entropy','Energy']:
            maxima = cal[cal.first_failure.isna() & (cal.severity>0)].groupby(TRAJECTORY)[name].max().to_numpy()
            if len(maxima)==0: raise ValueError('No nonfailing calibration trajectories.')
            threshold = float(calibrate_threshold(maxima,.05))
            specs.append(dict(model=name,threshold=threshold,calibration_negatives=len(maxima),calibration_fpr=float(np.mean(maxima>threshold))))
        specification = dict(methods=specs,primary_specifications_sha256=sha(frozen/'specifications.json'),supplementary_protocol_sha256=sha(out/'protocol.json'))
        path = out/classifier/'frozen_thresholds.json'
        if path.exists():
            if json.loads(path.read_text())!=specification: raise ValueError('Supplementary frozen thresholds changed.')
        else: save(path,specification)
        # Thresholds are saved and checked before any supplementary test scoring.
        data = collect(root,out,classifier,'test',model,device)
        results=[]; trajectories=[]
        for spec in specs:
            name=spec['model']; scores=data[name].to_numpy()
            trajectory=trajectory_outcomes(data,scores,spec['threshold'])
            trajectory['model']=name;trajectory['classifier']=classifier;trajectories.append(trajectory)
            failing=trajectory[trajectory.first_failure.notna()];nonfail=trajectory[trajectory.first_failure.isna()]
            mask=data.eligible_fit; y=data.loc[mask,'fails_next_step']; s=data.loc[mask,name]
            results.append(dict(classifier=classifier,model=name,n_trajectories=len(trajectory),n_failures=len(failing),n_nonfailures=len(nonfail),
                early_warning_rate=failing.early_warning.mean(),missed_warning_rate=failing.missed_warning.mean(),
                mean_lead_all_failures=failing.credited_lead_steps.mean(),false_alarm_rate=nonfail.false_alarm.mean(),
                auroc=roc_auc_score(y,s) if y.nunique()==2 else np.nan,
                aupr=average_precision_score(y,s) if y.nunique()==2 else np.nan,reporting_scope='supplementary'))
        result=pd.DataFrame(results);result.to_csv(out/classifier/'comparison.csv',index=False)
        pd.concat(trajectories).to_csv(out/classifier/'trajectories.csv',index=False)
        primary=pd.read_csv(root/'evaluation'/classifier/'comparison.csv').assign(reporting_scope='primary protocol')
        pd.concat([primary,result],ignore_index=True).to_csv(out/classifier/'comparison_with_primary.csv',index=False)
        print(f'Completed supplementary baselines: {classifier}',flush=True)
        del model
        if device=='cuda': torch.cuda.empty_cache()
    save(out/'status.json',dict(status='complete',finished_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','run'])
    parser.add_argument('--root',type=Path,default=Path('results/expanded_cifar10c_5000dev_2000test'))
    args=parser.parse_args()
    (setup if args.action=='setup' else run)(args.root)
