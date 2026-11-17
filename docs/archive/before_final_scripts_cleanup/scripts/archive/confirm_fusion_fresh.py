"""ResNet-50-only frozen IG+Grad-CAM fusion confirmation on a new image cohort.

Blur is previously examined. No new corruption-generalization claim; no test
fitting/calibration. Primary comparison is attribution-only fusion versus confidence trends.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.archive.fresh_trend_validation import fit_model,reserve_fresh,sha,evaluate_fresh,CONTROL
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import COMPONENTS,TRAJECTORY,calibrate_threshold
from scripts.freeze_warning_methods import predict
from scripts.archive.legacy_entrypoints.evaluate_ranking import ranking_metrics

NAME='resnet50'
FUSION='IG+GRADCAM attribution/trends'
GC='GRADCAM attribution + trends'
IG='IG attribution + trends'


def setup(root):
    if root.exists():raise ValueError('Use a new root; old experiments are preserved.')
    root.mkdir(parents=True)
    path=Path('results/classifier_benchmarks_500/resnet50/images.csv')
    df=pd.read_csv(path)
    if set(df.source)!={'train'} or set(df.corruption)!={'gaussian_noise','brightness'}:
        raise ValueError('Development-only fitting input required.')
    data,excluded=add_features(df)
    fit_ids,cal_ids=train_test_split(sorted(data.image_group.unique()),test_size=.25,random_state=42)
    training=data[data.image_group.isin(fit_ids)&data.eligible_fit]
    cal=data[data.image_group.isin(cal_ids)]
    confidence=['unconfidence','delta_unconfidence']
    static={method:[f'{method}_{metric}' for metric in ['loss','gain','wasserstein','mass_loss','low_retention']] for method in ['ig','gradcam']}
    trends={method:[f'delta_{method}_{metric}' for metric in COMPONENTS] for method in ['ig','gradcam']}
    methods=[dict(name='MSP',kind='msp'),dict(name='ODIN',kind='odin'),fit_model(CONTROL,confidence,training),
             fit_model(GC,static['gradcam']+trends['gradcam'],training),
             fit_model(IG,static['ig']+trends['ig'],training),
             fit_model(FUSION,static['ig']+static['gradcam']+trends['ig']+trends['gradcam'],training)]
    for spec in methods:
        scored=cal.assign(score=predict(spec,cal))
        maxima=scored[scored.first_failure.isna()&(scored.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
        threshold=calibrate_threshold(maxima,.05)
        if not np.isfinite(threshold):raise ValueError('Insufficient calibration negatives.')
        spec.update(threshold=threshold,calibration_negatives=len(maxima),calibration_fpr=float(np.mean(maxima>threshold)))
    settings={key:float(df[key].iloc[0]) for key in ['ig_steps','odin_temperature','odin_epsilon']}
    if any(df[key].nunique()!=1 for key in settings):raise ValueError('Inconsistent collection settings.')
    artifact=dict(classifier=NAME,methods=methods,settings=settings,fit_image_groups=fit_ids,
        calibration_image_groups=cal_ids,development_path=str(path),development_sha256=sha(path),
        excluded_development_trajectories=len(excluded),
        provenance=json.loads((Path('checkpoints/huggingface')/NAME/'provenance.json').read_text()),
        research_scope="attribution_only_vs_confidence",primary_candidate=FUSION,primary_control=CONTROL,
        feature_definition='attribution only: concatenate both generators L/G/W/mass-loss/low-retention and L/G/D deltas',
        score='fit-only StandardScaler + LogisticRegression C=1',target_fpr=.05,
        note='Frozen before new outcomes; fresh-image blur confirmation after prior blur results were viewed.')
    (root/'frozen').mkdir()
    artifact_path=root/'frozen/resnet50.json'
    artifact_path.write_text(json.dumps(artifact,indent=2,allow_nan=False))
    manifest=reserve_fresh(root,count=500,seed=20261103)
    protocol=dict(research_scope="attribution_only_vs_confidence",primary_candidate=FUSION,primary_control=CONTROL,
        comparison_controls=[GC,IG,CONTROL,'MSP','ODIN'],
        manifest_sha256=sha(manifest),artifact_sha256={NAME:sha(artifact_path)},
        sample_size=500,seed=20261103,target_calibration_fpr=.05,
        no_test_refitting=True,no_test_threshold_retuning=True,
        interpretation='paired timing and achieved false alarms; secondary next-step AUROC and AP; no post-result score selection',
        caveat='fresh images under previously examined blur; external checkpoint exposure not independently established',
        uncertainty='2000 paired image-bootstrap draws conditional on frozen models; no multiplicity adjustment')
    (root/'locked_protocol.json').write_text(json.dumps(protocol,indent=2))
    print('ResNet fusion frozen; 500 fresh candidates reserved:',manifest,flush=True)


def run(root):
    protocol=json.loads((root/'locked_protocol.json').read_text())
    if protocol.get('research_scope')!='attribution_only_vs_confidence':
        raise ValueError('Historical hybrid protocol cannot be launched as attribution-only.')
    artifact_path=root/'frozen/resnet50.json'
    if sha(root/'fresh_test_manifest.csv')!=protocol['manifest_sha256'] or sha(artifact_path)!=protocol['artifact_sha256'][NAME]:
        raise ValueError('Locked protocol changed.')
    artifact=json.loads(artifact_path.read_text())
    provenance=json.loads((Path('checkpoints/huggingface')/NAME/'provenance.json').read_text())
    if artifact['provenance']!=provenance or artifact['settings']['ig_steps'] not in [1,2,4,8,16,32,50,64,128]:
        raise ValueError('Checkpoint or settings changed.')
    out=root/NAME
    if out.exists():raise ValueError('Refusing to overwrite a confirmation collection.')
    subprocess.run([sys.executable,'-u','-m','scripts.collect_attribution_metrics','--classifier',NAME,
        '--checkpoint',str(Path('checkpoints/huggingface')/NAME),'--manifest',str(root/'fresh_test_manifest.csv'),
        '--split','test','--corruptions','blur','--all-clean-correct','--output-dir',str(out),
        '--odin-temperature',str(artifact['settings']['odin_temperature']),
        '--odin-epsilon',str(artifact['settings']['odin_epsilon'])]+(['--legacy-ig-steps-50'] if artifact['settings']['ig_steps']==50 else ['--ig-steps',str(int(artifact['settings']['ig_steps']))]),check=True)
    evaluate_fresh(root,NAME)
    data,_=add_features(pd.read_csv(out/'images.csv'))
    summary,predictions,roc,pr=ranking_metrics(data,artifact)
    folder=out/'evaluation'/'secondary_ranking'
    folder.mkdir()
    for label,frame in [('summary',summary),('predictions',predictions),('roc_curve',roc),('pr_curve',pr)]:
        frame.to_csv(folder/f'{label}.csv',index=False)
    print('Fusion confirmation and secondary ranking complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','run'])
    parser.add_argument('--root',type=Path,default=Path('results/fresh_attribution_fusion_v2'))
    args=parser.parse_args()
    if args.action=='setup':setup(args.root)
    else:run(args.root)
