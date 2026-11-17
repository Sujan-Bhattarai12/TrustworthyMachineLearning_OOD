"""ResNet development map normalization/smoothing robustness pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from captum.attr import IntegratedGradients
from scripts.collect_attribution_metrics import asws_components,compute_cam,compute_ig
from src.classifiers import load_published_classifier,attribution_layer
from src.data_loader import CIFARDataLoader
from src.corruptions import apply_corruption
from src.attribution_features import add_features
from scripts.archive.test_attribution_interactions import make_model
from scripts.evaluate_early_warnings import evaluate,COMPONENTS

CONDITIONS=['original','independent','smoothed','smoothed_independent']

def components(a,b,condition):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if 'smoothed' in condition:
        a=gaussian_filter(a,sigma=2,mode='reflect');b=gaussian_filter(b,sigma=2,mode='reflect')
    if 'independent' in condition:
        if a.max()<=1e-8 or b.max()<=1e-8:
            return dict(loss=np.nan,gain=np.nan,wasserstein=np.nan,retention=np.nan,weighted_wasserstein=np.nan,valid_cam=False)
        a=a/a.max();b=b/b.max()
    return asws_components(a,b)


def setup(root):
    if root.exists():raise ValueError('Use new directory.')
    df=pd.read_csv('results/classifier_benchmarks_500/resnet50/images.csv')
    if set(df.source)!={'train'} or set(df.corruption)!={'gaussian_noise','brightness'}:raise ValueError('Development only.')
    candidates=df[['image_id','true_label']].drop_duplicates().sort_values(['true_label','image_id'])
    selected=candidates.groupby('true_label',group_keys=False).head(10)
    if len(selected)!=100:raise ValueError('Expected 10 existing clean-correct images per class.')
    root.mkdir(parents=True)
    selected.to_csv(root/'selected_images.csv',index=False)
    (root/'protocol.json').write_text(json.dumps(dict(scope='exploratory ResNet development-only, 100 previously selected images, 10/class',
        conditions=CONDITIONS,smoothing='scipy Gaussian sigma=2 map pixels, reflect boundaries, before normalization',
        normalization='original: shared clean maximum; independent: each map divided by own maximum before original L/G/D implementation',
        retention_caveat='independent normalization changes mass retention into a relative shape-mass ratio; report W and R separately; D is not a pure spatial statistic',
        maps='same clean target; IG50 zero baseline absolute RGB sum; raw positive Grad-CAM; full maps cached',
        comparison='static L/G/D, dynamic L/G/D+trends, dynamic interactions; separate MSP, ODIN and confidence trends; same 5 image folds, 5% source mixed-corruption calibration target',
        no_blur=True,no_test=True),indent=2))


def collect(root):
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:',device,flush=True)
    raw=pd.read_csv('results/classifier_benchmarks_500/resnet50/images.csv')
    selected=pd.read_csv(root/'selected_images.csv')
    loader=CIFARDataLoader();model=load_published_classifier(Path('checkpoints/huggingface/resnet50'),'resnet50',device)
    layer,reshape=attribution_layer(model,'resnet50');ig=IntegratedGradients(model)
    folder=root/'maps';folder.mkdir(exist_ok=True)
    rows=[]
    for n,image in enumerate(selected.image_id,1):
        image=int(image);cache=folder/f'{image}.npz'
        saved=raw[raw.image_id==image]
        clean=loader.get_image(image_id=image,source='train',device=device)
        target=int(saved.clean_class.iloc[0])
        if cache.exists():
            with np.load(cache) as z:maps={k:z[k] for k in z.files}
        else:
            maps={}
            for corruption in ['gaussian_noise','brightness']:
                for severity in range(6):
                    x=apply_corruption(clean,corruption,severity,seed=42+image)
                    with torch.no_grad():
                        pred=int(model(x).argmax(1))
                    expected=int(saved[(saved.corruption==corruption)&(saved.severity==severity)].predicted_class.iloc[0])
                    if pred!=expected:raise ValueError('Classifier prediction differs from saved data.')
                    for method in ['ig','gradcam']:
                        key=f'{corruption}_{severity}_{method}'
                        maps[key]=compute_ig(ig,x,target,steps=50) if method=='ig' else compute_cam(model,x,target,layer,reshape)
            np.savez_compressed(cache,**maps)
        for _,row in saved.iterrows():
            a=maps[f'{row.corruption}_0_{row.method}'];b=maps[f'{row.corruption}_{row.severity}_{row.method}']
            for condition in CONDITIONS:
                score=components(a,b,condition);record=row.to_dict()
                record.update({k:v for k,v in score.items() if k!='valid_cam'})
                record.update(valid_map=score['valid_cam'],condition=condition)
                rows.append(record)
        pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
        print(f'Completed {n}/{len(selected)}: {image}',flush=True)
    analyze(root)


def analyze(root):
    raw=pd.read_csv(root/'metrics.csv')
    if raw.image_id.nunique()!=len(pd.read_csv(root/'selected_images.csv')):raise ValueError('Collection incomplete.')
    prepared={c:add_features(raw[raw.condition==c]) for c in CONDITIONS}
    common_trajectories=set.intersection(*[set(zip(d.image_group,d.corruption)) for d,e in prepared.values()])
    common={image for image,corruption in common_trajectories}
    if not common:raise ValueError('No common valid images.')
    all_summary=[];all_thresholds=[];all_traj=[]
    for condition,(data,excluded) in prepared.items():
        data=data[[pair in common_trajectories for pair in zip(data.image_group,data.corruption)]]
        def scores(fit,cal,test):
            training=fit[fit.eligible_fit];result={}
            for method in ['ig','gradcam']:
                base=[f'{method}_{c}' for c in COMPONENTS]
                for variant in ['static','dynamic','dynamic_interactions']:
                    cols=base+([f'delta_{method}_{c}' for c in COMPONENTS] if variant!='static' else [])
                    model=make_model(variant=='dynamic_interactions').fit(training[cols],training.fails_next_step)
                    result[f'{method.upper()} attribution {variant}']=(model.predict_proba(cal[cols])[:,1],model.predict_proba(test[cols])[:,1])
            cols=['unconfidence','delta_unconfidence'];model=make_model().fit(training[cols],training.fails_next_step)
            result['Confidence trends']=(model.predict_proba(cal[cols])[:,1],model.predict_proba(test[cols])[:,1])
            return result
        summary,traj,thresholds,splits,_=evaluate(data,extra_scores=scores)
        names=['MSP','ODIN','Confidence trends']+[f'{m} attribution {v}' for m in ['IG','GRADCAM'] for v in ['static','dynamic','dynamic_interactions']]
        all_summary.append(summary[summary.model.isin(names)].assign(condition=condition))
        all_traj.append(traj[traj.model.isin(names)].assign(condition=condition))
        all_thresholds.append(thresholds[thresholds.model.isin(names)].assign(condition=condition))
        splits.to_csv(root/f'{condition}_splits.csv',index=False)
    summary=pd.concat(all_summary);summary.to_csv(root/'summary.csv',index=False)
    pd.concat(all_traj).to_csv(root/'trajectories.csv',index=False)
    pd.concat(all_thresholds).to_csv(root/'thresholds.csv',index=False)
    (root/'common_cohort.json').write_text(json.dumps(dict(n_images=len(common),image_groups=sorted(common),n_trajectories=len(common_trajectories),trajectories=sorted(common_trajectories)),indent=2))
    print(summary[summary.corruption=='all'][['condition','model','n_early_warnings','n_failures','false_alarm_rate']].to_string(index=False),flush=True)
    print('Robustness evaluation complete.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['setup','run','analyze'])
    p.add_argument('--root',type=Path,default=Path('results/archive/exploratory/resnet_map_robustness'))
    args=p.parse_args()
    if args.action=='setup':setup(args.root)
    elif args.action=='run':collect(args.root)
    else:analyze(args.root)
