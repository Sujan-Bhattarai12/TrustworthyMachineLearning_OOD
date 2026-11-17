"""Development-only nested 5/10/20-strength sensitivity study.

Same images, corruption endpoints and Gaussian noise direction across grids.
Densest-grid first failure is the shared failure reference. Lead and derivatives
use normalized physical strength c=strength/original_max, not sample counts.
Fixed prediction horizon h=0.2 in c. This is a pilot, not fresh-test confirmation.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers import ARCHITECTURES,load_published_classifier,attribution_layer
from src.warning_features import validate_attribution_columns
from src.data_loader import CIFARDataLoader
from src.corruptions import apply_corruption_strength
from src.benchmark_scores import benchmark_scores
from scripts.collect_attribution_metrics import compute_cam,compute_ig,asws_components
from scripts.evaluate_early_warnings import calibrate_threshold

CORRUPTIONS=['gaussian_noise','brightness']
METRICS=['loss','gain','weighted_wasserstein','wasserstein','retention']
KEYS=['source','image_id','corruption','dense_index']
TRAJECTORY=KEYS[:-1]
HORIZON=.2


def strength_grid(corruption,subdivisions=4):
    anchors={'gaussian_noise':[0,.04,.08,.12,.16,.20],
             'brightness':[0,.10,.175,.25,.325,.40]}[corruption]
    values=[anchors[0]]
    for left,right in zip(anchors,anchors[1:]):
        values.extend(np.linspace(left,right,subdivisions+1)[1:].tolist())
    return np.asarray(values)


def collect(root,n_per_class,architectures=ARCHITECTURES):
    loader=CIFARDataLoader()
    torch.set_num_threads(2)
    for name in architectures:
        folder=root/name
        folder.mkdir(parents=True)
        model=load_published_classifier(Path('checkpoints/huggingface')/name,name,'cpu')
        previous=pd.read_csv(Path('results/classifier_benchmarks_500')/name/'images.csv')
        candidates=previous[['source','image_id','true_label']].drop_duplicates().sort_values(['true_label','image_id'])
        selected=candidates.groupby('true_label',group_keys=False).head(n_per_class)
        if len(selected)!=10*n_per_class or set(selected.source)!={'train'}:
            raise ValueError('Balanced development cohort unavailable.')
        selected.to_csv(folder/'selected_images.csv',index=False)
        (folder/'provenance.json').write_text(json.dumps(model.provenance,indent=2))
        layer,reshape=attribution_layer(model,name)
        ig=IntegratedGradients(model)
        rows=[]
        for number,row in enumerate(selected.to_dict('records'),1):
            image_id=int(row['image_id'])
            clean=loader.get_image(image_id,row['source'])
            with torch.no_grad():target=int(model(clean).argmax(1).item())
            if target!=int(row['true_label']):
                raise ValueError('Previously clean-correct candidate changed prediction.')
            clean_maps=dict(ig=compute_ig(ig,clean,target),gradcam=compute_cam(model,clean,target,layer,reshape))
            seed=42+image_id
            for corruption in CORRUPTIONS:
                grid=strength_grid(corruption)
                for index,strength in enumerate(grid):
                    x=apply_corruption_strength(clean,corruption,float(strength),seed)
                    logits,baseline=benchmark_scores(model,x)
                    probs=logits.softmax(1)[0]
                    maps=clean_maps if index==0 else dict(ig=compute_ig(ig,x,target),gradcam=compute_cam(model,x,target,layer,reshape))
                    for method,array in maps.items():
                        if np.isfinite(clean_maps[method]).all() and np.isfinite(array).all():
                            scores=asws_components(clean_maps[method],array)
                        else:
                            scores=dict(valid_cam=False,**{metric:np.nan for metric in METRICS})
                        rows.append(dict(classifier=name,source=row['source'],image_id=image_id,true_label=int(row['true_label']),
                            clean_class=target,predicted_class=int(probs.argmax()),corruption=corruption,
                            dense_index=index,strength=float(strength),c=float(strength/grid[-1]),method=method,
                            msp=float(probs.max()),odin=float(baseline['odin'][0]),noise_seed=seed,
                            valid_map=scores['valid_cam'],**{m:scores[m] for m in METRICS}))
            pd.DataFrame(rows).to_csv(folder/'dense_scores.csv',index=False)
            print(f'{name}: completed {number}/{len(selected)} images (20 nonzero strengths per corruption)',flush=True)
        del model,ig,layer
        gc.collect()


def prepare_dense(frame):
    if set(frame.source)!={'train'} or set(frame.corruption)!=set(CORRUPTIONS):
        raise ValueError('Only development train-source noise/brightness data accepted.')
    basecols=KEYS+['classifier','true_label','clean_class','predicted_class','strength','c','msp','odin']
    if (frame.groupby(KEYS)[['predicted_class','true_label','c','msp','odin']].nunique()>1).any().any():
        raise ValueError('Prediction metadata differs between map methods.')
    data=frame[basecols].drop_duplicates(KEYS)
    featurecols=[]
    for method in ['ig','gradcam']:
        part=frame[frame.method==method][KEYS+METRICS+['valid_map']].copy()
        if part.duplicated(KEYS).any():raise ValueError('Duplicate map rows.')
        valid=part.valid_map.astype(str).str.lower().isin(['true','1'])
        part.loc[~valid,METRICS]=np.nan
        rename={m:f'{method}_{m}' for m in METRICS}
        featurecols+=list(rename.values())
        data=data.merge(part.drop(columns='valid_map').rename(columns=rename),on=KEYS,how='left',validate='one_to_one')
    kept=[];excluded=[]
    for identity,g in data.groupby(TRAJECTORY):
        g=g.sort_values('dense_index').copy()
        expected=strength_grid(identity[-1])
        if len(g)!=21 or set(g.dense_index)!=set(range(21)) or not np.allclose(g.strength,expected):
            raise ValueError('Dense collection is incomplete or grid changed.')
        if not np.isfinite(g[featurecols+['msp','odin']]).all().all():
            excluded.append(dict(zip(TRAJECTORY,identity),reason='invalid map/score; excluded from every grid/method'))
            continue
        failure=g[g.predicted_class!=g.true_label]
        g['failure_c']=float(failure.c.min()) if len(failure) else np.nan
        g['failure_strength']=float(failure.strength.min()) if len(failure) else np.nan
        g['image_group']=g.source+':'+g.image_id.astype(str)
        kept.append(g)
    if not kept:raise ValueError('No valid complete dense trajectories.')
    return pd.concat(kept,ignore_index=True),pd.DataFrame(excluded)


def grid_features(dense,step):
    data=dense[dense.dense_index%step==0].sort_values(TRAJECTORY+['c']).copy()
    data['unconfidence']=1-data.msp
    differences=data.groupby(TRAJECTORY).c.diff()
    for method in ['ig','gradcam']:
        data[f'{method}_mass_loss']=1-data[f'{method}_retention']
        data[f'{method}_low_retention']=(data[f'{method}_retention']<=1e-6).astype(float)
        for metric in ['loss','gain','weighted_wasserstein']:
            col=f'{method}_{metric}'
            data[f'derivative_{col}']=(data.groupby(TRAJECTORY)[col].diff()/differences).fillna(0)
    data['derivative_unconfidence']=(data.groupby(TRAJECTORY).unconfidence.diff()/differences).fillna(0)
    data['eligible_fit']=(data.c>0)&(data.c<=1-HORIZON+1e-10)&(data.failure_c.isna()|(data.c<data.failure_c-1e-10))
    data['fails_within_horizon']=(data.failure_c.notna()&(data.failure_c>data.c+1e-10)&(data.failure_c<=data.c+HORIZON+1e-10)).astype(int)
    return data


def alarm_values(data,values,persistence):
    if not persistence:return np.asarray(values)
    frame=data[TRAJECTORY+['c']].copy()
    frame['score']=values
    frame=frame.sort_values(TRAJECTORY+['c'])
    previous=frame.groupby(TRAJECTORY).score.shift()
    previous_c=frame.groupby(TRAJECTORY).c.shift()
    transformed=np.minimum(frame.score,previous).where(previous_c>0,-np.inf)
    return transformed.reindex(data.index).to_numpy()


def outcomes(data,values,threshold):
    frame=data.assign(score=values)
    rows=[]
    for identity,g in frame.groupby(TRAJECTORY):
        failure=g.failure_c.iloc[0]
        alarms=g[(g.c>0)&(g.score>threshold)].sort_values('c')
        alarm=float(alarms.c.iloc[0]) if len(alarms) else np.nan
        early=bool(np.isfinite(failure) and np.isfinite(alarm) and alarm<failure-1e-10)
        lead=float(failure-alarm) if early else 0.
        maxstrength=float(strength_grid(identity[-1])[-1])
        rows.append(dict(zip(TRAJECTORY,identity),failure_c=failure,first_alarm_c=alarm,
            early_warning=early,missed_warning=bool(np.isfinite(failure) and not early),
            false_alarm=bool(np.isnan(failure) and np.isfinite(alarm)),lead_normalized_strength=lead,
            lead_physical_strength=lead*maxstrength))
    return pd.DataFrame(rows)


def analyze(root,architectures=ARCHITECTURES,output_root=None):
    output_root=Path(output_root) if output_root is not None else root
    output_root.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name in architectures:
        dense,excluded=prepare_dense(pd.read_csv(root/name/'dense_scores.csv'))
        trajectories=[];thresholds=[];memberships=[]
        # Define folds once on identical dense cohorts for all resolutions.
        folds=list(GroupKFold(n_splits=5).split(dense,groups=dense.image_group))
        for levels,step in [(5,4),(10,2),(20,1)]:
            data=grid_features(dense,step)
            for fold,(dev_idx,test_idx) in enumerate(folds):
                dev_ids=sorted(dense.iloc[dev_idx].image_group.unique())
                test_ids=set(dense.iloc[test_idx].image_group)
                fit_ids,cal_ids=train_test_split(dev_ids,test_size=.25,random_state=42+fold)
                fit=data[data.image_group.isin(fit_ids)]
                cal=data[data.image_group.isin(cal_ids)]
                test=data[data.image_group.isin(test_ids)]
                for role,ids in [('fit',fit_ids),('calibration',cal_ids),('evaluation',test_ids)]:
                    memberships.extend(dict(levels=levels,fold=fold,image_group=i,role=role) for i in ids)
                scores={'MSP':(1-cal.msp.to_numpy(),1-test.msp.to_numpy()),'ODIN':(cal.odin.to_numpy(),test.odin.to_numpy())}
                training=fit[fit.eligible_fit]
                feature_sets={'Confidence + derivative':['unconfidence','derivative_unconfidence']}
                for method in ['ig','gradcam']:
                    feature_sets[f'{method.upper()} attribution features/derivatives']=[
                        f'{method}_{c}' for c in ['loss','gain','wasserstein','mass_loss','low_retention']]+[
                        f'derivative_{method}_{c}' for c in ['loss','gain','weighted_wasserstein']]
                if training.fails_within_horizon.nunique()==2:
                    for label,cols in feature_sets.items():
                        if "attribution" in label:
                            validate_attribution_columns(cols)
                        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
                        model.fit(training[cols],training.fails_within_horizon)
                        scores[label]=(model.predict_proba(cal[cols])[:,1],model.predict_proba(test[cols])[:,1])
                else:
                    for label in feature_sets:thresholds.append(dict(levels=levels,fold=fold,model=label,status='skipped: one fitting label class'))
                negative=cal.failure_c.isna()&(cal.c>0)
                for label,(cs,ts) in scores.items():
                    for persistent in [False,True]:
                        cv=alarm_values(cal,cs,persistent);tv=alarm_values(test,ts,persistent)
                        maxima=cal.assign(score=cv).loc[negative].groupby(TRAJECTORY).score.max().to_numpy()
                        threshold=calibrate_threshold(maxima,.05)
                        if not np.isfinite(threshold):
                            thresholds.append(dict(levels=levels,fold=fold,model=label,status='skipped: no nonfailure calibration trajectories'))
                            continue
                        result=outcomes(test,tv,threshold)
                        result['levels'],result['fold'],result['model'],result['rule']=levels,fold,label,'two_consecutive' if persistent else 'single'
                        trajectories.append(result)
                        thresholds.append(dict(levels=levels,fold=fold,model=label,rule='two_consecutive' if persistent else 'single',
                            threshold=threshold,calibration_negatives=len(maxima),calibration_fpr=float(np.mean(maxima>threshold)),status='ok'))
        if not trajectories:raise ValueError('No evaluable pilot folds.')
        trajectories=pd.concat(trajectories,ignore_index=True)
        for (levels,label,rule),group in trajectories.groupby(['levels','model','rule']):
            for corruption in ['all']+CORRUPTIONS:
                g=group if corruption=='all' else group[group.corruption==corruption]
                fail=g[g.failure_c.notna()];neg=g[g.failure_c.isna()]
                summaries.append(dict(classifier=name,levels=levels,model=label,rule=rule,corruption=corruption,
                    n_trajectories=len(g),n_failures=len(fail),n_nonfailures=len(neg),
                    n_early_warnings=int(fail.early_warning.sum()),early_warning_rate=float(fail.early_warning.mean()) if len(fail) else np.nan,
                    mean_lead_normalized_strength=float(fail.lead_normalized_strength.mean()) if len(fail) else np.nan,
                    mean_lead_physical_strength=float(fail.lead_physical_strength.mean()) if len(fail) and corruption!='all' else np.nan,
                    false_alarm_rate=float(neg.false_alarm.mean()) if len(neg) else np.nan))
        folder=output_root/name/'evaluation'
        folder.mkdir(parents=True)
        trajectories.to_csv(folder/'trajectories.csv',index=False)
        pd.DataFrame(thresholds).to_csv(folder/'thresholds.csv',index=False)
        pd.DataFrame(memberships).to_csv(folder/'splits.csv',index=False)
        excluded.to_csv(folder/'excluded_trajectories.csv',index=False)
    pd.DataFrame(summaries).to_csv(output_root/'summary.csv',index=False)
    print('Nested grid study complete:',output_root/'summary.csv',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/severity_spacing_attribution_only_v2'))
    parser.add_argument('--classifiers', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--analysis-output', type=Path)
    parser.add_argument('--n-per-class',type=int,default=5)
    args=parser.parse_args()
    if args.analyze_only:
        if args.analysis_output is None:
            parser.error('--analysis-output is required for analyze-only to preserve existing results.')
        if args.analysis_output.exists():
            parser.error('Use a new analysis output directory.')
        analyze(args.root,args.classifiers,args.analysis_output)
        return
    if args.root.exists():parser.error('Use a new root; historical experiments are preserved.')
    if args.n_per_class<1:parser.error('Positive sample count required.')
    args.root.mkdir(parents=True)
    protocol=dict(status='exploratory development pilot after prior final results viewed',
        n_per_class=args.n_per_class,grids='5 original nonzero strengths; subdivide each interval into 2 and 4 for nested 10 and 20',
        failure_reference='first failure on common 20-strength grid for all resolutions',
        derivative='delta feature divided by delta normalized physical strength',horizon_normalized_strength=HORIZON,
        lead='normalized and physical corruption-strength distance; zero for misses, never sample count',
        max_strength=dict(gaussian_noise=.2,brightness=.4),baseline_settings=dict(odin_temperature=1000,odin_epsilon=.0014,ig_steps=50),
        calibration='separate image groups; 5% trajectory false-alarm target recalibrated for each resolution and rule',
        calibration_caveat='small pilot negative counts yield coarse thresholds; report counts and achieved evaluation false alarms',
        sequence='same Gaussian draw per image across all noise strengths; each corruption applies to original clean image',
        interpretation='known corruption only; no unseen test results or frozen methods modified; denser-grid failure still approximates true failure')
    (args.root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    collect(args.root,args.n_per_class,args.classifiers)
    analyze(args.root,args.classifiers)


if __name__=='__main__':main()
