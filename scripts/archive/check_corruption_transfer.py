"""Source-only calibration and image-disjoint cross-corruption development check."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score
from src.attribution_features import add_features
from scripts.archive.test_attribution_interactions import make_model
from scripts.evaluate_early_warnings import COMPONENTS,TRAJECTORY,calibrate_threshold,trajectory_outcomes
from src.paired_analysis import grouped_effects
from src.warning_features import validate_attribution_columns


def role_split(data,source,target,train_ids,test_ids,fold):
    fit_ids,cal_ids=train_test_split(sorted(train_ids),test_size=.25,random_state=42+fold)
    assert not(set(fit_ids)&set(cal_ids) or set(fit_ids)&set(test_ids) or set(cal_ids)&set(test_ids))
    fit=data[data.image_group.isin(fit_ids)&(data.corruption==source)]
    cal=data[data.image_group.isin(cal_ids)&(data.corruption==source)]
    test=data[data.image_group.isin(test_ids)&(data.corruption==target)]
    return fit,cal,test,fit_ids,cal_ids


def run(path,output):
    if output.exists():raise ValueError('Use a new output directory.')
    raw=pd.read_csv(path)
    if set(raw.source)!={'train'} or set(raw.corruption)!={'gaussian_noise','brightness'} or set(raw.classifier)!={'resnet50'}:
        raise ValueError('Only ResNet development noise/brightness permitted.')
    data,excluded=add_features(raw)
    outcomes=[];predictions=[];thresholds=[];memberships=[];models=[]
    for source,target in [('gaussian_noise','brightness'),('brightness','gaussian_noise')]:
        direction=f'{source}_to_{target}'
        for fold,(train_idx,test_idx) in enumerate(GroupKFold(5).split(data,groups=data.image_group)):
            train_ids=data.iloc[train_idx].image_group.unique();test_ids=data.iloc[test_idx].image_group.unique()
            fit,cal,test,fit_ids,cal_ids=role_split(data,source,target,train_ids,test_ids,fold)
            training=fit[fit.eligible_fit]
            if training.fails_next_step.nunique()!=2:raise ValueError('Both source fitting labels required.')
            for role,ids in [('fit',fit_ids),('calibration',cal_ids),('evaluation',test_ids)]:
                memberships.extend(dict(direction=direction,fold=fold,role=role,image_group=i) for i in ids)
            scores={'MSP':(1-cal.msp.to_numpy(),1-test.msp.to_numpy()),'ODIN':(cal.odin.to_numpy(),test.odin.to_numpy())}
            for method in ['ig','gradcam']:
                for variant in ['additive','interactions','retention interactions']:
                    cols=validate_attribution_columns([f'{method}_{c}' for c in COMPONENTS]+[f'delta_{method}_{c}' for c in COMPONENTS]+
                        ([f'{method}_retention'] if variant=='retention interactions' else []))
                    name=f'{method.upper()} attribution {variant}'
                    model=make_model(variant!='additive').fit(training[cols],training.fails_next_step)
                    scores[name]=(model.predict_proba(cal[cols])[:,1],model.predict_proba(test[cols])[:,1])
                    models.append(dict(direction=direction,fold=fold,name=name,columns=cols,
                        feature_names=model[:-1].get_feature_names_out(cols).tolist(),weights=model[-1].coef_[0].tolist()))
            cols=['unconfidence','delta_unconfidence']
            model=make_model().fit(training[cols],training.fails_next_step)
            scores['Confidence trends']=(model.predict_proba(cal[cols])[:,1],model.predict_proba(test[cols])[:,1])
            for name,(cs,ts) in scores.items():
                negative=cal.assign(score=cs)
                maxima=negative[negative.first_failure.isna()&(negative.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
                if not len(maxima):raise ValueError('No source nonfailing calibration trajectories.')
                threshold=calibrate_threshold(maxima,.05)
                thresholds.append(dict(direction=direction,fold=fold,model=name,threshold=threshold,n_calibration_negatives=len(maxima),source_calibration_fpr=float(np.mean(maxima>threshold))))
                result=trajectory_outcomes(test,ts,threshold)
                outcomes.append(result.assign(direction=direction,fold=fold,model=name))
                eligible=test.eligible_fit.to_numpy()
                rows=test.loc[eligible,['image_group','severity','fails_next_step']].copy()
                predictions.append(rows.assign(score=ts[eligible],direction=direction,fold=fold,model=name))
    trajectories=pd.concat(outcomes,ignore_index=True);pred=pd.concat(predictions,ignore_index=True)
    summary=[];pairs=[]
    for (direction,name),g in trajectories.groupby(['direction','model']):
        failures=g[g.first_failure.notna()];nonfail=g[g.first_failure.isna()]
        rank=pred[(pred.direction==direction)&(pred.model==name)]
        summary.append(dict(direction=direction,model=name,n_trajectories=len(g),n_failures=len(failures),n_nonfailures=len(nonfail),
            n_early=int(failures.early_warning.sum()),early_warning_rate=failures.early_warning.mean(),false_alarm_rate=nonfail.false_alarm.mean(),
            mean_lead_all_failures=failures.credited_lead_steps.mean(),auroc=roc_auc_score(rank.fails_next_step,rank.score),
            aupr=average_precision_score(rank.fails_next_step,rank.score)))
    for direction,g in trajectories.groupby('direction'):
        for method in ['IG','GRADCAM']:
            control=f'{method} attribution additive'
            for variant in ['interactions','retention interactions']:
                candidate=f'{method} attribution {variant}'
                pairs.append(dict(direction=direction,candidate=candidate,control=control,
                    **grouped_effects(g[g.model==candidate],g[g.model==control])))
    output.mkdir(parents=True)
    for name,frame in [('summary',pd.DataFrame(summary)),('trajectories',trajectories),('next_step_predictions',pred),
                       ('thresholds',pd.DataFrame(thresholds)),('memberships',pd.DataFrame(memberships)),('paired_effects',pd.DataFrame(pairs)),('excluded',excluded)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    (output/'models.json').write_text(json.dumps(models,indent=2))
    (output/'protocol.json').write_text(json.dumps(dict(scope='development-only cross-corruption check; both families previously examined',
        input=str(path),splits='5 outer image folds; source-only fit and image-disjoint calibration within outer training IDs; target evaluated only on outer evaluation IDs',
        objective='ERM logistic C=1; L/G/D + adjacent trends; products of standardized inputs, optionally retention; no confidence in attribution models',
        threshold='source nonfailure trajectory-max scores; 5% target; strict exceedance; no target recalibration',
        ranking='pooled OOF next-step scores on eligible prefailure rows, not all trajectory rows',
        caveat='source-corruption severity increments are not equal physical increments across types; all results exploratory, not untouched-domain confirmation; intervals conditional on folds'),indent=2))
    print(pd.DataFrame(summary).round(4).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/classifier_benchmarks_500/resnet50/images.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('results/archive/exploratory/resnet_corruption_transfer'))
    args=p.parse_args();run(args.input,args.output_dir)
