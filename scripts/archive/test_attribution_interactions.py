"""Development-only ResNet pilot: additive L/G/D versus pairwise interactions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from src.warning_features import validate_attribution_columns
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import evaluate,COMPONENTS
from src.paired_analysis import grouped_effects


def make_model(interactions=False):
    steps=[StandardScaler()]
    if interactions:
        steps.extend([PolynomialFeatures(degree=2,interaction_only=True,include_bias=False),StandardScaler()])
    return make_pipeline(*steps,LogisticRegression(C=1,max_iter=2000,random_state=42))


def run(path,output):
    if output.exists():raise ValueError('Use a new output directory.')
    raw=pd.read_csv(path)
    if set(raw.source)!={'train'} or set(raw.corruption)!={'gaussian_noise','brightness'} or set(raw.classifier)!={'resnet50'}:
        raise ValueError('Only ResNet development noise/brightness allowed.')
    data,excluded=add_features(raw)
    fitted=[];predictions=[]
    def scores(fit,cal,test):
        training=fit[fit.eligible_fit]
        if training.fails_next_step.nunique()!=2:raise ValueError('Both training labels required.')
        fold=len(fitted)//13
        result={}
        for method in ['ig','gradcam']:
            base=[f'{method}_{c}' for c in COMPONENTS]
            for temporal in [False,True]:
                for retention in [False,True]:
                    cols=validate_attribution_columns(base+([f'delta_{method}_{c}' for c in COMPONENTS] if temporal else [])+([f'{method}_retention'] if retention else []))
                    # Retention is already saved and used in D; extra model tests whether exposing it helps.
                    variants=[False,True] if not retention else [True]
                    for interactions in variants:
                        name=f'{method.upper()} attribution '+('trends ' if temporal else '')+('LGD+retention interactions' if retention else 'LGD interactions' if interactions else 'LGD additive')
                        model=make_model(interactions).fit(training[cols],training.fails_next_step)
                        result[name]=(model.predict_proba(cal[cols])[:,1],model.predict_proba(test[cols])[:,1])
                        eligible=test[test.eligible_fit]
                        frame=eligible[['image_group','corruption','severity','fails_next_step']].copy()
                        frame['score']=model.predict_proba(eligible[cols])[:,1];frame['model']=name;frame['fold']=fold
                        predictions.append(frame)
                        fitted.append(dict(fold=fold,model=name,columns=cols,feature_names=model[:-1].get_feature_names_out(cols).tolist(),
                            coefficients=model[-1].coef_[0].tolist(),intercept=float(model[-1].intercept_[0])))
        cols=['unconfidence','delta_unconfidence']
        model=make_model().fit(training[cols],training.fails_next_step)
        name='Confidence trends baseline'
        result[name]=(model.predict_proba(cal[cols])[:,1],model.predict_proba(test[cols])[:,1])
        fitted.append(dict(fold=fold,model=name,columns=cols))
        return result
    summary,traj,thresholds,splits,_=evaluate(data,target_fpr=.05,extra_scores=scores)
    names={r['model'] for r in fitted}|{'MSP','ODIN'}
    summary=summary[summary.model.isin(names)];traj=traj[traj.model.isin(names)]
    thresholds=thresholds[thresholds.model.isin(names)]
    pairs=[]
    for method in ['IG','GRADCAM']:
        for prefix in ['', 'trends ']:
            control=f'{method} attribution {prefix}LGD additive'
            for variant in ['LGD interactions','LGD+retention interactions']:
                candidate=f'{method} attribution {prefix}{variant}'
                pairs.append(dict(candidate=candidate,control=control,**grouped_effects(traj[traj.model==candidate],traj[traj.model==control])))
    pred=pd.concat(predictions,ignore_index=True)
    ranking=[]
    for name,g in pred.groupby('model'):
        ranking.append(dict(model=name,n_steps=len(g),auroc=roc_auc_score(g.fails_next_step,g.score),
                            aupr=average_precision_score(g.fails_next_step,g.score)))
    output.mkdir(parents=True)
    for label,frame in [('summary',summary),('trajectories',traj),('thresholds',thresholds),('splits',splits),
                        ('paired_effects',pd.DataFrame(pairs)),('next_step_predictions',pred),('ranking',pd.DataFrame(ranking)),('excluded',excluded)]:
        frame.to_csv(output/f'{label}.csv',index=False)
    (output/'models.json').write_text(json.dumps(fitted,indent=2))
    (output/'protocol.json').write_text(json.dumps(dict(scope='exploratory development-only; no test access',
        inputs=str(path),features='L/G/D, optionally retention; no confidence in attribution models',
        interaction_definition='pairwise products of fit-standardized inputs; no squares; expanded features scaled on fit only',
        fixed_hyperparameters='C=1 for all candidates; no tuning',
        evaluation='same 5 image-group folds and inner image-disjoint calibration; 5% nonfailing-trajectory target',
        caveats='Both corruptions observed in fitting; no OOD generalization claim. OOF ranking pools fold scores. Bootstrap conditional on folds; no multiplicity correction.'),indent=2))
    print(summary[summary.corruption=='all'][['model','n_early_warnings','n_failures','false_alarm_rate','mean_lead_all_failures']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('results/classifier_benchmarks_500/resnet50/images.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('results/archive/exploratory/resnet_attribution_interactions_v2'))
    args=p.parse_args();run(args.input,args.output_dir)
