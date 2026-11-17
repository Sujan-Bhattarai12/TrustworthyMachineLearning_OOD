"""Attribution-only ERM versus corruption-group DRO, development ResNet pilot.

Same regularization, features, scaling and nested image splits. Balanced ERM
separates group reweighting from worst-group optimization. No unseen corruption
claim is possible from only two observed corruption families.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.archive.robust_warning import fit_attribution_warning,fit_logistic_objective,probability
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import evaluate,COMPONENTS
from src.paired_analysis import grouped_effects


def run(path,output,l2,objectives=('erm','balanced_erm','group_dro'),beta=1.):
    if output.exists():raise ValueError('Use a new experiment directory.')
    df=pd.read_csv(path)
    if set(df.source)!={'train'} or set(df.corruption)!={'gaussian_noise','brightness'}:
        raise ValueError('Only development noise/brightness data permitted.')
    data,excluded=add_features(df)
    confidence=['unconfidence','delta_unconfidence']
    recipes={m.upper():[f'{m}_{c}' for c in ['loss','gain','wasserstein','mass_loss','low_retention']]+[
        f'delta_{m}_{c}' for c in COMPONENTS] for m in ['ig','gradcam']}
    recipes['IG+GRADCAM fusion']=recipes['IG']+recipes['GRADCAM']
    fitted=[];row_predictions=[]
    def scores(fit,cal,test):
        training=fit[fit.eligible_fit]
        if training.fails_next_step.nunique()!=2:
            raise ValueError('A fitting fold lacks both failure classes; expand pilot or revise splitting.')
        output_scores={}
        fold=len(fitted)//(len(recipes)*len(objectives)+1)
        for recipe,columns in recipes.items():
            for objective in objectives:
                name=f'{recipe} attribution {objective}'
                model=fit_attribution_warning(training,columns,objective,l2,beta)
                output_scores[name]=(probability(model,cal),probability(model,test))
                fitted.append(dict(fold=fold,name=name,model=model,fit_images=sorted(fit.image_group.unique().tolist()),
                                   calibration_images=sorted(cal.image_group.unique().tolist()),evaluation_images=sorted(test.image_group.unique().tolist())))
                eligible=test[test.eligible_fit]
                values=probability(model,eligible)
                risk=-(eligible.fails_next_step.to_numpy()*np.log(np.clip(values,1e-12,1))+
                    (1-eligible.fails_next_step.to_numpy())*np.log(np.clip(1-values,1e-12,1)))
                frame=eligible[['source','image_id','corruption','severity','fails_next_step']].copy()
                frame['model'],frame['probability'],frame['log_loss'],frame['fold']=name,values,risk,fold
                row_predictions.append(frame)
        model=fit_logistic_objective(training[confidence],training.fails_next_step,training.corruption,'erm',l2)
        model.update(columns=confidence,role='baseline')
        name='Confidence trends ERM baseline'
        output_scores[name]=(probability(model,cal),probability(model,test))
        fitted.append(dict(fold=fold,name=name,model=model))
        return output_scores
    summary,trajectories,thresholds,splits,_=evaluate(data,target_fpr=.05,extra_scores=scores)
    names={m['name'] for m in fitted}|{'MSP','ODIN'}
    summary=summary[summary.model.isin(names)]
    trajectories=trajectories[trajectories.model.isin(names)]
    thresholds=thresholds[thresholds.model.isin(names)]
    risks=pd.concat(row_predictions,ignore_index=True)
    group_risks=risks.groupby(['model','corruption']).agg(n_steps=('log_loss','size'),mean_log_loss=('log_loss','mean'),
        n_failures=('fails_next_step','sum')).reset_index()
    worst=group_risks.groupby('model').mean_log_loss.max().rename('worst_corruption_log_loss')
    summary=summary.merge(worst,on='model',how='left')
    pairs=[]
    for recipe in recipes:
        control=f'{recipe} attribution erm'
        for objective in objectives:
            if objective=='erm':continue
            candidate=f'{recipe} attribution {objective}'
            effect=grouped_effects(trajectories[trajectories.model==candidate],trajectories[trajectories.model==control])
            pairs.append(dict(candidate=candidate,control=control,**effect))
    output.mkdir(parents=True)
    for label,frame in [('summary',summary),('trajectories',trajectories),('thresholds',thresholds),('splits',splits),
        ('group_log_loss',group_risks),('next_step_predictions',risks),('paired_effects',pd.DataFrame(pairs)),('excluded_trajectories',excluded)]:
        frame.to_csv(output/f'{label}.csv',index=False)
    (output/'fitted_models.json').write_text(json.dumps(fitted,indent=2,allow_nan=False))
    (output/'protocol.json').write_text(json.dumps(dict(status='exploratory development-only ResNet warning objective pilot',
        input=str(path),features='attribution only: separate L/G/W/retention features and L/G/D trends; fusion concatenates IG and Grad-CAM',
        objectives=list(objectives),vrex_beta=beta,
        regularization=l2,group_labels='corruption type, supplied only during training; not a predictor',
        solver='SLSQP for DRO; L-BFGS-B for ERM/V-REx; V-REx is not generally convex; zero initialization; analytic gradients; fitting-only StandardScaler',
        calibration='same disjoint-image 5% trajectory false-alarm policy for every method',
        caveat='both noise and brightness observed in fitting; not validation on an unseen domain; fresh corruption study requires more development families',
        no_test_read=True,no_frozen_changes=True,ci='paired image bootstrap, conditional on existing folds; no multiple-comparison correction'),indent=2))
    print(summary[summary.corruption=='all'][['model','n_early_warnings','n_failures','mean_lead_all_failures','false_alarm_rate','worst_corruption_log_loss']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/classifier_benchmarks_500/resnet50/images.csv'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/resnet_warning_erm_dro'))
    parser.add_argument('--l2',type=float,default=.01)
    parser.add_argument('--objectives',nargs='+',choices=['erm','balanced_erm','group_dro','vrex'],default=['erm','balanced_erm','group_dro'])
    parser.add_argument('--beta',type=float,default=1.)
    args=parser.parse_args()
    if 'erm' not in args.objectives or len(set(args.objectives))!=len(args.objectives) or args.beta<0:parser.error('Unique objectives including erm and nonnegative beta required.')
    if args.l2<=0:parser.error('Positive regularization required.')
    run(args.input,args.output_dir,args.l2,args.objectives,args.beta)
