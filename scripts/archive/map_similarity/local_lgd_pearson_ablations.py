"""Complete exploratory Grad-CAM L/G/D subset ablations with/without Pearson."""
from itertools import combinations
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.attribution_features import add_features
from scripts.archive.map_similarity.local_sparse_map_pilot import ROOT
from scripts.archive.map_similarity.local_map_pearson_pilot import prepare
from scripts.archive.map_similarity.local_sparse_dot_product import evaluate
from scripts.evaluate_early_warnings import COMPONENTS,TRAJECTORY,calibrate_threshold
import scripts.archive.map_similarity.evaluate_fresh_map_pearson as fresh

OUT=ROOT/'lgd_pearson_ablations'
BUDGETS=[.02,.05,.1]
BASE=[f'gradcam_{c}' for c in COMPONENTS]
LGD=BASE+[f'delta_{k}' for k in BASE]


def feature_sets(data,pearson):
    pearson=np.asarray(pearson).reshape(-1,1)
    assert len(data)==len(pearson) and np.isfinite(pearson).all()
    result={'Pearson alone':pearson}
    for size in [1,2,3]:
        for indices in combinations(range(3),size):
            label='+'.join('LGD'[i] for i in indices)
            x=data[[BASE[i] for i in indices]].to_numpy()
            result[label]=x
            result[label+'+Pearson']=np.column_stack([x,pearson])
    result['L/G/D + trends']=data[LGD].to_numpy()
    result['L/G/D + trends + Pearson']=np.column_stack([data[LGD].to_numpy(),pearson])
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split=json.loads((ROOT/'protocol.json').read_text())
    dev,similarities,excluded=prepare(ROOT/'input')
    excluded.to_csv(OUT/'excluded_development.csv',index=False)
    values=feature_sets(dev,similarities[('gradcam','Pearson clean similarity')])
    fit=dev.image_group.isin(split['fitting_images']).to_numpy()&dev.eligible_fit.to_numpy()
    cal=dev.image_group.isin(split['calibration_images']).to_numpy();val=dev.image_group.isin(split['validation_images']).to_numpy()
    models={};specs=[];results=[]
    for name,x in values.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42)).fit(x[fit],dev.loc[fit,'fails_next_step'])
        models[name]=model;score=model.predict_proba(x)[:,1]
        path=OUT/f'model_{len(models)}.joblib';joblib.dump(model,path)
        for target in BUDGETS:
            c=dev.loc[cal].assign(score=score[cal]);maxima=c[c.first_failure.isna()&(c.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
            threshold=calibrate_threshold(maxima,target)
            specs.append(dict(name=name,target=target,threshold=float(threshold),model_file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            result=evaluate(dev.loc[val],score[val],threshold,'gradcam',name,'development validation');result['target_calibration_fpr']=target;results.append(result)
    for name,column in [('MSP','unconfidence'),('ODIN','odin')]:
        score=dev[column].to_numpy()
        for target in BUDGETS:
            c=dev.loc[cal].assign(score=score[cal]);maxima=c[c.first_failure.isna()&(c.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
            threshold=calibrate_threshold(maxima,target)
            specs.append(dict(name=name,column=column,target=target,threshold=float(threshold)))
            result=evaluate(dev.loc[val],score[val],threshold,'baseline',name,'development validation');result['target_calibration_fpr']=target;results.append(result)
    # Save models and calibration-only thresholds before either blur cohort is loaded.
    (OUT/'frozen_specifications.json').write_text(json.dumps(dict(scope='Exploratory repeated-cohort comparisons; no untouched confirmation',
        primary_target=.05,secondary_targets=BUDGETS,features=LGD+['clean_map_pearson'],confidence_in_attribution_model=False,
        reporting='Report all subsets; no test-based model selection',methods=specs,
        training='Original pilot image partitions, common valid Pearson/cosine trajectories for all comparators',
        retained_fit_images=int(dev.loc[dev.image_group.isin(split['fitting_images'])].image_group.nunique()),
        retained_calibration_images=int(dev.loc[cal].image_group.nunique()),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    original,sim,excluded=prepare(ROOT/'gaussian_blur_transfer/input')
    excluded.to_csv(OUT/'excluded_original_blur.csv',index=False)
    additional,_=add_features(pd.read_csv(fresh.ROOT/'images.csv'));additional=additional.reset_index(drop=True)
    with patch.object(fresh,'ROOT',Path('results/fresh_map_pearson_200')):pearson,_=fresh.similarities(additional,'gradcam')
    keys=pd.read_csv(fresh.ROOT/'trajectories.csv')[TRAJECTORY].drop_duplicates()
    keep=pd.MultiIndex.from_frame(additional[TRAJECTORY]).isin(pd.MultiIndex.from_frame(keys))
    additional=additional.loc[keep].reset_index(drop=True);pearson=pearson[keep]
    for name,data,p in [('original blur, 165 trajectories',original,sim[('gradcam','Pearson clean similarity')]),
                         ('additional blur, 168 trajectories',additional,pearson)]:
        assert not set(data.image_group)&set(dev.image_group)
        features=feature_sets(data,p)
        for spec in specs:
            label=spec['name'];score=data[spec['column']].to_numpy() if 'column' in spec else models[label].predict_proba(features[label])[:,1]
            result=evaluate(data,score,spec['threshold'],'baseline' if 'column' in spec else 'gradcam',label,name)
            result['target_calibration_fpr']=spec['target'];results.append(result)
    result=pd.DataFrame(results);result.to_csv(OUT/'all_calibrated_targets.csv',index=False)
    primary=result[result.target_calibration_fpr==.05];primary.to_csv(OUT/'comparison.csv',index=False)
    print(primary.to_string(index=False))

if __name__=='__main__':main()
