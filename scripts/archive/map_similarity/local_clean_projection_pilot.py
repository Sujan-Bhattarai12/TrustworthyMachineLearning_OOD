"""Clean-reference attribution projection coefficient; no previous-severity features."""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.attribution_features import add_features
from scripts.archive.map_similarity.local_sparse_map_pilot import ROOT
from scripts.archive.map_similarity.local_sparse_dot_product import evaluate
from scripts.evaluate_early_warnings import TRAJECTORY,calibrate_threshold

OUT=ROOT/'clean_projection_pilot'

def projection_coefficient(clean,current):
    a=np.asarray(clean,dtype=np.float64).reshape(-1)
    b=np.asarray(current,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Projection requires matching finite maps.')
    norm=float(a@a)
    if norm<=0:raise ValueError('Cannot project onto an empty clean map.')
    return float((b@a)/norm)


def features(data,folder,method):
    result=[]
    for r in data.itertuples():
        with np.load(folder/f'{r.source}_{r.image_id}.npz',allow_pickle=False) as saved:
            result.append(projection_coefficient(saved[f'{r.corruption}_0_{method}'],saved[f'{r.corruption}_{r.severity}_{method}']))
    return np.asarray(result).reshape(-1,1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    development,_=add_features(pd.read_csv(ROOT/'input/images.csv'));development=development.reset_index(drop=True)
    fit=development.image_group.isin(protocol['fitting_images']).to_numpy()&development.eligible_fit.to_numpy()
    cal=development.image_group.isin(protocol['calibration_images']).to_numpy()
    val=development.image_group.isin(protocol['validation_images']).to_numpy()
    models={};specs=[];results=[];observations=[]
    for method in ['ig','gradcam']:
        x=features(development,ROOT/'input/maps',method)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42)).fit(x[fit],development.loc[fit,'fails_next_step'])
        score=model.predict_proba(x)[:,1]
        c=development.loc[cal].assign(score=score[cal])
        maxima=c[c.first_failure.isna()&(c.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
        threshold=calibrate_threshold(maxima,.05)
        path=OUT/f'{method}_warning.joblib';joblib.dump(model,path);models[method]=model
        specs.append(dict(method=method,threshold=float(threshold),calibration_fpr=float(np.mean(maxima>threshold)),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            standardized_coefficient=float(model[-1].coef_[0,0]),intercept=float(model[-1].intercept_[0])))
        results.append(evaluate(development.loc[val],score[val],threshold,method,'Clean-reference projection coefficient','development validation'))
        observations.append(development[TRAJECTORY+['severity','first_failure','image_group']].assign(method=method,cohort='development',projection_coefficient=x[:,0],score=score))
    (OUT/'frozen_specifications.json').write_text(json.dumps(dict(scope='Exploratory after repeated prior blur analyses; not untouched confirmation',
        feature='alpha = dot(corrupted attribution map, clean attribution map) / dot(clean map, clean map)',
        no_previous_severity=True,no_sparse_encoder=True,target_calibration_fpr=.05,methods=specs,
        source_development_sha256=hashlib.sha256((ROOT/'input/images.csv').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    blur,_=add_features(pd.read_csv(ROOT/'gaussian_blur_transfer/input/images.csv'));blur=blur.reset_index(drop=True)
    assert not set(blur.image_group)&set(development.image_group)
    for spec in specs:
        method=spec['method'];x=features(blur,ROOT/'gaussian_blur_transfer/input/maps',method)
        score=models[method].predict_proba(x)[:,1]
        results.append(evaluate(blur,score,spec['threshold'],method,'Clean-reference projection coefficient','Gaussian blur transfer'))
        observations.append(blur[TRAJECTORY+['severity','first_failure','image_group']].assign(method=method,cohort='Gaussian blur transfer',projection_coefficient=x[:,0],score=score))
    result=pd.DataFrame(results);result.to_csv(OUT/'comparison.csv',index=False)
    pd.concat(observations,ignore_index=True).to_csv(OUT/'projection_coefficients.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':main()
