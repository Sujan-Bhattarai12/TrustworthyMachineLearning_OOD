"""Exploratory sparse-code dot products, with development-frozen thresholds."""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from src.attribution_features import add_features
from scripts.archive.map_similarity.local_sparse_map_pilot import ROOT,distances
from scripts.evaluate_early_warnings import TRAJECTORY,calibrate_threshold,trajectory_outcomes

OUT=ROOT/'dot_product_pilot'

def dot_features(z,data,normalized=False):
    if normalized:
        norms=np.linalg.norm(z,axis=1,keepdims=True)
        z=z/np.maximum(norms,1e-12)
    result=np.empty((len(z),2),dtype=np.float64)
    for _,group in data.groupby(TRAJECTORY,sort=False):
        ids=group.sort_values('severity').index.to_numpy();previous=np.r_[ids[0],ids[:-1]]
        result[ids,0]=np.sum(z[ids]*z[ids[0]],axis=1)
        result[ids,1]=np.sum(z[ids]*z[previous],axis=1)
        if normalized:
            zero=np.linalg.norm(z[ids],axis=1)==0
            result[ids[zero & (np.linalg.norm(z[ids[0]])==0)],0]=1
            bothzero=zero & (np.linalg.norm(z[previous],axis=1)==0)
            result[ids[bothzero],1]=1
    return result


def encode(data,method,folder):
    state=torch.load(ROOT/f'{method}_sparse_encoder.pt',map_location='cpu',weights_only=True)
    encoder=nn.Sequential(nn.Linear(state['input_dimension'],64),nn.ReLU());encoder.load_state_dict(state['encoder']);encoder.eval()
    maps=[]
    for r in data.itertuples():
        with np.load(folder/f'{r.source}_{r.image_id}.npz',allow_pickle=False) as stored:
            a=stored[f'{r.corruption}_0_{method}'].reshape(-1);b=stored[f'{r.corruption}_{r.severity}_{method}'].reshape(-1)
            maps.append(np.log1p(np.maximum(b,0)/max(float(a.max()),1e-12)))
    x=np.asarray(maps,dtype=np.float32)/max(state['rms'],1e-12)
    with torch.inference_mode():return encoder(torch.from_numpy(x)).numpy()


def evaluate(data,scores,threshold,method,representation,cohort):
    outcome=trajectory_outcomes(data,scores,threshold)
    fail=outcome[outcome.first_failure.notna()];nonfail=outcome[outcome.first_failure.isna()]
    mask=data.eligible_fit.to_numpy();y=data.loc[mask,'fails_next_step'];s=scores[mask]
    return dict(cohort=cohort,method=method,representation=representation,threshold=float(threshold),
        n_failures=len(fail),n_nonfailures=len(nonfail),n_early=int(fail.early_warning.sum()),n_false=int(nonfail.false_alarm.sum()),
        early_warning_rate=fail.early_warning.mean(),false_alarm_rate=nonfail.false_alarm.mean(),
        auroc=roc_auc_score(y,s),aupr=average_precision_score(y,s))


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    development,_=add_features(pd.read_csv(ROOT/'input/images.csv'));development=development.reset_index(drop=True)
    fit=development.image_group.isin(protocol['fitting_images']).to_numpy()&development.eligible_fit.to_numpy()
    cal=development.image_group.isin(protocol['calibration_images']).to_numpy()
    val=development.image_group.isin(protocol['validation_images']).to_numpy()
    recipes={'Raw dot products':False,'Normalized dot products (cosine)':True}
    models={};specs=[];results=[]
    for method in ['ig','gradcam']:
        z=encode(development,method,ROOT/'input/maps')
        features={name:dot_features(z,development,normalized) for name,normalized in recipes.items()}
        features['Original L2 + cosine distances']=distances(z,development)
        for name,x in features.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42)).fit(x[fit],development.loc[fit,'fails_next_step'])
            scores=model.predict_proba(x)[:,1]
            calibration=development.loc[cal].assign(score=scores[cal])
            maxima=calibration[calibration.first_failure.isna()&(calibration.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
            threshold=calibrate_threshold(maxima,.05)
            key=f'{method}_{len(specs)}';path=OUT/f'{key}.joblib';joblib.dump(model,path)
            models[(method,name)]=model
            specs.append(dict(method=method,representation=name,threshold=float(threshold),calibration_fpr=float(np.mean(maxima>threshold)),
                model_file=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),weights=model[-1].coef_[0].tolist()))
            results.append(evaluate(development.loc[val],scores[val],threshold,method,name,'development validation'))
    # Save fitted models/thresholds before any blur data are loaded; no test tuning.
    frozen=dict(scope='Exploratory after prior blur analyses; not untouched confirmation',target_calibration_fpr=.05,
        features='Two similarities: current vs clean and current vs previous severity; fixed saved sparse encoders; zero/zero cosine defined as 1',
        methods=specs,encoder_sha256={m:hashlib.sha256((ROOT/f'{m}_sparse_encoder.pt').read_bytes()).hexdigest() for m in ['ig','gradcam']})
    (OUT/'frozen_specifications.json').write_text(json.dumps(frozen,indent=2))
    blur,_=add_features(pd.read_csv(ROOT/'gaussian_blur_transfer/input/images.csv'));blur=blur.reset_index(drop=True)
    assert not set(blur.image_group)&set(development.image_group)
    for method in ['ig','gradcam']:
        z=encode(blur,method,ROOT/'gaussian_blur_transfer/input/maps')
        features={name:dot_features(z,blur,normalized) for name,normalized in recipes.items()}
        features['Original L2 + cosine distances']=distances(z,blur)
        for name,x in features.items():
            spec=next(s for s in specs if s['method']==method and s['representation']==name)
            scores=models[(method,name)].predict_proba(x)[:,1]
            results.append(evaluate(blur,scores,spec['threshold'],method,name,'Gaussian blur transfer'))
    result=pd.DataFrame(results);result.to_csv(OUT/'comparison.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':main()
