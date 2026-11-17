"""Clean-only Pearson versus cosine similarity of saved sparse attribution codes."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.attribution_features import add_features
from scripts.archive.map_similarity.local_sparse_map_pilot import ROOT
from scripts.archive.map_similarity.local_sparse_dot_product import encode,evaluate
from scripts.evaluate_early_warnings import TRAJECTORY,calibrate_threshold

OUT=ROOT/'pearson_pilot'

def clean_similarity(z,data,pearson=True):
    z=np.asarray(z,dtype=np.float64)
    if pearson:z=z-z.mean(axis=1,keepdims=True)
    norms=np.linalg.norm(z,axis=1)
    result=np.full(len(z),np.nan)
    for _,g in data.groupby(TRAJECTORY,sort=False):
        ids=g.sort_values('severity').index.to_numpy();reference=ids[0]
        denominator=norms[ids]*norms[reference]
        valid=denominator>1e-12
        result[ids[valid]]=np.clip((z[ids[valid]]@z[reference])/denominator[valid],-1,1)
    return result.reshape(-1,1)


def prepare(folder):
    data,_=add_features(pd.read_csv(folder/'images.csv'));data=data.reset_index(drop=True)
    values={};bad=np.zeros(len(data),dtype=bool)
    for method in ['ig','gradcam']:
        z=encode(data,method,folder/'maps')
        for name,center in [('Pearson clean similarity',True),('Cosine clean similarity',False)]:
            x=clean_similarity(z,data,center);values[(method,name)]=x;bad|=~np.isfinite(x[:,0])
    # Undefined correlations for constant/empty codes: common whole-trajectory exclusion.
    keys=data.loc[bad,TRAJECTORY].drop_duplicates()
    keep=~pd.MultiIndex.from_frame(data[TRAJECTORY]).isin(pd.MultiIndex.from_frame(keys))
    return data.loc[keep].reset_index(drop=True),{key:x[keep] for key,x in values.items()},keys


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    dev,features,excluded=prepare(ROOT/'input');excluded.to_csv(OUT/'excluded_development.csv',index=False)
    fit=dev.image_group.isin(protocol['fitting_images']).to_numpy()&dev.eligible_fit.to_numpy()
    cal=dev.image_group.isin(protocol['calibration_images']).to_numpy();val=dev.image_group.isin(protocol['validation_images']).to_numpy()
    specs=[];models={};results=[]
    for (method,name),x in features.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42)).fit(x[fit],dev.loc[fit,'fails_next_step'])
        score=model.predict_proba(x)[:,1];c=dev.loc[cal].assign(score=score[cal])
        maxima=c[c.first_failure.isna()&(c.severity>0)].groupby(TRAJECTORY).score.max().to_numpy();threshold=calibrate_threshold(maxima,.05)
        path=OUT/f'{method}_{name.split()[0].lower()}_warning.joblib';joblib.dump(model,path);models[(method,name)]=model
        specs.append(dict(method=method,representation=name,threshold=float(threshold),calibration_fpr=float(np.mean(maxima>threshold)),
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        results.append(evaluate(dev.loc[val],score[val],threshold,method,name,'development validation'))
    (OUT/'frozen_specifications.json').write_text(json.dumps(dict(scope='Exploratory after repeated blur analyses, not untouched confirmation',
        feature='Pearson = cosine of mean-centered 64-dimensional sparse codes; clean reference only; one feature per warning model',
        encoder_retrained=False,previous_severity=False,constant_code_policy='Exclude whole trajectory for all comparators if Pearson or cosine undefined',
        target_calibration_fpr=.05,methods=specs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        encoders_sha256={m:hashlib.sha256((ROOT/f'{m}_sparse_encoder.pt').read_bytes()).hexdigest() for m in ['ig','gradcam']}),indent=2))
    blur,values,excluded=prepare(ROOT/'gaussian_blur_transfer/input');excluded.to_csv(OUT/'excluded_blur.csv',index=False)
    assert not set(blur.image_group)&set(dev.image_group)
    observations=[]
    for (method,name),x in values.items():
        spec=next(s for s in specs if s['method']==method and s['representation']==name)
        score=models[(method,name)].predict_proba(x)[:,1]
        results.append(evaluate(blur,score,spec['threshold'],method,name,'Gaussian blur transfer'))
        observations.append(blur[TRAJECTORY+['severity','first_failure']].assign(method=method,representation=name,similarity=x[:,0],score=score))
    result=pd.DataFrame(results);result.to_csv(OUT/'comparison.csv',index=False)
    pd.concat(observations).to_csv(OUT/'blur_similarities.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':main()
