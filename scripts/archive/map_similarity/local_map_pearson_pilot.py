"""Clean-only Pearson versus cosine similarity of attribution-map matrices."""
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
from scripts.archive.map_similarity.local_sparse_dot_product import evaluate
from scripts.evaluate_early_warnings import TRAJECTORY,calibrate_threshold

OUT=ROOT/'map_pearson_pilot'

def clean_similarity(z,data,pearson=True):
    z=np.asarray(z,dtype=np.float64)
    z=z/np.maximum(np.max(np.abs(z),axis=1,keepdims=True),np.finfo(float).tiny)
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
        maps=[]
        for row in data.itertuples():
            with np.load(folder/'maps'/f'{row.source}_{row.image_id}.npz',allow_pickle=False) as saved:
                maps.append(saved[f'{row.corruption}_{row.severity}_{method}'].reshape(-1))
        z=np.asarray(maps,dtype=np.float64)
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
    for name,column in [('MSP','unconfidence'),('ODIN','odin')]:
        score=dev[column].to_numpy();c=dev.loc[cal].assign(score=score[cal])
        maxima=c[c.first_failure.isna()&(c.severity>0)].groupby(TRAJECTORY).score.max().to_numpy();threshold=calibrate_threshold(maxima,.05)
        specs.append(dict(method='baseline',representation=name,threshold=float(threshold),calibration_fpr=float(np.mean(maxima>threshold)),column=column))
        results.append(evaluate(dev.loc[val],score[val],threshold,'baseline',name,'development validation'))
    (OUT/'frozen_specifications.json').write_text(json.dumps(dict(scope='Exploratory after repeated blur analyses, not untouched confirmation',
        feature='Pearson over all 1024 corresponding attribution-map pixels after subtracting each map mean; clean reference only; one feature per warning model',
        sparse_encoding=False,previous_severity=False,constant_code_policy='Exclude whole trajectory for all comparators if Pearson or cosine undefined',
        target_calibration_fpr=.05,methods=specs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        development_csv_sha256=hashlib.sha256((ROOT/'input/images.csv').read_bytes()).hexdigest()),indent=2))
    blur,values,excluded=prepare(ROOT/'gaussian_blur_transfer/input');excluded.to_csv(OUT/'excluded_blur.csv',index=False)
    assert not set(blur.image_group)&set(dev.image_group)
    observations=[]
    for (method,name),x in values.items():
        spec=next(s for s in specs if s['method']==method and s['representation']==name)
        score=models[(method,name)].predict_proba(x)[:,1]
        results.append(evaluate(blur,score,spec['threshold'],method,name,'Gaussian blur transfer'))
        observations.append(blur[TRAJECTORY+['severity','first_failure']].assign(method=method,representation=name,similarity=x[:,0],score=score))
    for spec in specs:
        if spec['method']!='baseline':continue
        score=blur[spec['column']].to_numpy()
        results.append(evaluate(blur,score,spec['threshold'],'baseline',spec['representation'],'Gaussian blur transfer'))
    result=pd.DataFrame(results);result.to_csv(OUT/'comparison.csv',index=False)
    pd.concat(observations).to_csv(OUT/'blur_similarities.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':main()
