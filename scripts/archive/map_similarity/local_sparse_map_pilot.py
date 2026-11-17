"""Development-only pilot: sparse attribution encodings versus PCA and L/G/D."""
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import TRAJECTORY, calibrate_threshold, trajectory_outcomes, COMPONENTS

ROOT=Path('results/local_sparse_map_pilot')
SEED=42

def distances(z, data):
    frame=pd.DataFrame(z,index=data.index)
    ref=np.empty_like(z);previous=np.empty_like(z)
    for _,g in data.groupby(TRAJECTORY,sort=False):
        ids=g.sort_values('severity').index.to_numpy()
        ref[ids]=z[ids[0]];previous[ids]=z[np.r_[ids[0],ids[:-1]]]
    def cosine(a,b):
        denominator=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        result=1-np.sum(a*b,axis=1)/np.maximum(denominator,1e-12)
        result[(np.linalg.norm(a,axis=1)==0)&(np.linalg.norm(b,axis=1)==0)]=0
        return np.clip(result,0,2)
    values=np.column_stack([np.linalg.norm(z-ref,axis=1),cosine(z,ref),np.linalg.norm(z-previous,axis=1),cosine(z,previous)])
    return values


def main():
    start=time.time();torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    data,_=add_features(pd.read_csv(ROOT/'input/images.csv'));data=data.reset_index(drop=True)
    original=json.loads((ROOT/'input/specifications.json').read_text())
    fitting,validation=train_test_split(original['fit_image_groups'],test_size=.25,random_state=SEED)
    calibration=original['calibration_image_groups']
    assert not(set(fitting)&set(validation) or set(fitting)&set(calibration) or set(validation)&set(calibration))
    masks={name:data.image_group.isin(ids).to_numpy() for name,ids in [('fit',fitting),('calibration',calibration),('validation',validation)]}
    # Fixed choices before results: no hyperparameter search or test data access.
    protocol=dict(scope='Exploratory local development-only holdout; not final test confirmation',classifier='resnet50',seed=SEED,
        fitting_images=fitting,calibration_images=calibration,validation_images=validation,
        preprocessing='log1p(nonnegative map / corresponding clean map maximum), then divide by fitting-only global RMS',
        latent_dimensions=64,epochs=40,batch_size=128,learning_rate=.001,sparsity_penalty=.001,
        encoder='1024->64 ReLU; linear decoder; MSE + .001 mean absolute latent activation; compact sparse bottleneck, not circuit discovery',
        warning_features=['clean displacement L2','clean cosine distance','consecutive displacement L2','consecutive cosine distance'],
        warning_classifier='StandardScaler + LogisticRegression C=1; eligible next-step failure labels',
        calibration='Same frozen calibration groups; nonfailing trajectory maxima, target5%, strict exceedance',
        source_sha256=hashlib.sha256((ROOT/'input/images.csv').read_bytes()).hexdigest())
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[];diagnostics=[];outcomes=[];saved_scores=[]
    for method in ['ig','gradcam']:
        maps=[]
        for r in data.itertuples():
            with np.load(ROOT/'input/maps'/f'{r.source}_{r.image_id}.npz',allow_pickle=False) as saved:
                clean=saved[f'{r.corruption}_0_{method}'].reshape(-1)
                value=saved[f'{r.corruption}_{r.severity}_{method}'].reshape(-1)
                maps.append(np.log1p(np.maximum(value,0)/max(float(clean.max()),1e-12)))
        x=np.asarray(maps,dtype=np.float32)
        rms=float(np.sqrt(np.mean(x[masks['fit']]**2)));x/=max(rms,1e-12)
        fitx=x[masks['fit']]
        encoder=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU())
        decoder=nn.Linear(64,x.shape[1]);optimizer=torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=.001)
        train=torch.from_numpy(fitx)
        for epoch in range(40):
            for indices in torch.randperm(len(train)).split(128):
                batch=train[indices];latent=encoder(batch);reconstruction=decoder(latent)
                loss=(reconstruction-batch).square().mean()+.001*latent.abs().mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            latent=encoder(torch.from_numpy(x)).numpy()
            reconstructed=decoder(torch.from_numpy(latent)).numpy()
        torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict(),input_dimension=x.shape[1],rms=rms),ROOT/f'{method}_sparse_encoder.pt')
        pca=PCA(n_components=64,random_state=SEED).fit(fitx);pca_latent=pca.transform(x)
        joblib.dump(pca,ROOT/f'{method}_pca.joblib')
        for split,mask in masks.items():
            diagnostics.append(dict(method=method,split=split,n_maps=int(mask.sum()),sparse_zero_fraction=float(np.mean(latent[mask]==0)),
                sparse_reconstruction_mse=float(np.mean((reconstructed[mask]-x[mask])**2)),
                pca_reconstruction_mse=float(np.mean((pca.inverse_transform(pca_latent[mask])-x[mask])**2))))
        base=[f'{method}_{c}' for c in COMPONENTS]
        features={'L/G/D + trends':data[base+[f'delta_{k}' for k in base]].to_numpy(),
                  'Sparse encoding distances':distances(latent,data),'PCA encoding distances':distances(pca_latent,data)}
        for name,values in features.items():
            fitmask=masks['fit'] & data.eligible_fit.to_numpy()
            warning=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=SEED)).fit(values[fitmask],data.loc[fitmask,'fails_next_step'])
            scores=warning.predict_proba(values)[:,1]
            saved_scores.append(data[TRAJECTORY+["severity","first_failure","image_group"]].assign(method=method,representation=name,score=scores))
            cal=data.loc[masks['calibration']].assign(score=scores[masks['calibration']])
            maxima=cal[cal.first_failure.isna()&(cal.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
            threshold=calibrate_threshold(maxima,.05)
            validation_data=data.loc[masks['validation']]
            trajectory=trajectory_outcomes(validation_data,scores[masks['validation']],threshold)
            trajectory['method']=method;trajectory['representation']=name;outcomes.append(trajectory)
            fail=trajectory[trajectory.first_failure.notna()];nonfail=trajectory[trajectory.first_failure.isna()]
            eligible=masks['validation']&data.eligible_fit.to_numpy();y=data.loc[eligible,'fails_next_step'];s=scores[eligible]
            results.append(dict(method=method,representation=name,validation_images=validation_data.image_group.nunique(),
                n_trajectories=len(trajectory),n_failures=len(fail),n_nonfailures=len(nonfail),early_warning_rate=fail.early_warning.mean(),
                false_alarm_rate=nonfail.false_alarm.mean(),mean_lead_all_failures=fail.credited_lead_steps.mean(),
                auroc=roc_auc_score(y,s),aupr=average_precision_score(y,s),next_failure_prevalence=float(y.mean()),
                threshold=float(threshold),calibration_fpr=float(np.mean(maxima>threshold))))
            joblib.dump(warning,ROOT/f'{method}_{name.split()[0].replace("/","_")}_warning.joblib')
        print(f'Completed {method}: {time.time()-start:.1f}s',flush=True)
    # Output-only MSP comparison on exactly the same development splits and cohort.
    for name,column in [('MSP','unconfidence'),('ODIN','odin')]:
        scores=data[column].to_numpy();saved_scores.append(data[TRAJECTORY+['severity','first_failure','image_group']].assign(method='baseline',representation=name,score=scores));cal=data.loc[masks['calibration']].assign(score=scores[masks['calibration']])
        maxima=cal[cal.first_failure.isna()&(cal.severity>0)].groupby(TRAJECTORY).score.max().to_numpy();threshold=calibrate_threshold(maxima,.05)
        val=data.loc[masks['validation']];trajectory=trajectory_outcomes(val,scores[masks['validation']],threshold)
        fail=trajectory[trajectory.first_failure.notna()];nonfail=trajectory[trajectory.first_failure.isna()]
        eligible=masks['validation']&data.eligible_fit.to_numpy();y=data.loc[eligible,'fails_next_step'];s=scores[eligible]
        results.append(dict(method='baseline',representation=name,validation_images=val.image_group.nunique(),n_trajectories=len(trajectory),n_failures=len(fail),n_nonfailures=len(nonfail),
            early_warning_rate=fail.early_warning.mean(),false_alarm_rate=nonfail.false_alarm.mean(),mean_lead_all_failures=fail.credited_lead_steps.mean(),
            auroc=roc_auc_score(y,s),aupr=average_precision_score(y,s),next_failure_prevalence=float(y.mean()),threshold=float(threshold),calibration_fpr=float(np.mean(maxima>threshold))))
    pd.concat(saved_scores,ignore_index=True).to_csv(ROOT/'observation_scores.csv',index=False)
    result=pd.DataFrame(results);result.to_csv(ROOT/'comparison.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT/'encoding_diagnostics.csv',index=False)
    pd.concat(outcomes).to_csv(ROOT/'attribution_trajectories.csv',index=False)
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',seconds=time.time()-start),indent=2))
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
