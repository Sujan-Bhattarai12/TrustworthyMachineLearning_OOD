"""Prepare attribution-only summary/trend features from saved observations."""
import numpy as np
import pandas as pd
from scripts.evaluate_early_warnings import prepare,KEYS,TRAJECTORY,COMPONENTS

def add_features(df):
    data,excluded=prepare(df)
    for method in ['ig','gradcam']:
        extra=df[df.method==method][KEYS+['wasserstein','retention']].copy()
        if extra.duplicated(KEYS).any():
            raise ValueError('Duplicate attribution rows.')
        extra=extra.rename(columns={k:f'{method}_{k}' for k in ['wasserstein','retention']})
        data=data.merge(extra,on=KEYS,validate='one_to_one')
    cols=[f'{m}_{c}' for m in ['ig','gradcam'] for c in ['wasserstein','retention']]
    bad=~np.isfinite(data[cols]).all(axis=1)
    bad_keys=data.loc[bad,TRAJECTORY].drop_duplicates()
    if len(bad_keys):
        data=data.merge(bad_keys.assign(bad_extra=True),on=TRAJECTORY,how='left')
        data=data[data.bad_extra.isna()].drop(columns='bad_extra')
        excluded=pd.concat([excluded,bad_keys.assign(reason='invalid additional W/retention feature')],ignore_index=True)
    if data.empty:
        raise ValueError('No common valid trajectories.')
    data=data.sort_values(KEYS).reset_index(drop=True)
    data['unconfidence']=1-data.msp
    for method in ['ig','gradcam']:
        data[f'{method}_mass_loss']=1-data[f'{method}_retention']
        # Proxy only: the stored data cannot reconstruct the exact vanished-map cutoff.
        data[f'{method}_low_retention']=(data[f'{method}_retention']<=1e-6).astype(float)
        for col in COMPONENTS+['wasserstein','mass_loss']:
            key=f'{method}_{col}'
            data[f'delta_{key}']=data.groupby(TRAJECTORY,sort=False)[key].diff().fillna(0)
    data['delta_unconfidence']=data.groupby(TRAJECTORY,sort=False).unconfidence.diff().fillna(0)
    return data,excluded

