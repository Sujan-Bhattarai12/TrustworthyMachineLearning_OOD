"""Reserve test IDs not present in recorded experiment CSVs, including archives."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reserve_fresh(root,count=500,seed=20261101):
    excluded=set()
    evidence=[]
    # Reserve against prior manifests even when some candidates were clean-incorrect.
    paths=list(Path('splits').glob('*.csv'))+list(Path('results').rglob('*.csv'))
    for path in sorted(paths):
        if root==path or root in path.parents:
            continue
        try:
            header=pd.read_csv(path,nrows=0).columns
        except pd.errors.EmptyDataError:
            continue
        if 'image_id' not in header:
            continue
        columns=['image_id']+(['source'] if 'source' in header else [])
        frame=pd.read_csv(path,usecols=columns)
        # Source-less older pilots are conservatively treated as possible exposure.
        if 'source' in frame:
            frame=frame[frame.source=='test']
        ids=set(pd.to_numeric(frame.image_id,errors='raise').dropna().astype(int))
        ids={i for i in ids if 0<=i<10000}
        if ids:
            excluded.update(ids)
            evidence.append(dict(path=str(path),sha256=sha(path),n_possible_test_ids=len(ids)))
    with Path('data/cifar-10-batches-py/test_batch').open('rb') as stream:
        labels=np.asarray(pickle.load(stream,encoding='bytes')[b'labels'])
    if count%10 or count<=0:
        raise ValueError('Fresh cohort must be a positive multiple of ten.')
    rng=np.random.default_rng(seed)
    rows=[]
    for label in range(10):
        available=[i for i in np.flatnonzero(labels==label) if int(i) not in excluded]
        if len(available)<count//10:
            raise ValueError('Insufficient unused test candidates.')
        for image in rng.choice(available,count//10,replace=False):
            rows.append(dict(image_id=int(image),class_id=label,source='test',split='test'))
    manifest=pd.DataFrame(rows).sort_values('image_id')
    assert not (set(manifest.image_id)&excluded)
    path=root/'fresh_test_manifest.csv'
    manifest.to_csv(path,index=False)
    audit=dict(seed=seed,n_candidates=len(manifest),per_class=count//10,
               excluded_possible_prior_test_ids=sorted(excluded),prior_evidence=evidence,
               manifest_sha256=sha(path),selection='label-stratified without classifier predictions or warning scores',
               scope='fresh to recorded warning experiments; external classifier training exposure is not established')
    (root/'reservation_audit.json').write_text(json.dumps(audit,indent=2))
    return path

