"""Render saved final maps beside the same official CIFAR-10-C image at levels 0–5.

No inference, fitting, corruption generation, or per-panel map normalization.
"""
import hashlib
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path('results/expanded_cifar10c_5000dev_2000test')
OUTPUT=Path('figures/official_examples')
MODELS={'resnet50':'ResNet-50','vit_b_16':'ViT-B/16','swin_t':'Swin-T'}
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    frames={name:pd.read_csv(ROOT/'test'/name/'images.csv') for name in MODELS}
    ids=[]
    for name,df in frames.items():
        assert set(df.dataset)=={'cifar10c'}
        ids.append(set(df.image_id)&{int(p.stem.split('_')[1]) for p in (ROOT/'test'/name/'maps').glob('test_*.npz')})
    image_id=min(set.intersection(*ids))
    with Path('data/cifar-10-batches-py/test_batch').open('rb') as stream:batch=pickle.load(stream,encoding='bytes')
    clean=batch[b'data'][image_id].reshape(3,32,32).transpose(1,2,0)
    label=int(batch[b'labels'][image_id])
    blur=np.load('data/CIFAR-10-C/gaussian_blur.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load('data/CIFAR-10-C/labels.npy',mmap_mode='r',allow_pickle=False)
    assert all(int(labels[(s-1)*10000+image_id])==label for s in range(1,6))
    images=[clean]+[blur[(s-1)*10000+image_id] for s in range(1,6)]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    sources={}
    for name,title in MODELS.items():
        df=frames[name];df=df[(df.image_id==image_id)&(df.corruption=='gaussian_blur')&(df.method=='ig')].sort_values('severity')
        assert list(df.severity)==list(range(6)) and df.true_label.eq(label).all()
        target=int(df.clean_class.iloc[0]);path=ROOT/'test'/name/'maps'/f'test_{image_id}.npz'
        with np.load(path,allow_pickle=False) as saved:
            maps={m:[saved[f'gaussian_blur_{s}_{m}'] for s in range(6)] for m in ['ig','gradcam']}
        fig,axes=plt.subplots(3,6,figsize=(15,8),layout='constrained')
        fig.suptitle(f'{title} | Official CIFAR-10-C Gaussian blur | Test image {image_id} ({CLASSES[label]})\nFixed attribution target: {CLASSES[target]} | IG: 64 steps',fontsize=16)
        for s,r in enumerate(df.itertuples()):
            axes[0,s].imshow(images[s],interpolation='nearest')
            axes[0,s].set_title(f'{"Clean" if s==0 else "Severity "+str(s)}\n{CLASSES[int(r.predicted_class)]}, MSP {r.msp:.3f}',fontsize=10)
        for row,method in enumerate(['ig','gradcam'],start=1):
            clean_max=float(np.max(maps[method][0]));assert clean_max>0
            scaled=[a/clean_max for a in maps[method]]
            vmax=max(float(a.max()) for a in scaled)
            for s,a in enumerate(scaled):
                im=axes[row,s].imshow(a,cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
            cb=fig.colorbar(im,ax=axes[row,:],shrink=.75,pad=.01)
            cb.set_label('Attribution / clean-map maximum',fontsize=9)
        for row,label_text in enumerate(['Official image','IG','Grad-CAM']):
            axes[row,0].set_ylabel(label_text,fontsize=13)
            for ax in axes[row,:]:ax.set_xticks([]);ax.set_yticks([])
        for suffix in ['png','pdf']:fig.savefig(OUTPUT/f'{name}_same_image_gaussian_blur.{suffix}',dpi=180)
        plt.close(fig)
        sources[name]={'observations':str(ROOT/'test'/name/'images.csv'),'maps':str(path),'maps_sha256':sha(path),'observations_sha256':sha(ROOT/'test'/name/'images.csv')}
    provenance={'image_id':image_id,'selection':'Lowest test image ID with saved maps across all three final classifier cohorts; independent of warning scores.','source':'Original CIFAR-10 test_batch and official CIFAR-10-C gaussian_blur.npy blocks','corruption':'gaussian_blur','severities':list(range(6)),'map_scaling':'Each method divided by its own clean-map maximum. One color range across severities per model/method row; no per-panel rescaling.','sources':sources,'purpose':'Illustrative qualitative examples, not representative performance evidence.','script_sha256':sha(__file__)}
    (OUTPUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (OUTPUT/'README.md').write_text(f'''# Official same-image attribution examples\n\nEach model figure shows the **same original test image, ID {image_id}**, at severity 0 (clean) and official CIFAR-10-C Gaussian-blur severities 1–5. Rows show image pixels, IG maps and Grad-CAM maps. All three model figures use the same image ID.\n\nImages are loaded directly from official arrays; no manual corruption generator is used. Maps and predictions are reused from the completed final run, with IG64 and the fixed clean predicted target. Each map row uses its clean-map maximum as the scale reference and a shared color range across severities, preserving magnitude changes. Color scales can differ between methods/models; do not compare their raw magnitudes directly.\n\nSelection is the lowest common saved test ID, independent of score or success. This example is illustrative, not proof of typical behavior. Source paths/hashes and selection are in `provenance.json`.\n\nRebuild: `MPLCONFIGDIR=/tmp/asws-final-mpl .venv/bin/python -m scripts.build_official_attribution_examples`. No inference or fitting is performed.\n\nAggregate performance remains in `results/reporting/final/`; these qualitative figures do not replace it.\n''')
    print('Built official examples for common test image',image_id)

if __name__=='__main__':main()
