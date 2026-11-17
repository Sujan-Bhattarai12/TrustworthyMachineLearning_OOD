"""Small execution-consistency diagnostic; does not evaluate warning scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from src.classifiers import load_published_classifier
from src.cifar10c_loader import CIFAR10CLoader

if __name__=='__main__':
    root=Path('results/final_cifar10c');results=[]
    torch.set_num_threads(2)
    loader=CIFAR10CLoader(root/'development_manifest.csv')
    for name in ['resnet50','vit_b_16','swin_t']:
        data=pd.read_csv(root/'development'/name/'images.csv').drop_duplicates(['source','image_id','corruption','severity'])
        data=pd.concat([data.head(64),data[(data.image_id==24)&(data.severity==5)]]).drop_duplicates(['source','image_id','corruption','severity'])
        model=load_published_classifier(Path('checkpoints/huggingface')/name,name,'cuda')
        differences=[];mismatches=0
        with torch.no_grad():
            for row in data.itertuples():
                image=loader.get_corrupted_image(int(row.image_id),row.source,row.corruption,int(row.severity),device='cuda')
                p=model(image).softmax(1)[0]
                differences.append(abs(float(p.max())-row.msp))
                mismatches+=int(int(p.argmax())!=row.predicted_class or not np.isclose(float(p.max()),row.msp,atol=1e-4,rtol=1e-4))
        results.append(dict(classifier=name,n_checked=len(data),max_msp_difference=max(differences),mismatches=mismatches))
        print(results[-1],flush=True)
        del model;torch.cuda.empty_cache()
    path=Path('results/expanded_cifar10c_5000dev_2000test/supplementary_entropy_energy/forward_validation.json')
    path.write_text(json.dumps(results,indent=2))
    if any(r['mismatches'] for r in results):raise SystemExit('Forward-consistency diagnostic failed.')
