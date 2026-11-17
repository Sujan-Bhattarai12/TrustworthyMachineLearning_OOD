"""Standalone 50-image ASWS pilot for BOTH Grad-CAM and Integrated Gradients.

Run from the project root:
    pip install captum POT
    python -m scripts.collect_attribution_metrics --classifier resnet50 --checkpoint PATH

Needs your existing project modules: src.data_loader, src.corruptions, src.grad_cam.
Selects 5 clean-correct development images per CIFAR-10 class.
Uses Gaussian noise and brightness at severities 0 through 5 and the clean predicted class for every map.
IG is converted to a nonnegative map by summing absolute RGB attribution.
Output: results/classifier_benchmarks/<classifier>/images.csv and summary.csv
"""
import argparse
import csv
import time
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import ot
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from src.classifiers import ARCHITECTURES, load_classifier, load_published_classifier, attribution_layer
from src.benchmark_scores import benchmark_scores

from src.data_loader import CIFARDataLoader
from src.cifar10c_loader import CIFAR10CLoader,CIFAR10C_CORRUPTIONS
from src.corruptions import apply_corruption
from src.grad_cam import grad_cam

SEVERITIES = tuple(range(6))
CORRUPTIONS = ('gaussian_noise', 'brightness')
N_PER_CLASS = 5
SEED = 42
IG_STEPS = 64
EPS = 1e-8
PENALTY_P = 1
TRANSPORT_SIZE = 8  # 8x8 OT approximation, matching the prior Grad-CAM pilot


def select_clean_correct_images(loader, model, device, n_per_class=N_PER_CLASS, split="development", all_clean_correct=False):
    """Choose exactly five clean-correct development images for each class."""
    development = loader.get_split(split)
    selection_audit = []
    rng = np.random.default_rng(SEED)
    by_class = defaultdict(list)
    for row in development:
        label = row.get('class_id', row.get('true_label', row.get('label')))
        if label is None:
            raise ValueError('Development rows must include class_id, true_label, or label')
        by_class[int(label)].append(row)
    selected = []
    for class_id in range(10):
        candidates = by_class[class_id]
        order = rng.permutation(len(candidates))
        found = 0
        for idx in order:
            row = candidates[int(idx)]
            image_id = int(row['image_id'])
            source = row.get('source', 'train')
            x = loader.get_image(image_id=image_id, source=source, device=device)
            with torch.no_grad():
                prediction = int(model(x).argmax(dim=1).item())
            selection_audit.append(dict(image_id=image_id, source=source, true_label=class_id,
                                        predicted_class=prediction, clean_correct=int(prediction==class_id)))
            if prediction != class_id:
                continue
            selected.append({'image_id': image_id, 'source': source, 'true_label': class_id})
            found += 1
            if not all_clean_correct and found == n_per_class:
                break
        if not all_clean_correct and found != n_per_class:
            raise RuntimeError(f'Class {class_id}: only {found} clean-correct development images found')
    select_clean_correct_images.audit = selection_audit
    print(f'Selected {len(selected)} clean-correct {split} images; all_clean_correct={all_clean_correct}.')
    return selected


def transport_cost(n):
    yy, xx = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing='ij')
    coords = np.stack((yy.ravel(), xx.ravel()), axis=1)
    return ot.dist(coords, coords, metric='euclidean')


COST = transport_cost(TRANSPORT_SIZE)


def asws_components(clean, altered):
    A = np.asarray(clean, dtype=np.float64)
    B = np.asarray(altered, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError('Attribution maps must have the same 2D shape')
    if not np.isfinite(A).all() or not np.isfinite(B).all() or (A < 0).any() or (B < 0).any():
        raise ValueError('Grad-CAM must be finite and nonnegative')
    clean_max = A.max()
    if clean_max <= EPS:
        # No meaningful clean attribution: flag rather than fabricate a score.
        return dict(loss=np.nan, gain=np.nan, wasserstein=np.nan,
                    retention=np.nan, weighted_wasserstein=np.nan, valid_cam=False)
    a, b = A / clean_max, B / clean_max
    weights = a / (a.sum() + EPS)
    loss = np.sum(weights * np.maximum(0, a - b) ** PENALTY_P)
    novelty = np.clip(1 - a / (a.max() + EPS), 0, 1)
    gain = np.mean(novelty * np.maximum(0, b - a) ** PENALTY_P)
    mass_a, mass_b = float(a.sum()), float(b.sum())
    retention = min(1.0, mass_b / (mass_a + EPS))
    # If the target CAM vanishes, the spatial comparison is undefined.
    if mass_b <= 1e-6 * a.size:
        w1 = 0.0
    else:
        def coarse_distribution(arr):
            t = torch.from_numpy(arr).float()[None, None]
            small = F.interpolate(t, size=(TRANSPORT_SIZE, TRANSPORT_SIZE),
                                  mode='area').squeeze().numpy().astype(np.float64)
            small = np.maximum(small, 0)
            return (small / small.sum()).ravel()
        w1 = float(ot.emd2(coarse_distribution(a), coarse_distribution(b), COST))
    return dict(loss=float(loss), gain=float(gain), wasserstein=w1,
                retention=retention, weighted_wasserstein=retention * w1,
                valid_cam=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)



def compute_ig(ig, x, target, steps=IG_STEPS):
    """Nonnegative 2D IG magnitude: sum absolute RGB contributions per pixel.

    Signed IG must not be fed directly to the nonnegative ASWS formula.
    A fixed zero baseline and clean predicted class are used at all severities.
    """
    with torch.enable_grad():
        attribution = ig.attribute(
            x.detach().clone().requires_grad_(True),
            baselines=torch.zeros_like(x),
            target=target,
            n_steps=steps,
            method='gausslegendre',
            internal_batch_size=8,
        )
    return attribution.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()


def compute_cam(model, x, target, layer, reshape_transform=None):
    result = grad_cam(model, x, target, layer, normalize=False, reshape_transform=reshape_transform)
    if isinstance(result, torch.Tensor):
        result = result.detach().squeeze().cpu().numpy()
    return np.asarray(result, dtype=np.float64)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['custom','cifar10c'],default='custom')
    parser.add_argument('--corruption-dir',type=Path,default=Path('data/CIFAR-10-C'))
    parser.add_argument('--manifest', type=Path, default=Path('splits/split_manifest_v2.csv'))
    parser.add_argument('--split', choices=['development', 'test'], default='development')
    parser.add_argument('--corruptions', nargs='+', choices=sorted(set(CIFAR10C_CORRUPTIONS)|{'blur'}), default=list(CORRUPTIONS))
    parser.add_argument('--all-clean-correct', action='store_true')
    parser.add_argument('--n-per-class', type=int, default=N_PER_CLASS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deadline', type=float, help='Unix deadline; checkpoints and exits 75 at an image boundary.')
    parser.add_argument('--save-maps', action='store_true')
    parser.add_argument('--max-images', type=int, help='Limit selected images for a smoke run only.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--classifier', choices=ARCHITECTURES, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mc-passes', type=int, help=argparse.SUPPRESS)  # Compatibility with an already-running launcher; unused.
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--odin-temperature', type=float, default=1000.0)
    parser.add_argument('--ig-steps', type=int, choices=[1, 2, 4, 8, 16, 32, 64, 128], default=IG_STEPS)
    parser.add_argument('--legacy-ig-steps-50', dest='ig_steps', action='store_const', const=50, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--odin-epsilon', type=float, default=0.0014)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("CPU thread count must be positive.")
    torch.set_num_threads(args.cpu_threads)
    if args.odin_temperature <= 0 or args.odin_epsilon < 0:
        parser.error('Invalid ODIN parameters.')
    if not args.checkpoint.exists():
        parser.error(f'Missing trained CIFAR-10 checkpoint: {args.checkpoint}')
    if args.output_dir is None:
        args.output_dir = Path('results/classifier_benchmarks') / args.classifier
    if args.n_per_class < 1 or (args.max_images is not None and args.max_images < 1):
        parser.error('Image counts must be positive.')
    out = args.output_dir / 'images.csv'
    summary_path = args.output_dir / 'summary.csv'
    device = ('mps' if torch.backends.mps.is_available() else
              'cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if args.dataset=='cifar10c' and any(c not in CIFAR10C_CORRUPTIONS for c in args.corruptions):
        parser.error('Official CIFAR-10-C requires official file names (for example gaussian_blur, not blur).')
    if args.dataset=='custom' and any(c not in ['gaussian_noise','brightness','blur'] for c in args.corruptions):
        parser.error('Custom mode only implements noise, brightness and blur.')
    loader = CIFAR10CLoader(args.manifest,args.corruption_dir) if args.dataset=='cifar10c' else CIFARDataLoader(args.manifest)
    load = load_published_classifier if args.checkpoint.is_dir() else load_classifier
    model = load(args.checkpoint, args.classifier, device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    selected = select_clean_correct_images(loader, model, device, args.n_per_class, args.split, args.all_clean_correct)
    if not selected:
        raise ValueError('No clean-correct images available.')
    write_csv(args.output_dir / 'selection_audit.csv', select_clean_correct_images.audit)
    if args.max_images is not None:
        selected = selected[:args.max_images]
    layer, reshape_transform = attribution_layer(model, args.classifier)
    ig = IntegratedGradients(model)
    results = []
    completed=set()
    if out.exists():
        if not args.resume:raise ValueError('Output exists; use --resume or a new directory.')
        import pandas as pd
        previous=pd.read_csv(out)
        if 'dataset' not in previous or set(previous.dataset)!={args.dataset}:raise ValueError('Resume dataset mismatch.')
        if set(previous.ig_steps)!={args.ig_steps} or set(previous.classifier)!={args.classifier} or set(previous.corruption)!=set(args.corruptions):
            raise ValueError('Resume settings mismatch.')
        if set(previous.image_id)-{int(r['image_id']) for r in selected}:raise ValueError('Resume cohort differs.')
        for image,g in previous.groupby('image_id'):
            expected={(c,s,m) for c in args.corruptions for s in SEVERITIES for m in ('gradcam','ig')}
            if set(zip(g.corruption,g.severity,g.method))!=expected or len(g)!=len(expected):raise ValueError('Partial image checkpoint.')
        results=previous.to_dict('records');completed=set(previous.image_id)
    if args.save_maps:(args.output_dir/'maps').mkdir(parents=True,exist_ok=True)

    for n, row in enumerate(selected, start=1):
        image_id = int(row['image_id'])
        source = row['source']
        if image_id in completed:continue
        if args.deadline and time.time()>=args.deadline:
            if results:write_csv(out,results)
            print('Budget reached; collection checkpointed.',flush=True)
            sys.exit(75)
        cached_maps={}
        label_text = str(row.get('true_label', '')).strip()
        true_label = int(label_text) if label_text else None
        clean = loader.get_image(image_id=image_id, source=source, device=device)
        with torch.no_grad():
            clean_class = int(model(clean).argmax(1).item())
        clean_maps = {
            'gradcam': compute_cam(model, clean, clean_class, layer, reshape_transform),
            'ig': compute_ig(ig, clean, clean_class, args.ig_steps),
        }

        noise_seed = SEED + image_id + (50000 if source == 'test' else 0)
        for corruption in args.corruptions:
            for severity in SEVERITIES:
                distorted = (loader.get_corrupted_image(image_id,source,corruption,severity,device)
                    if args.dataset=='cifar10c' else clean if severity==0 else apply_corruption(
                        clean, corruption=corruption, severity=severity, seed=noise_seed))
                with torch.no_grad():
                    logits = model(distorted)
                    probs = logits.softmax(dim=1)[0]
                    pred = int(probs.argmax().item())
                    msp = float(probs.max().item())
                    target_conf = float(probs[clean_class].item())

                _, baseline = benchmark_scores(model, distorted,
                                               args.odin_temperature, args.odin_epsilon)
                maps = clean_maps if severity == 0 else {
                    'gradcam': compute_cam(model, distorted, clean_class, layer, reshape_transform),
                    'ig': compute_ig(ig, distorted, clean_class, args.ig_steps),
                }
                if args.save_maps:
                    for map_method,map_value in maps.items():cached_maps[f'{corruption}_{severity}_{map_method}']=map_value
                for method in ('gradcam', 'ig'):
                    A, B = clean_maps[method], maps[method]
                    valid = bool(np.isfinite(A).all() and np.isfinite(B).all())
                    if valid:
                        scores = asws_components(A, B)
                    else:
                        scores = dict(loss=np.nan, gain=np.nan, wasserstein=np.nan,
                                      retention=np.nan, weighted_wasserstein=np.nan,
                                      valid_cam=False)
                    results.append(dict(
                        image_id=image_id, source=source,dataset=args.dataset,
                        true_label='' if true_label is None else true_label,
                        clean_class=clean_class, corruption=corruption, severity=severity, method=method,
                        seed=SEED, noise_seed=noise_seed if args.dataset=='custom' else '', ig_steps=args.ig_steps,
                        classifier=args.classifier, checkpoint=str(args.checkpoint), msp=msp,
                        odin=float(baseline['odin'][0]),
                        odin_temperature=args.odin_temperature, odin_epsilon=args.odin_epsilon,
                        predicted_class=pred,
                        prediction_changed=int(pred != clean_class),
                        correct='' if true_label is None else int(pred == true_label),
                        clean_correct='' if true_label is None else int(clean_class == true_label),
                        target_confidence=target_conf,
                        loss=scores['loss'], gain=scores['gain'],
                        wasserstein=scores['wasserstein'],
                        retention=scores['retention'],
                        weighted_wasserstein=scores['weighted_wasserstein'],
                        valid_map=scores['valid_cam'],
                    ))
        if args.save_maps:np.savez_compressed(args.output_dir/'maps'/f'{source}_{image_id}.npz',**cached_maps)
        print(f'Completed {n}/{len(selected)}: image {image_id}', flush=True)
        if n % 5 == 0:
            write_csv(out, results)

    write_csv(out, results)
    summary = []
    for corruption in args.corruptions:
        for method in ('gradcam', 'ig'):
            for severity in SEVERITIES:
                subset = [r for r in results if r['method'] == method and r['severity'] == severity and r['corruption'] == corruption]
                valid = [r for r in subset if r['valid_map']]
                known = [r for r in subset if r['correct'] != '']
                def mean_valid(key):
                    return float(np.mean([r[key] for r in valid])) if valid else float('nan')
                summary.append(dict(
                    method=method, corruption=corruption, severity=severity, n_images=len(subset),
                    n_valid_maps=len(valid),
                    accuracy=float(np.mean([r['correct'] for r in known])) if known else '',
                    prediction_flip_rate=float(np.mean([r['prediction_changed'] for r in subset])),
                    mean_msp=float(np.mean([r['msp'] for r in subset])),
                    mean_odin=float(np.mean([r['odin'] for r in subset])),
                    mean_target_confidence=float(np.mean([r['target_confidence'] for r in subset])),
                    mean_loss=mean_valid('loss'), mean_gain=mean_valid('gain'),
                    mean_wasserstein=mean_valid('wasserstein'),
                    mean_retention=mean_valid('retention'),
                    mean_weighted_wasserstein=mean_valid('weighted_wasserstein'),
                ))
    write_csv(summary_path, summary)
    print('\nSUMMARY')
    for r in summary:
        print(f"{r['corruption']:14} {r['method']:7} severity={r['severity']} "
              f"valid={r['n_valid_maps']}/{r['n_images']} "
              f"loss={r['mean_loss']:.5f} gain={r['mean_gain']:.5f} "
              f"weighted_W1={r['mean_weighted_wasserstein']:.5f}")
    print('\nSaved:', out, 'and', summary_path)
    print('These are raw attribution metrics, not calibrated failure warnings.')


if __name__ == '__main__':
    main()
