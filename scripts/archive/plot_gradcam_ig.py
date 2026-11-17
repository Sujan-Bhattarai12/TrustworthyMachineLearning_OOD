"""Create a slide-ready Grad-CAM / Integrated Gradients severity comparison.

From the P01 project root:
    pip install captum matplotlib
    python -m scripts.plot_gradcam_ig_severity --image-id 7

Or omit --image-id to select an image from the existing 50-image pilot
that changes prediction (preferring an early flip). Uses the same
CIFAR-10 loader, RobustBench checkpoint, and Gaussian corruption as P01.

Outputs:
    results/figures/gradcam_ig_severity_<image_id>.png
    results/figures/gradcam_ig_severity_<image_id>.pdf
    results/figures/gradcam_ig_severity_<image_id>.csv
"""
import argparse
import random
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from captum.attr import IntegratedGradients
from robustbench.utils import load_model

from src.data_loader import CIFARDataLoader
from src.corruptions import apply_corruption
from src.grad_cam import grad_cam

CLASS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
SEVERITIES = (0, 1, 2, 3, 4, 5)
SEED = 42


def choose_id(csv_path):
    """Select a pilot image with an early prediction change, if available."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f'{csv_path} not found. Supply --image-id from your development set.')
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    candidates = {}
    for r in rows:
        if int(r['severity']) == 0:
            continue
        if int(r['prediction_changed']) != 1:
            continue
        if r.get('correct', '') == '':
            continue
        key = (int(r['image_id']), r.get('source', 'train'))
        candidates[key] = min(candidates.get(key, 999), int(r['severity']))
    if not candidates:
        raise ValueError('No changed predictions in pilot CSV. Supply --image-id.')
    (image_id, source), first_flip = min(candidates.items(), key=lambda x: (x[1], x[0][0]))
    print(f'Auto-selected image {image_id} ({source}); pilot first flip at severity {first_flip}')
    return image_id, source


def to_image(tensor):
    # Project input is expected to be float RGB in [0, 1].
    arr = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(arr, 0, 1)


def compute_ig(ig, image, target_class, steps):
    # Fixed zero baseline and fixed clean predicted class for every severity.
    # Captum estimates the integral with Gauss-Legendre quadrature.
    with torch.enable_grad():
        attr = ig.attribute(
            image.detach().clone().requires_grad_(True),
            baselines=torch.zeros_like(image),
            target=target_class,
            n_steps=steps,
            method='gausslegendre',
            internal_batch_size=8,
        )
    # Absolute IG aggregated over RGB: strength of attribution, not sign.
    return attr.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-id', type=int, default=None)
    p.add_argument('--source', default='train', choices=['train', 'test'])
    p.add_argument('--steps', type=int, default=32, help='IG integration steps')
    p.add_argument('--pilot-csv', type=Path, default=Path('results/asws_50_images.csv'))
    p.add_argument('--output-dir', type=Path, default=Path('results/figures'))
    args = p.parse_args()
    args.image_id = random.randint(0, 49999)
    print(f"Selected image ID: {args.image_id}")

    if args.image_id is None:
        image_id, source = choose_id(args.pilot_csv)
    else:
        image_id, source = args.image_id, args.source

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Device:', device)
    model = load_model('Standard', model_dir='checkpoints',
                       dataset='cifar10', threat_model='Linf').to(device).eval()
    layer = model.block3.layer[3].conv2
    loader = CIFARDataLoader()
    clean = loader.get_image(image_id=image_id, source=source, device=device)
    ig = IntegratedGradients(model)

    with torch.no_grad():
        clean_class = int(model(clean).argmax(dim=1).item())
    print('Clean prediction:', CLASS_NAMES[clean_class])

    records = []
    for s in SEVERITIES:
        x = clean if s == 0 else apply_corruption(
            clean, corruption='gaussian_noise', severity=s, seed=SEED)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            probs = logits.softmax(dim=1)[0]
            original_class_conf = float(probs[clean_class].item())
            predicted_class_conf = float(probs[pred].item())

        # Raw CAM; normalize only for visualization using clean CAM reference.
        cam = grad_cam(model, x, clean_class, layer, normalize=False)
        if isinstance(cam, torch.Tensor):
            cam = cam.detach().squeeze().cpu().numpy()
        cam = np.asarray(cam, dtype=float)
        attr_ig = compute_ig(ig, x, clean_class, args.steps)
        records.append(dict(severity=s, image=to_image(x), cam=cam, ig=attr_ig,
                            predicted_class=pred, prediction_changed=int(pred != clean_class),
                            original_class_confidence=original_class_conf,
                            predicted_class_confidence=predicted_class_conf))
        print(f's={s}: predicted={CLASS_NAMES[pred]:10s}  '
              f'changed={pred != clean_class}  '
              f'P(clean class)={original_class_conf:.4f}', flush=True)

    # One common reference scale per method across all severity columns.
    # Using clean max preserves attenuation instead of artificially brightening
    # each distorted attribution map. Saturation above the clean max is clipped.
    eps = 1e-12
    cam_scale = max(float(np.max(records[0]['cam'])), eps)
    ig_scale = max(float(np.max(records[0]['ig'])), eps)

    fig, axes = plt.subplots(3, len(records), figsize=(17, 9.2),
                             gridspec_kw={'height_ratios': [1, 1, 1]})
    for j, r in enumerate(records):
        s = r['severity']
        changed = bool(r['prediction_changed'])
        axes[0, j].imshow(r['image'], interpolation='nearest')
        axes[1, j].imshow(np.clip(r['cam'] / cam_scale, 0, 1),
                          cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        axes[2, j].imshow(np.clip(r['ig'] / ig_scale, 0, 1),
                          cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        column_title = 'ORIGINAL IMAGE\nSeverity 0' if s == 0 else f'Noise severity {s}'
        axes[0, j].set_title(
            f'{column_title}\n{CLASS_NAMES[r["predicted_class"]]}'
            + ('  [CHANGED]' if changed else '')
            + f'\nP(clean class)={r["original_class_confidence"]:.2f}',
            fontsize=10, color=('firebrick' if changed else 'black'),
            fontweight=('bold' if changed else 'normal'), pad=8)
        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)
            if changed:
                axes[i, j].add_patch(Rectangle((-.5, -.5), 32, 32,
                                    fill=False, edgecolor='firebrick', linewidth=2.8))

    for i, label in enumerate(('Original / noisy\nimage', 'Grad-CAM', 'Integrated\nGradients')):
        axes[i, 0].set_ylabel(label, fontsize=13, rotation=90, labelpad=16)

    fig.suptitle(f'Attribution under increasing Gaussian noise  |  Image {image_id} '
                 f'|  Clean class: {CLASS_NAMES[clean_class]}', fontsize=15, y=.995)
    fig.text(.5, .02, 'Both attribution methods target the CLEAN predicted class. '
             'Heatmap intensity uses each method’s clean-image scale; '
             'IG shows absolute attribution.', ha='center', fontsize=10)
    fig.subplots_adjust(left=.09, right=.99, top=.86, bottom=.08, wspace=.12, hspace=.25)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f'gradcam_ig_severity_{image_id}'
    fig.savefig(stem.with_suffix('.png'), dpi=220, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)

    with stem.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'source', 'severity',
            'clean_class', 'predicted_class', 'prediction_changed',
            'original_class_confidence', 'predicted_class_confidence'])
        writer.writeheader()
        for r in records:
            writer.writerow(dict(image_id=image_id, source=source,
                severity=r['severity'], clean_class=CLASS_NAMES[clean_class],
                predicted_class=CLASS_NAMES[r['predicted_class']],
                prediction_changed=r['prediction_changed'],
                original_class_confidence=r['original_class_confidence'],
                predicted_class_confidence=r['predicted_class_confidence']))
    print('Saved:', stem.with_suffix('.png'), stem.with_suffix('.pdf'),
          stem.with_suffix('.csv'), sep='\n  ')


if __name__ == '__main__':
    main()
