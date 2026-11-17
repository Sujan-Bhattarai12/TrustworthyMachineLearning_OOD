"""Run all three CIFAR-10 classifiers, then compare warning scores per classifier.

Uses pinned published CIFAR-10 checkpoints downloaded with scripts.download_hf_classifiers.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from src.classifiers import ARCHITECTURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/huggingface'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/classifier_benchmarks'))
    parser.add_argument('--n-per-class', type=int, default=5)
    parser.add_argument('--odin-temperature', type=float, default=1000.0)
    parser.add_argument('--ig-steps',type=int,choices=[1,2,4,8,16,32,64,128],default=64)
    parser.add_argument('--odin-epsilon', type=float, default=0.0014)
    args = parser.parse_args()
    missing = [str(args.checkpoint_dir / name) for name in ARCHITECTURES
               if not (args.checkpoint_dir / name / 'provenance.json').is_file()]
    if missing:
        parser.error('Missing trained CIFAR-10 checkpoints: ' + ', '.join(missing))
    for name in ARCHITECTURES:
        out = args.output_dir / name
        subprocess.run([sys.executable, '-m', 'scripts.collect_attribution_metrics',
                        '--classifier', name, '--checkpoint', str(args.checkpoint_dir / name),
                        '--output-dir', str(out), '--n-per-class', str(args.n_per_class),
                        '--odin-temperature', str(args.odin_temperature),
                        '--odin-epsilon', str(args.odin_epsilon),
                        '--ig-steps', str(args.ig_steps)], check=True)
        subprocess.run([sys.executable, '-m', 'scripts.evaluate_warning_models',
                        '--input', str(out / 'images.csv'),
                        '--output-dir', str(out / 'comparison')], check=True)


if __name__ == '__main__':
    main()
