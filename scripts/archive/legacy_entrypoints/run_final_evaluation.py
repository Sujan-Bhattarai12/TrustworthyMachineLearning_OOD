"""Collect reserved blur scores with frozen settings and evaluate without refitting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('results/frozen_warning_methods'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/final_blur_evaluation'))
    args=parser.parse_args()
    for name in ['resnet50','vit_b_16','swin_t']:
        artifact_path=args.artifacts/f'{name}.json'
        frozen=json.loads(artifact_path.read_text())
        provenance=json.loads((Path('checkpoints/huggingface')/name/'provenance.json').read_text())
        if frozen['provenance']!=provenance:
            raise ValueError('Checkpoint provenance changed since freezing.')
        if frozen['settings']['ig_steps'] not in [1,2,4,8,16,32,50,64,128]:
            raise ValueError('Runner integration requires frozen IG steps = 50.')
    for name in ['resnet50','vit_b_16','swin_t']:
        artifact_path=args.artifacts/f'{name}.json'
        frozen=json.loads(artifact_path.read_text())
        out=args.output_dir/name
        if (out/'images.csv').exists():
            raise ValueError(f'Refusing to overwrite final collection: {out}')
        subprocess.run([sys.executable,'-u','-m','scripts.collect_attribution_metrics',
                        '--classifier',name,'--checkpoint',str(Path('checkpoints/huggingface')/name),
                        '--split','test','--corruptions','blur','--all-clean-correct',
                        '--output-dir',str(out),'--odin-temperature',str(frozen['settings']['odin_temperature']),
                        '--odin-epsilon',str(frozen['settings']['odin_epsilon'])]+(['--legacy-ig-steps-50'] if frozen['settings']['ig_steps']==50 else ['--ig-steps',str(int(frozen['settings']['ig_steps']))]),check=True)
        subprocess.run([sys.executable,'-u','-m','scripts.evaluate_frozen_warnings',
                        '--artifact',str(artifact_path),'--input',str(out/'images.csv'),
                        '--output-dir',str(out/'evaluation')],check=True)


if __name__=='__main__':
    main()
