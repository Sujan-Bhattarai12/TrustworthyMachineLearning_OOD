"""Download only model config, preprocessing, card and weights at pinned revisions."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('configs/hf_classifiers.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints/huggingface'))
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    for architecture, spec in registry.items():
        target = args.output_dir / architecture
        target.mkdir(parents=True, exist_ok=True)
        provenance_path = target / 'provenance.json'
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text())
            if any(previous.get(key) != spec[key] for key in ('repo_id', 'revision', 'weights')):
                raise ValueError(f'{target}: existing files belong to another checkpoint; use a new output directory.')
        files = ['config.json', 'README.md', spec['weights']]
        if spec['backend'] == 'transformers':
            files.append('preprocessor_config.json')
        checksums = {}
        for name in files:
            destination = target / name
            if not destination.exists():
                url = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{name}"
                temporary = target / (name + '.partial')
                print(f'Downloading {architecture}/{name}', flush=True)
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(destination)
            digest = hashlib.sha256()
            with destination.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            checksums[name] = digest.hexdigest()
        (target / 'provenance.json').write_text(json.dumps(dict(spec, sha256=checksums), indent=2))
        print(f'Ready: {architecture}', flush=True)


if __name__ == '__main__':
    main()
