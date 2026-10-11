"""Download verified SmolLM2-135M base weights and run the 2-of-5 experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.request
import urllib.error

from demo import run

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M'
FILENAME = 'model.safetensors'
DEFAULT_CACHE = Path('models/SmolLM2-135M')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    if manifest.get('model_id') != MODEL_ID or manifest.get('filename') != FILENAME:
        raise ValueError('manifest must identify the SmolLM2-135M base model weights')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('revision', '')):
        raise ValueError('manifest requires an immutable 40-character revision')
    if not re.fullmatch(r'[0-9a-f]{64}', manifest.get('sha256', '')):
        raise ValueError('manifest requires a SHA-256 digest')
    if type(manifest.get('size_bytes')) is not int or manifest['size_bytes'] <= 0:
        raise ValueError('manifest requires a positive file size')


def verify_file(path, manifest):
    validate_manifest(manifest)
    path = Path(path)
    if path.stat().st_size != manifest['size_bytes'] or file_hash(path) != manifest['sha256']:
        raise ValueError('model size or SHA-256 does not match the manifest')


def download(cache=DEFAULT_CACHE, revision='main'):
    """Resolve a revision once, download through HTTPS, verify the HF LFS digest."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', revision):
        raise ValueError('invalid revision')
    url = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    entry = next(item for item in metadata['siblings'] if item['rfilename'] == FILENAME)
    lfs = entry['lfs']
    manifest = {'model_id': MODEL_ID, 'filename': FILENAME, 'revision': metadata['sha'],
                'sha256': lfs['sha256'], 'size_bytes': lfs['size'],
                'source': 'Hugging Face HTTPS API LFS metadata'}
    validate_manifest(manifest)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / FILENAME
    manifest_path = cache / 'manifest.json'
    # Preserve a previously downloaded version rather than silently replace it.
    if target.exists():
        verify_file(target, manifest)
    else:
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{manifest["revision"]}/{FILENAME}'
        fd, temporary = tempfile.mkstemp(prefix='weights-', suffix='.part', dir=cache)
        try:
            with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            verify_file(temporary, manifest)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return target, manifest_path


def experiment(model_file, manifest_path, output):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    verify_file(model_file, manifest)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty; choose a new --output directory')
    output.mkdir(parents=True, exist_ok=True)
    data = Path(model_file).read_bytes()
    # Check exactly the bytes used by the experiment, not only the earlier file read.
    if len(data) != manifest['size_bytes'] or hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise ValueError('model changed after verification')
    def save_first_pair(pair, restored):
        if tuple(s.index for s in pair) == (1, 2):
            (output / 'recovered.safetensors').write_bytes(restored)
    start = time.perf_counter()
    package, report = run(data, on_recovered=save_first_pair)
    # Binary output avoids doubling a model-sized ciphertext through hex JSON.
    (output / 'encrypted.bin').write_bytes(package.ciphertext)
    envelope = {'metadata': json.loads(package.metadata()), 'nonce': package.nonce.hex(),
                'tag': package.tag.hex(), 'ciphertext_file': 'encrypted.bin'}
    (output / 'envelope.json').write_text(json.dumps(envelope, indent=2) + '\n', encoding='utf-8')
    report['model'] = manifest
    report['recovered_file_sha256'] = file_hash(output / 'recovered.safetensors')
    if report['recovered_file_sha256'] != manifest['sha256']:
        raise RuntimeError('saved recovered file failed integrity check')
    report['experiment_seconds'] = time.perf_counter() - start
    report['model_inference_performed'] = False
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    get = sub.add_parser('download', help='download official base weights and record their digest')
    get.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    get.add_argument('--revision', default='main', help='branch/tag or exact HF commit')
    run_parser = sub.add_parser('run', help='run offline with downloaded weights and manifest')
    run_parser.add_argument('--model-file', type=Path, default=DEFAULT_CACHE / FILENAME)
    run_parser.add_argument('--manifest', type=Path, default=DEFAULT_CACHE / 'manifest.json')
    run_parser.add_argument('--output', type=Path, default=Path('artifacts/smollm2'))
    args = parser.parse_args()
    try:
        if args.command == 'download':
            target, manifest_path = download(args.cache_dir, args.revision)
            print(f'Verified weights: {target}\nManifest: {manifest_path}')
        else:
            report = experiment(args.model_file, args.manifest, args.output)
            print(json.dumps(report, indent=2))
    except (urllib.error.URLError, TimeoutError):
        parser.exit(1, 'Hugging Face download failed. Check HTTPS access to huggingface.co and its download CDN; TLS verification must remain enabled.\n')
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
