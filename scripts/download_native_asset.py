"""Download one public project evidence capsule with an explicitly trusted SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_asset_parts import join, validate_manifest


def release_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'github.com' or parsed.query or parsed.fragment
            or not re.fullmatch(r'/Leo-7GOAT/Intelligent-driving-system/releases/download/'
                                r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.parts\.json', parsed.path)):
        raise ValueError('Require a public HTTPS parts manifest from this project release')
    return url.rsplit('/', 1)


def read_bounded(url, maximum, deadline):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Download wall budget exhausted')
    request = Request(url, headers={'User-Agent': 'native-evidence-download/1'})
    chunks, size = [], 0
    with urlopen(request, timeout=min(60., remaining)) as response:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Download wall budget exhausted')
            chunk = response.read(min(65536, maximum-size+1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise ValueError('Response exceeds its declared size bound')
    return b''.join(chunks)


def download(manifest_url, directory, expected_sha256, wall_seconds=1200):
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Require the trusted archive SHA256 from the published verification report')
    if type(wall_seconds) not in (int, float) or not 1 <= wall_seconds <= 1200:
        raise ValueError('Download wall budget must be1..1200seconds')
    base, manifest_name = release_url(manifest_url)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('Use a fresh output directory; interrupted evidence is preserved')
    started = time.monotonic()
    deadline = started+wall_seconds
    payload = read_bounded(manifest_url, 1024**2, deadline)
    manifest = json.loads(payload)
    parts = validate_manifest(manifest, expected_sha256)
    archive_name = manifest.get('archive')
    if (not isinstance(archive_name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip', archive_name)
            or manifest_name != archive_name+'.parts.json'
            or any(part['name'] != archive_name+f'.part{index:03d}'
                   for index, part in enumerate(parts, 1))):
        raise ValueError('Archive and ordered part names differ from the transfer format')
    ancestor = directory.resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 2*manifest['bytes']+64*1024**2:
        raise OSError('Insufficient free space for checked parts and assembled archive')
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory/manifest_name
    with manifest_path.open('xb') as stream:
        stream.write(payload)

    def fetch(part):
        data = read_bounded(base+'/'+part['name'], part['bytes'], deadline)
        temporary = directory/(part['name']+'.partial')
        with temporary.open('xb') as stream:
            stream.write(data)
        if len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
            raise ValueError('Downloaded part differs: '+part['name'])
        temporary.rename(directory/part['name'])

    with ThreadPoolExecutor(max_workers=min(3, len(parts))) as workers:
        futures = [workers.submit(fetch, part) for part in parts]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    result = join(manifest_path, directory/archive_name, expected_sha256)
    result.update(manifest_url=manifest_url, archive=archive_name,
                  wall_seconds=time.monotonic()-started, new_controls=0, optimizer_updates=0)
    with (directory/'download_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-url', required=True)
    parser.add_argument('--output', required=True, help='Fresh directory for checked parts and assembled ZIP')
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--wall-seconds', type=float, default=1200)
    args = parser.parse_args()
    print(json.dumps(download(args.manifest_url, args.output, args.sha256, args.wall_seconds)))
