"""
Download pretrained ONNX models for the Live feature.

Usage:
    python scripts/download_models.py

Models are stored in models/ (gitignored). Downloaded at build time on Render
via the build command so the free tier cold start stays fast.
"""
import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

ENABLE_DEEPFAKE = os.environ.get('DD_DOWNLOAD', '0') == '1'

URLS = {
    'yolov8n.onnx': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx',
    # Optional real CNN deepfake detector. Set DD_DOWNLOAD=1 and point this at a
    # 224x224 ONNX classifier whose output is [original, deepfake] (sigmoid/softmax).
    # Leave unset to ship in heuristic mode.
    'deepfake_lite.onnx': os.environ.get('DEEPFAKE_ONNX_URL', ''),
}


def _download(url, dest, fsize, bar_len=40):
    print("  downloading %s ..." % os.path.basename(dest))
    req = urllib.request.Request(url, headers={'User-Agent': 'PixelLab-model-download/1.0'})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, 'wb') as out:
        total = int(resp.headers.get('Content-Length') or fsize)
        done = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            pct = int(done / total * 100) if total else 0
            filled = int(bar_len * done / total) if total else bar_len
            print('\r[%s%s] %3d%%' % ('=' * filled, ' ' * (bar_len - filled), pct), end='', flush=True)
    print()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in URLS.items():
        dest = os.path.join(MODELS_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print("[skip] %s already present" % name)
            continue
        if not url:
            print("[skip] %s -- no URL configured" % name)
            continue
        try:
            _download(url, dest, fsize=0)
            print("[ok]   %s (%.1f MB)" % (name, os.path.getsize(dest) / 1e6))
        except Exception as e:
            print("[fail] %s: %s" % (name, e))
            sys.exit(1)
    print("\nDone. Model folder:", MODELS_DIR)


if __name__ == '__main__':
    main()