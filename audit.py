"""PixelLab Audit — 9 checks + smoke test"""
import re, os, sys
sys.path.insert(0, '.')

print("=" * 60)
print("PIXELLAB AUDIT")
print("=" * 60)

# Check 1: Stubs
print("\n--- CHECK 1: Stub Detector ---")
stub_count = 0
for f in ['core/processor.py', 'config.py', 'app.py']:
    if os.path.exists(f):
        with open(f, encoding='utf-8') as fh:
            for i, line in enumerate(fh, 1):
                if re.search(r'^\s*pass\s*$|TODO|NotImplementedError', line):
                    print(f"  HIT: {f}:{i}: {line.strip()}")
                    stub_count += 1
    else:
        print(f"  MISSING: {f}")
        stub_count += 1
print(f"  Result: {stub_count} issues")

# Check 2: Op count
print("\n--- CHECK 2: Operation Count ---")
try:
    from config import OPERATIONS
    print(f"  Count: {len(OPERATIONS)}")
    for k in sorted(OPERATIONS.keys()):
        print(f"    {k}: {OPERATIONS[k].get('category', '?')}")
except Exception as e:
    print(f"  FAIL: {e}")

# Check 3: Noise seeding
print("\n--- CHECK 3: Noise Seeding ---")
with open('core/processor.py', encoding='utf-8') as fh:
    content = fh.read()
    hits = content.count('default_rng')
    print(f"  default_rng hits: {hits} (need >= 2)")

# Check 4: FFT fix
print("\n--- CHECK 4: FFT rfft2 ---")
hits = len(re.findall(r'rfft2|irfft2|complex64', content))
print(f"  rfft2/irfft2/complex64 hits: {hits} (need >= 2)")

# Check 5: Headless
print("\n--- CHECK 5: opencv-python-headless ---")
with open('requirements.txt', encoding='utf-8') as fh:
    for line in fh:
        if 'opencv' in line.lower():
            print(f"  {line.strip()}")

# Check 6: Scratch purity
print("\n--- CHECK 6: Scratch Purity ---")
with open('core/processor.py', encoding='utf-8') as fh:
    lines = fh.read().split('\n')
in_scratch = False
scratch_fails = 0
for i, line in enumerate(lines, 1):
    if '_scratch_' in line and 'def ' in line:
        in_scratch = True
    elif in_scratch and line.strip().startswith('def '):
        in_scratch = False
    elif in_scratch and 'cv2.' in line:
        print(f"  FAIL: cv2 in scratch at line {i}: {line.strip()}")
        scratch_fails += 1
print(f"  Scratch purity: {scratch_fails} cv2 calls in scratch methods")

# Check 7: Shape guards
print("\n--- CHECK 7: Shape Guards ---")
with open('config.py', encoding='utf-8') as fh:
    cfg = fh.read()
has_input = "'input'" in cfg
has_output = "'output'" in cfg
print(f"  config.py 'input' metadata: {has_input}")
print(f"  config.py 'output' metadata: {has_output}")
has_ensure = '_ensure_shape' in content
print(f"  processor.py _ensure_shape: {has_ensure}")

# Check 8: Import/requirements match
print("\n--- CHECK 8: Import/Requirements Match ---")
req_pkgs = set()
with open('requirements.txt') as fh:
    for line in fh:
        line = line.strip()
        if line and not line.startswith('#') and '==' in line:
            pkg = line.split('==')[0].lower().replace('-', '_')
            req_pkgs.add(pkg)

code_imports = set()
for f in ['core/processor.py', 'app.py']:
    if os.path.exists(f):
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                m = re.match(r'^(?:import|from)\s+(\w+)', line)
                if m:
                    code_imports.add(m.group(1).lower())

stdlib = {'os', 'time', 'uuid', 'io', 're', 'sys', 'json', 'pathlib', 'typing', 'dataclasses'}
project = {'pixellab', 'core', 'config'}
external = code_imports - stdlib - project - {'flask', 'jinja2', 'werkzeug', 'markupsafe', 'blinker', 'click', 'itsdangerous'}
print(f"  External imports: {sorted(external)}")
print(f"  Requirements: {sorted(req_pkgs)}")
missing = external - req_pkgs
if missing:
    print(f"  MISSING from requirements: {missing}")
else:
    print(f"  All imports covered")

# Check 9: Theory content
print("\n--- CHECK 9: Theory Content ---")
from config import THEORY
for k, v in THEORY.items():
    clen = len(str(v))
    status = "OK" if clen > 100 else "TOO SHORT"
    print(f"  {k}: {clen} chars [{status}]")

print("\n" + "=" * 60)
print("SMOKE TEST: All operations on color + gray images")
print("=" * 60)

import numpy as np
from core.processor import ImageProcessor

color = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
gray = (np.random.rand(300, 400) * 255).astype(np.uint8)

passes = 0
fails = 0
for name, meta in OPERATIONS.items():
    try:
        if meta.get('visual_only'):
            ImageProcessor.apply_operation(color, name, meta.get('params', {}))
            passes += 1
            print(f"  PASS: {name}")
            continue

        # Test with color image
        out = ImageProcessor.apply_operation(color.copy(), name, meta.get('params', {}))
        assert out is not None and out.size > 0, "empty output"

        # Test with gray image
        out_g = ImageProcessor.apply_operation(gray.copy(), name, meta.get('params', {}))
        assert out_g is not None and out_g.size > 0, "empty output"

        passes += 1
        print(f"  PASS: {name}")
    except Exception as e:
        fails += 1
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")

# Noise determinism test
print("\n--- NOISE DETERMINISM ---")
a = ImageProcessor.apply_operation(color.copy(), 'gaussian_noise', {'sigma': 20, 'seed': 42})
b = ImageProcessor.apply_operation(color.copy(), 'gaussian_noise', {'sigma': 20, 'seed': 42})
if np.array_equal(a, b):
    print("  PASS: Gaussian noise deterministic with same seed")
else:
    print("  FAIL: Gaussian noise NOT deterministic")
    fails += 1

a = ImageProcessor.apply_operation(color.copy(), 'salt_pepper', {'amount': 0.05, 'seed': 42})
b = ImageProcessor.apply_operation(color.copy(), 'salt_pepper', {'amount': 0.05, 'seed': 42})
if np.array_equal(a, b):
    print("  PASS: Salt & pepper deterministic with same seed")
else:
    print("  FAIL: Salt & pepper NOT deterministic")
    fails += 1

# Shape guard test
print("\n--- SHAPE GUARDS ---")
try:
    ImageProcessor.apply_operation(color.copy(), 'canny', {'threshold1': 100, 'threshold2': 200})
    print("  PASS: Canny on color (auto-converts)")
except Exception as e:
    print(f"  FAIL: Canny on color: {e}")
    fails += 1

try:
    ImageProcessor.apply_operation(gray.copy(), 'hue_saturation', {'hue_shift': 10, 'sat_factor': 1.2})
    print("  PASS: HueSat on gray (auto-converts)")
except Exception as e:
    print(f"  FAIL: HueSat on gray: {e}")
    fails += 1

print(f"\n{'='*60}")
print(f"TOTAL: {passes} passed, {fails} failed out of {passes+fails}")
print(f"{'='*60}")
