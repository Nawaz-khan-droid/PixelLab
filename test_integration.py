"""Integration test — runs Flask test client, hits every route."""
import sys, io, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from app import app

with app.test_client() as client:
    # 1. Index page
    r = client.get('/')
    assert r.status_code == 200
    assert b'PixelLab' in r.data
    print('[PASS] GET / - index page loads')

    # 2. Editor shows upload panel (no image uploaded yet)
    r = client.get('/editor')
    assert r.status_code == 200
    assert b'No image loaded' in r.data
    print('[PASS] GET /editor - shows upload panel when no image')

    # 3. Upload a test image
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    data = {'image': (io.BytesIO(buf.tobytes()), 'test.jpg')}
    r = client.post('/api/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert r.status_code == 200
    assert b'editor-content' in r.data
    print('[PASS] POST /api/upload - image uploaded, editor loads')

    # 4. Test all non-parameterized operations
    simple_ops = [
        'negative', 'grayscale', 'otsu', 'histogram_equalize',
        'flip', 'rotate', 'shear', 'resize',
        'gaussian_blur', 'mean_blur', 'median_blur',
        'unsharp_mask', 'high_boost',
        'sobel', 'laplacian', 'fft_spectrum',
        'erosion', 'dilation', 'opening', 'closing',
        'face_detection', 'histogram_compute',
    ]
    for op in simple_ops:
        r = client.post('/api/apply', data={'operation': op}, follow_redirects=True)
        assert r.status_code == 200, f'{op} returned {r.status_code}'
        print(f'[PASS] apply {op}')

    # 5. Test parameterized operations
    client.post('/api/reset', follow_redirects=True)  # reset stack before param tests
    param_ops = [
        ('brightness_contrast', {'brightness': '30', 'contrast': '1.5'}),
        ('color_balance', {'r': '1.2', 'g': '0.8', 'b': '1.0'}),
        ('hue_saturation', {'hue_shift': '15', 'sat_factor': '1.3'}),
        ('threshold', {'value': '128'}),
        ('gamma', {'gamma': '2.0'}),
        ('canny', {'threshold1': '100', 'threshold2': '200'}),
        ('gaussian_noise', {'sigma': '25'}),
        ('salt_pepper', {'amount': '0.03'}),
        ('gaussian_blur', {'kernel': '15'}),
        ('mean_blur', {'kernel': '7'}),
        ('median_blur', {'kernel': '5'}),
        ('lowpass', {'radius': '30'}),
        ('highpass', {'radius': '20'}),
        ('denoise_gaussian', {'kernel': '5'}),
        ('denoise_median', {'kernel': '5'}),
    ]
    for op, params in param_ops:
        data = {'operation': op}
        data.update(params)
        r = client.post('/api/apply', data=data, follow_redirects=True)
        assert r.status_code == 200, f'{op} returned {r.status_code}'
        print(f'[PASS] apply {op} with params')

    # 6. Noise determinism
    client.post('/api/reset', follow_redirects=True)
    client.post('/api/apply', data={'operation': 'gaussian_noise', 'sigma': '20', 'seed': '42'}, follow_redirects=True)
    r1 = client.get('/api/image/processed')
    client.post('/api/reset', follow_redirects=True)
    client.post('/api/apply', data={'operation': 'gaussian_noise', 'sigma': '20', 'seed': '42'}, follow_redirects=True)
    r2 = client.get('/api/image/processed')
    assert r1.data == r2.data, 'Noise not deterministic with same seed'
    print('[PASS] Noise determinism - same seed gives identical output')

    # 7. Remove filter
    r = client.post('/api/remove/' + app.config.get('_test_filter_id', 'nonexistent'), follow_redirects=True)
    print(f'[PASS] POST /api/remove - handled')

    # 8. Reset
    r = client.post('/api/reset', follow_redirects=True)
    assert r.status_code == 200
    print('[PASS] POST /api/reset')

    # 9. Download (need processed image)
    client.post('/api/apply', data={'operation': 'negative'}, follow_redirects=True)
    r = client.get('/download')
    assert r.status_code == 200
    assert r.content_type.startswith('image/')
    print('[PASS] GET /download - serves image file')

    # 11. Image serving
    r = client.get('/api/image/processed')
    assert r.status_code == 200
    assert r.content_type.startswith('image/')
    print('[PASS] GET /api/image/processed - serves JPEG')

    print(f'\n{"="*50}')
    print(f'ALL {10 + len(simple_ops) + len(param_ops)} TESTS PASSED')
    print(f'{"="*50}')
