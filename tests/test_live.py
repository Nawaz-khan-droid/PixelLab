"""
PixelLab Live Feature tests.
Covers: /live page, /api/live-detect, /api/live-deepfake.
"""
import sys
import io
import cv2
import base64
import numpy as np

sys.path.insert(0, 'D:/PixelLab')
from app import app
from core import detection as detection_module


def make_frame(width=320, height=240, seed=7):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode('ascii')


def test_live_page():
    c = app.test_client()
    r = c.get('/live')
    assert r.status_code == 200, r.status_code
    body = r.data.decode()
    assert 'Live Analysis' in body
    assert 'live-detect' in body     # JS targets
    assert 'live-deepfake' in body
    return r


def test_live_detect_ok():
    c = app.test_client()
    r = c.post('/api/live-detect', json={'frame': make_frame()})
    data = r.get_json()
    if r.status_code == 503:
        # Model not present on this machine -- endpoint must still be well-formed
        assert data.get('error') == 'model_not_available'
        return
    assert r.status_code == 200, (r.status_code, data)
    assert 'objects' in data and 'count' in data and 'inference_ms' in data
    for o in data['objects']:
        assert set(o) == {'label', 'confidence', 'x', 'y', 'w', 'h'}


def test_live_detect_bad_frame():
    c = app.test_client()
    r = c.post('/api/live-detect', json={})
    assert r.status_code == 400
    r = c.post('/api/live-detect', json={'frame': 'not-base64!'})
    assert r.status_code == 400
    r = c.post('/api/live-detect', data='not json', content_type='application/json')
    assert r.status_code == 400


def test_live_detect_real_image():
    from core import detection as det
    if not det.model_available():
        return  # skip when model absent
    img = cv2.imread('D:/PixelLab/_bus.jpg')
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = det.detect(img)
    assert res['count'] >= 1
    labels = {o['label'] for o in res['objects']}
    assert 'bus' in labels, labels
    assert all(0.0 <= o['confidence'] <= 1.0 for o in res['objects'])
    assert all(-0.02 <= o['x'] <= 1.02 for o in res['objects'])
    assert all(-0.02 <= o['y'] <= 1.02 for o in res['objects'])


def test_live_deepfake_ok():
    c = app.test_client()
    r = c.post('/api/live-deepfake', json={'frame': make_frame()})
    assert r.status_code == 200, r.status_code
    data = r.get_json()
    assert data['label'] in ('DEEPFAKE', 'ORIGINAL')
    assert 'score' in data and 'confidence' in data
    assert data['model'] in ('cnn', 'heuristic')
    assert 0.0 <= data['score'] <= 1.0
    assert 'inference_ms' in data


def test_live_deepfake_bad_frame():
    c = app.test_client()
    r = c.post('/api/live-deepfake', json={})
    assert r.status_code == 400


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except Exception as e:
            failed += 1
            import traceback
            print("FAIL  %s: %s" % (t.__name__, e))
            traceback.print_exc()
    print("\nRESULTS: %d/%d passed, %d failed" % (len(tests) - failed, len(tests), failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())