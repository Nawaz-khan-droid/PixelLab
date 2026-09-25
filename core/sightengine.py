"""
SightEngine API proxy — secure server-side integration.
Credentials stay on server (.env), never exposed to browser.

Free tier limits: 2000 ops/month, 500/day, 1 req/sec.
Built-in: SHA-256 result caching + 1 req/sec throttle.
"""
import os
import time
import hashlib
import logging
from collections import OrderedDict

import cv2
import numpy as np

try:
    import requests as _requests
except ImportError:
    _requests = None

log = logging.getLogger(__name__)

_API_URL = 'https://api.sightengine.com/1.0/check.json'
_MAX_DIM = 1280          # downscale before sending (saves bandwidth + API time)
_CACHE_MAX = 500         # LRU cache entries
_MIN_INTERVAL = 1.05     # seconds between API calls (slightly > 1 to be safe)

_last_call_time = 0.0
_cache: OrderedDict = OrderedDict()


def _get_credentials():
    """Read SightEngine credentials from environment."""
    user = os.environ.get('api_user', '').strip()
    secret = os.environ.get('api_secret', '').strip()
    if not user or not secret:
        raise ValueError(
            'SightEngine credentials not found. '
            'Set api_user and api_secret in .env or environment variables.'
        )
    return user, secret


def _downscale(img_bytes: bytes) -> bytes:
    """Downscale image to max _MAX_DIM on longest side. Returns JPEG bytes."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')
    h, w = img.shape[:2]
    if max(h, w) > _MAX_DIM:
        scale = _MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _hash_image(img_bytes: bytes) -> str:
    """SHA-256 hash for cache key."""
    return hashlib.sha256(img_bytes).hexdigest()[:16]


def _throttle():
    """Enforce minimum interval between API calls."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time = time.time()


def scan_image(img_bytes: bytes, models: str = 'genai') -> dict:
    """
    Send image to SightEngine API and return results.

    Args:
        img_bytes: Raw image bytes (any format cv2 can decode)
        models: Comma-separated model string, e.g. 'genai', 'deepfake',
                'genai,deepfake', 'type', 'qr-content',
                'genai,deepfake,type,qr-content'

    Returns:
        dict with API response or error info
    """
    if _requests is None:
        return {'error': True, 'message': 'requests library not installed'}

    # Downscale for bandwidth
    try:
        img_bytes = _downscale(img_bytes)
    except ValueError as e:
        return {'error': True, 'message': str(e)}

    # Check cache
    cache_key = _hash_image(img_bytes) + ':' + models
    if cache_key in _cache:
        _cache.move_to_end(cache_key)
        log.info('SightEngine cache hit for %s', cache_key[:12])
        result = _cache[cache_key].copy()
        result['cached'] = True
        return result

    # Get credentials
    try:
        api_user, api_secret = _get_credentials()
    except ValueError as e:
        return {'error': True, 'message': str(e)}

    # Throttle
    _throttle()

    # Call API
    try:
        resp = _requests.post(
            _API_URL,
            files={'media': ('image.jpg', img_bytes, 'image/jpeg')},
            data={
                'models': models,
                'api_user': api_user,
                'api_secret': api_secret,
            },
            timeout=15,
        )
    except _requests.exceptions.Timeout:
        return {'error': True, 'message': 'SightEngine API timeout (15s)'}
    except _requests.exceptions.ConnectionError:
        return {'error': True, 'message': 'Could not connect to SightEngine API'}
    except Exception as e:
        return {'error': True, 'message': f'API request failed: {str(e)}'}

    # Parse response
    if resp.status_code == 429:
        return {
            'error': True,
            'message': 'Rate limit exceeded. Free tier: 1 req/sec, 500/day, 2000/month.',
            'rate_limited': True,
        }
    if resp.status_code != 200:
        return {
            'error': True,
            'message': f'SightEngine returned HTTP {resp.status_code}',
        }

    try:
        data = resp.json()
    except Exception:
        return {'error': True, 'message': 'Invalid JSON from SightEngine'}

    if data.get('status') != 'success':
        err = data.get('error', {})
        return {
            'error': True,
            'message': err.get('message', 'SightEngine API error'),
            'code': err.get('code'),
        }

    # Build clean result
    result = {
        'error': False,
        'cached': False,
        'operations_used': data.get('request', {}).get('operations', 0),
    }

    # AI-Generated detection
    if 'type' in data and 'ai_generated' in data['type']:
        result['ai_generated'] = data['type']['ai_generated']
    # Per-generator scores (genai model)
    if 'type' in data and 'genai' in models:
        result['type_scores'] = {
            k: v for k, v in data.get('type', {}).items()
            if k not in ('ai_generated',)
        }

    # Deepfake detection
    if 'type' in data and 'deepfake' in data['type']:
        result['deepfake'] = data['type']['deepfake']

    # Photo vs Illustration
    if 'type' in data:
        if 'photo' in data['type']:
            result['photo'] = data['type']['photo']
        if 'illustration' in data['type']:
            result['illustration'] = data['type']['illustration']

    # QR content
    if 'qr' in data:
        result['qr'] = data['qr']

    # Cache result
    _cache[cache_key] = result.copy()
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)

    return result
