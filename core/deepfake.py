"""
Deepfake vs Original frame classification.

Priority:
  1. If models/deepfake_lite.onnx exists -> real CNN classifier (sigmoid score).
  2. Otherwise -> forensic-artifact heuristic (scratch NumPy implementation),
     clearly labelled as EXPERIMENTAL (educational baseline).

Both paths operate on the largest detected face (Haar cascade), falling back
to a center crop when no face can be found.
"""
import os
import cv2
import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
_MODEL_PATH = os.path.join(_MODELS_DIR, 'deepfake_lite.onnx')
_INPUT_SIZE = 224

_cascade = None
_net = None
_load_tried = False


def model_available():
    return os.path.exists(_MODEL_PATH)


def model_status():
    if model_available():
        return 'ready (CNN)'
    return 'heuristic mode (no model -- run scripts/download_models.py)'


def _load_face_cascade():
    global _cascade
    if _cascade is None:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            _cascade = cv2.CascadeClassifier(cascade_path)
        else:
            _cascade = False  # unavailable
    return _cascade or None


def _load_net():
    global _net, _load_tried
    if _net is None and not _load_tried:
        _load_tried = True
        if model_available():
            try:
                net = cv2.dnn.readNetFromONNX(_MODEL_PATH)
                _net = net
            except Exception:
                _net = False
    return _net or None


def _largest_face_crop(img_rgb):
    """Return (crop_rgb, face_count) of the largest face, else center crop."""
    cascade = _load_face_cascade()
    h, w = img_rgb.shape[:2]
    if cascade:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(24, int(h * 0.06)), max(24, int(w * 0.06))))
        if len(faces) > 0:
            (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
            return img_rgb[y:y + fh, x:x + fw], len(faces)
    # Fallback: center crop 60%
    cw, ch = int(w * 0.6), int(h * 0.6)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img_rgb[y0:y0 + ch, x0:x0 + cw], 0


def _normalize_score(raw):
    """Clamp raw score into 0..1."""
    return float(max(0.0, min(1.0, raw)))


# --- Heuristic forensic scan (scratch, educational) ---------------------

def _heuristic_score(crop_rgb):
    """Cheap artifact scan. Higher -> more likely synthetic/deepfake."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (_INPUT_SIZE, _INPUT_SIZE))
    gray_f = gray.astype(np.float32)

    # 1) High-frequency residual energy (FFT) -- synthesized faces are
    #    frequently under-/over-smoothed so their spectral rolloff differs.
    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    c = (h // 2, w // 2)
    r = min(h, w) // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - c[1]) ** 2 + (yy - c[0]) ** 2)
    low = mag[dist <= r * 0.15].mean()
    high = mag[dist > r * 0.6].mean()
    rolloff = (high + 1.0) / (low + 1.0)

    # 2) Noise inconsistency across blocks -- face-swap pipelines often
    #    blend differently in regions, raising block-level variance.
    bs = 32
    blocks = gray[:gray.shape[0] - gray.shape[0] % bs, :gray.shape[1] - gray.shape[1] % bs]
    grid = blocks.reshape(blocks.shape[0] // bs, bs, blocks.shape[1] // bs, bs)
    block_var = grid.var(axis=(1, 3)).flatten()
    noise_std = block_var.std() / (block_var.mean() + 1e-6)

    # 3) Sharpness ratio (Laplacian) -- GAN outputs trend softer.
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()

    # Combine into 0..1 likelihood (empirical weighting, experimental!)
    rolloff_score = float(np.clip(rolloff * 8.0, 0, 1))
    noise_score = float(np.clip(noise_std * 20.0, 0, 1))
    sharp_score = float(np.clip(1.0 - lap / 1200.0, 0, 1))
    score = 0.4 * rolloff_score + 0.35 * noise_score + 0.25 * sharp_score
    return _normalize_score(score), {
        'rolloff': round(rolloff_score, 3),
        'noise': round(noise_score, 3),
        'sharpness': round(sharp_score, 3),
    }


# --- CNN path ------------------------------------------------------------

def _cnn_score(crop_rgb):
    """Forward through deepfake_lite.onnx. Expects 1x1 softmax/sigmoid."""
    net = _load_net()
    if net is None:
        return None
    resized = cv2.resize(crop_rgb, (_INPUT_SIZE, _INPUT_SIZE))
    blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (_INPUT_SIZE, _INPUT_SIZE),
                                 (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = np.asarray(out).flatten()
    if out.size >= 2:
        p = float(out[1])  # assume [original, deepfake]
    else:
        p = float(out[0])
    return _normalize_score(p)


# --- Public API ----------------------------------------------------------

def classify(img_rgb):
    """Returns a verdict dict for a single RGB frame."""
    import time
    t0 = time.time()
    crop, faces = _largest_face_crop(img_rgb)
    if crop.size == 0:
        crop = img_rgb

    net_score = _cnn_score(crop)
    if net_score is not None:
        score = net_score
        model = 'cnn'
        detail = {}
    else:
        score, detail = _heuristic_score(crop)
        model = 'heuristic'

    label = 'DEEPFAKE' if score >= 0.5 else 'ORIGINAL'
    return {
        'label': label,
        'score': round(score, 4),
        'confidence': round(abs(score - 0.5) * 2.0, 4),
        'model': model,
        'detail': detail,
        'faces': faces,
        'inference_ms': round((time.time() - t0) * 1000.0, 1),
        'model_status': model_status(),
    }