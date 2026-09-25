"""
PixelLab -- Flask Application
Single-process, no-threads, lazy session GC.
"""
import os
import time
import uuid
import base64
import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect, url_for,
                   session, send_file, jsonify)
from io import BytesIO

# Load .env before anything reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; credentials must be in real env vars

from config import (OPERATIONS, THEORY, SESSION_TTL_SECONDS,
                    MAX_FILE_SIZE_MB, MAX_IMAGE_DIMENSION, MAX_FILTER_STACK,
                    SUPPORTED_FORMATS, new_session)
from core.processor import ImageProcessor
from core import detection as detection_module
from core import deepfake as deepfake_module
from core import sightengine as sightengine_module

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

# -- In-memory session store (single-process, no DB) --
sessions = {}


def get_session():
    """Get or create session, with lazy TTL cleanup."""
    sid = session.get('sid')
    now = time.time()

    # Lazy GC: remove expired sessions on every request
    expired = [k for k, v in sessions.items() if now - v['created_at'] > SESSION_TTL_SECONDS]
    for k in expired:
        del sessions[k]

    if sid and sid in sessions:
        sessions[sid]['created_at'] = now  # refresh TTL
        return sessions[sid]

    # New session
    sid = str(uuid.uuid4())
    session['sid'] = sid
    sessions[sid] = new_session()
    sessions[sid]['created_at'] = now
    return sessions[sid]


# -- PAGE ROUTES ------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/editor')
def editor():
    s = get_session()
    if s['original'] is None:
        return render_template('editor.html',
                               operations=OPERATIONS,
                               theory=THEORY,
                               is_gray=False,
                               filename='',
                               dimensions=(0, 0),
                               filter_stack=[],
                               processed=None,
                               render_count=0,
                               histogram={})
    return render_template('editor.html',
                           operations=OPERATIONS,
                           theory=THEORY,
                           is_gray=s['is_gray'],
                           filename=s['filename'],
                           dimensions=s['dimensions'],
                           filter_stack=s['filter_stack'],
                           processed=s['processed'],
                           render_count=s['render_count'],
                           histogram=ImageProcessor.get_histogram_data(s['processed']) if s['processed'] is not None else {})


@app.route('/lab')
def lab():
    s = get_session()
    if s['original'] is None:
        return render_template('lab.html',
                               operations=OPERATIONS,
                               theory=THEORY,
                               is_gray=False,
                               filename='',
                               dimensions=(0, 0),
                               filter_stack=[],
                               processed=None,
                               render_count=0,
                               histogram={})
    return render_template('lab.html',
                           operations=OPERATIONS,
                           theory=THEORY,
                           is_gray=s['is_gray'],
                           filename=s['filename'],
                           dimensions=s['dimensions'],
                           filter_stack=s['filter_stack'],
                           processed=s['processed'],
                           render_count=s['render_count'],
                           histogram=ImageProcessor.get_histogram_data(s['processed']) if s['processed'] is not None else {})


@app.route('/download')
def download():
    s = get_session()
    if s['processed'] is None:
        return render_template('download.html',
                               has_image=False)
    bgr = cv2.cvtColor(s['processed'], cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', bgr)
    return send_file(
        BytesIO(buf.tobytes()),
        mimetype='image/png',
        as_attachment=True,
        download_name=f"pixellab_{s['filename']}"
    )


@app.route('/live')
def live():
    """Live webcam analysis (object detection / deepfake)."""
    get_session()
    return render_template('live.html',
                           detection_status=detection_module.model_status(),
                           deepfake_status=deepfake_module.model_status())


def _decode_frame_payload(data):
    """Decode a base64 JPEG frame from JSON. Returns RGB ndarray or None."""
    frame = (data or {}).get('frame')
    if not frame:
        return None
    try:
        buf = np.frombuffer(base64.b64decode(frame), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.route('/api/live-detect', methods=['POST'])
def live_detect():
    if not detection_module.model_available():
        return jsonify({'error': 'model_not_available',
                        'message': detection_module.model_status()}), 503
    img = _decode_frame_payload(request.get_json(silent=True))
    if img is None:
        return jsonify({'error': 'bad_frame'}), 400
    try:
        return jsonify(detection_module.detect(img))
    except Exception:
        return jsonify({'error': 'inference_failed'}), 500


@app.route('/api/live-deepfake', methods=['POST'])
def live_deepfake():
    img = _decode_frame_payload(request.get_json(silent=True))
    if img is None:
        return jsonify({'error': 'bad_frame'}), 400
    try:
        return jsonify(deepfake_module.classify(img))
    except Exception:
        return jsonify({'error': 'inference_failed'}), 500


# ?- HTMX PARTIAL ROUTES ----------------------------------------------

@app.route('/api/upload', methods=['POST'])
def upload():
    s = get_session()
    file = request.files.get('image')
    if not file or not file.filename:
        return redirect(url_for('index'))

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in SUPPORTED_FORMATS:
        return "Unsupported format", 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "Failed to read image", 400

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if exceeds max dimension
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # RGBA -> RGB
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    s['original'] = img
    s['processed'] = img.copy()
    s['filter_stack'] = []
    s['filename'] = file.filename
    s['dimensions'] = (h, w)
    s['is_gray'] = False

    return redirect(url_for('editor'))


@app.route('/api/apply', methods=['POST'])
def apply_operation():
    s = get_session()
    if s['original'] is None:
        return "No image loaded", 400

    op_type = request.form.get('operation')
    if op_type not in OPERATIONS:
        return "Unknown operation", 400

    if len(s['filter_stack']) >= MAX_FILTER_STACK:
        return "Filter stack limit reached", 400

    # Parse params from form
    params = {}
    for key, val in request.form.items():
        if key == 'operation':
            continue
        # Try numeric conversion
        try:
            params[key] = int(val)
        except ValueError:
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val

    # Seed noise operations for reproducibility
    if op_type in ('gaussian_noise', 'salt_pepper'):
        if 'seed' not in params or not params['seed']:
            params['seed'] = int(time.time() * 1000) % (2**31)

    # Auto-set grayscale flag
    op_meta = OPERATIONS[op_type]
    if op_type == 'grayscale':
        s['is_gray'] = True

    # Add to stack and replay
    filter_entry = {
        'id': str(uuid.uuid4()),
        'name': op_meta['label'],
        'type': op_type,
        'params': params,
    }
    s['filter_stack'].append(filter_entry)
    s['processed'] = ImageProcessor.replay_stack(s['original'], s['filter_stack'])
    s['render_count'] = s.get('render_count', 0) + 1

    return render_template('partials/editor_content.html',
                           operations=OPERATIONS,
                           theory=THEORY,
                           filter_stack=s['filter_stack'],
                           is_gray=s['is_gray'],
                           processed=s['processed'],
                           dimensions=s['dimensions'],
                           render_count=s['render_count'],
                           histogram=ImageProcessor.get_histogram_data(s['processed']))


FILTER_PRESETS = {
    'grayscale': [{'type': 'grayscale', 'params': {}}],
    'sepia': [{'type': 'grayscale', 'params': {}}, {'type': 'color_balance', 'params': {'r': 1.2, 'g': 1.0, 'b': 0.8}}],
    'vintage': [{'type': 'color_balance', 'params': {'r': 1.3, 'g': 1.1, 'b': 0.9}}, {'type': 'gamma', 'params': {'gamma': 0.8}}],
    'noir': [{'type': 'grayscale', 'params': {}}, {'type': 'brightness_contrast', 'params': {'brightness': 0, 'contrast': 1.5}}],
    'sketch': [{'type': 'grayscale', 'params': {}}, {'type': 'negative', 'params': {}}, {'type': 'gaussian_blur', 'params': {'kernel': 11}}],
    'pop': [{'type': 'hue_saturation', 'params': {'hue_shift': 0, 'sat_factor': 1.8}}, {'type': 'brightness_contrast', 'params': {'brightness': 0, 'contrast': 1.3}}],
}


@app.route('/api/filter-preset', methods=['POST'])
def apply_filter_preset():
    s = get_session()
    preset = request.form.get('preset', '')
    steps = FILTER_PRESETS.get(preset, [])
    if not steps:
        return '', 400

    for step in steps:
        op_type = step['type']
        params = step['params']
        op_meta = OPERATIONS.get(op_type, {})
        filter_entry = {
            'id': str(uuid.uuid4()),
            'name': op_meta.get('label', op_type),
            'type': op_type,
            'params': params,
        }
        s['filter_stack'].append(filter_entry)

    s['is_gray'] = any(f['type'] == 'grayscale' for f in s['filter_stack'])
    s['processed'] = ImageProcessor.replay_stack(s['original'], s['filter_stack'])
    s['render_count'] = s.get('render_count', 0) + 1

    return render_template('partials/editor_content.html',
                           operations=OPERATIONS,
                           theory=THEORY,
                           filter_stack=s['filter_stack'],
                           is_gray=s['is_gray'],
                           processed=s['processed'],
                           dimensions=s['dimensions'],
                           render_count=s['render_count'],
                           histogram=ImageProcessor.get_histogram_data(s['processed']))


@app.route('/api/remove/<filter_id>', methods=['POST'])
def remove_filter(filter_id):
    s = get_session()
    s['filter_stack'] = [f for f in s['filter_stack'] if f['id'] != filter_id]

    # Recompute grayscale flag
    s['is_gray'] = any(f['type'] == 'grayscale' for f in s['filter_stack'])

    # Replay from scratch (noise seeds are deterministic now)
    if s['filter_stack']:
        s['processed'] = ImageProcessor.replay_stack(s['original'], s['filter_stack'])
    else:
        s['processed'] = s['original'].copy()
    s['render_count'] = s.get('render_count', 0) + 1

    return render_template('partials/editor_content.html',
                           operations=OPERATIONS,
                           theory=THEORY,
                           filter_stack=s['filter_stack'],
                           is_gray=s['is_gray'],
                           processed=s['processed'],
                           dimensions=s['dimensions'],
                           render_count=s['render_count'],
                           histogram=ImageProcessor.get_histogram_data(s['processed']))


@app.route('/api/undo', methods=['POST'])
def undo():
    s = get_session()
    if s['filter_stack']:
        s['filter_stack'].pop()

    s['is_gray'] = any(f['type'] == 'grayscale' for f in s['filter_stack'])

    if s['filter_stack']:
        s['processed'] = ImageProcessor.replay_stack(s['original'], s['filter_stack'])
    else:
        s['processed'] = s['original'].copy()
    s['render_count'] = s.get('render_count', 0) + 1

    return render_template('partials/editor_content.html',
                           operations=OPERATIONS,
                           theory=THEORY,
                           filter_stack=s['filter_stack'],
                           is_gray=s['is_gray'],
                           processed=s['processed'],
                           dimensions=s['dimensions'],
                           render_count=s['render_count'],
                           histogram=ImageProcessor.get_histogram_data(s['processed']))


@app.route('/api/reset', methods=['POST'])
def reset():
    s = get_session()
    s['filter_stack'] = []
    s['processed'] = s['original'].copy()
    s['is_gray'] = False
    s['render_count'] = s.get('render_count', 0) + 1

    return render_template('partials/editor_content.html',
                           operations=OPERATIONS,
                           theory=THEORY,
                           filter_stack=[],
                           is_gray=False,
                           processed=s['processed'],
                           dimensions=s['dimensions'],
                           render_count=s['render_count'],
                           histogram=ImageProcessor.get_histogram_data(s['processed']))


@app.route('/api/image/<img_type>')
def serve_image(img_type):
    """Serve image as JPEG bytes for <img> tags (display size)."""
    s = get_session()
    if img_type == 'original' and s['original'] is not None:
        return send_file(
            BytesIO(ImageProcessor.get_display_bytes(s['original'])),
            mimetype='image/jpeg'
        )
    elif img_type == 'processed' and s['processed'] is not None:
        return send_file(
            BytesIO(ImageProcessor.get_display_bytes(s['processed'])),
            mimetype='image/jpeg'
        )
    return '', 404


@app.route('/sightengine')
def sightengine_page():
    """Page for SightEngine AI and Deepfake scan."""
    return render_template('sightengine.html')


@app.route('/api/sightengine/scan', methods=['POST'])
def sightengine_scan():
    """Proxy route to SightEngine API."""
    if 'image' not in request.files:
        return jsonify({'error': True, 'message': 'No image uploaded'})
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': True, 'message': 'No selected file'})
        
    models = request.form.get('models', 'genai,deepfake')
    img_bytes = file.read()
    
    # Call the proxy
    result = sightengine_module.scan_image(img_bytes, models=models)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)
