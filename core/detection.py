"""
Live object detection using YOLOv8n (ONNX) via OpenCV DNN.
No PyTorch/TensorFlow -- cv2.dnn only. Model lazy-loaded on first frame.
"""
import os
import cv2
import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
_MODEL_PATH = os.path.join(_MODELS_DIR, 'yolov8n.onnx')
_INPUT_SIZE = 640
_CONF_THRESH = 0.30
_IOU_THRESH = 0.45

COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

_net = None


def model_available():
    return os.path.exists(_MODEL_PATH)


def model_status():
    if model_available():
        return 'ready'
    return 'missing (run scripts/download_models.py)'


def _get_net():
    global _net
    if _net is None:
        if not model_available():
            raise FileNotFoundError('models/yolov8n.onnx not found -- run scripts/download_models.py')
        net = cv2.dnn.readNetFromONNX(_MODEL_PATH)
        _net = net
    return _net


def _postprocess(preds, conf_thresh, iou_thresh):
    """YOLOv8 ONNX output -> list of detection dicts (normalized coords)."""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.asarray(preds)
    # Layout is (1, 84, 8400) -> transpose to (8400, 84)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] == 84 and preds.shape[1] != 84:
        preds = preds.T

    boxes_px, scores, class_ids = [], [], []
    for p in preds:
        box = p[:4]
        classes = p[4:]
        class_id = int(np.argmax(classes))
        score = float(classes[class_id])
        if score < conf_thresh:
            continue
        cx, cy, w, h = box
        boxes_px.append([float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h)])
        scores.append(score)
        class_ids.append(class_id)

    if not boxes_px:
        return []

    keep = cv2.dnn.NMSBoxes(boxes_px, scores, conf_thresh, iou_thresh)
    if keep is None or len(keep) == 0:
        return []
    keep = np.asarray(keep).reshape(-1)

    results = []
    for i in keep:
        i = int(i)
        x, y, w, h = boxes_px[i]
        results.append({
            'label': COCO_LABELS[class_ids[i]] if class_ids[i] < len(COCO_LABELS) else 'unknown',
            'confidence': round(scores[i], 4),
            'x': round(x, 4),
            'y': round(y, 4),
            'w': round(w, 4),
            'h': round(h, 4),
        })
    return results


def detect(img_rgb, conf_thresh=_CONF_THRESH, iou_thresh=_IOU_THRESH):
    """Run detection on an RGB image. Returns {'objects': [...], 'count': n}."""
    h, w = img_rgb.shape[:2]
    net = _get_net()
    blob = cv2.dnn.blobFromImage(img_rgb, 1 / 255.0, (_INPUT_SIZE, _INPUT_SIZE),
                                 (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    t0 = cv2.getTickCount()
    outputs = net.forward()
    infer_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0

    dets = _postprocess(outputs, conf_thresh, iou_thresh)
    # Normalize coordinates to 0..1 relative to frame
    for d in dets:
        d['x'] = d['x'] / w
        d['y'] = d['y'] / h
        d['w'] = d['w'] / w
        d['h'] = d['h'] / h
    return {
        'objects': dets,
        'count': len(dets),
        'inference_ms': round(infer_ms, 1),
    }