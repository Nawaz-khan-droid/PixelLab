"""
PixelLab Image Processor -- Corrected Architecture
- Shape guards: auto-converts color/gray based on registry metadata
- Noise seeding: deterministic results on filter reorder
- Scratch implementations: pure numpy for viva demonstration
- FFT: uses rfft2 + complex64 to stay within RAM budget
"""
import cv2
import numpy as np
import time
from typing import Any, Dict, List, Optional

from config import OPERATIONS, SCRATCH_OPS, FFT_MAX_DIMENSION


class ImageProcessor:
    """Stateless image processor -- all state lives in session dict."""

    # -- SHAPE GUARDS --------------------------------------------------
    @staticmethod
    def _ensure_shape(image: np.ndarray, required: str) -> np.ndarray:
        """Convert image to match operation's input requirement.

        Args:
            image: input array (RGB or Gray)
            required: 'any' | 'color' | 'gray'
        Returns:
            image in correct shape
        Raises:
            ValueError if conversion impossible
        """
        if required == 'any':
            return image

        is_3ch = len(image.shape) == 3 and image.shape[2] == 3

        if required == 'gray' and is_3ch:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif required == 'color' and not is_3ch:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    # -- MAIN DISPATCH -------------------------------------------------
    @staticmethod
    def apply_operation(image: np.ndarray, op_type: str, params: Dict[str, Any],
                        scratch: bool = False) -> np.ndarray:
        """Apply a single operation to an image.

        Args:
            image: input RGB or grayscale uint8 array
            op_type: key from OPERATIONS registry
            params: operation parameters (may include seed)
            scratch: if True, use pure-numpy implementation where available
        Returns:
            processed image
        """
        op_meta = OPERATIONS[op_type]
        if op_meta.get('visual_only'):
            return image  # histogram display doesn't modify image

        # Shape guard
        img = ImageProcessor._ensure_shape(image, op_meta['input'])

        # Dispatch
        handler = getattr(ImageProcessor, f'_op_{op_type}')

        if scratch and op_type in SCRATCH_OPS:
            handler = getattr(ImageProcessor, f'_scratch_{op_type}', handler)

        result = handler(img, params)

        # Ensure output matches declared shape
        if op_meta['output'] == 'color' and len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif op_meta['output'] == 'gray' and len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        return result

    # -- REPLAY ENTIRE STACK (with seeding for noise) ------------------
    @staticmethod
    def replay_stack(original: np.ndarray, filter_stack: List[Dict]) -> np.ndarray:
        """Reapply all filters from original. Noise ops use stored seeds."""
        result = original.copy()
        for f in filter_stack:
            params = f['params'].copy()
            # Noise ops already have 'seed' in params -- deterministic replay
            result = ImageProcessor.apply_operation(result, f['type'], params)
        return result

    # ------------------------------------------------------------------
    # GEOMETRIC OPERATIONS
    # ------------------------------------------------------------------
    @staticmethod
    def _op_resize(img, params):
        h, w = img.shape[:2]
        nw = max(1, int(w * params.get('width_pct', 100) / 100))
        nh = max(1, int(h * params.get('height_pct', 100) / 100))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _op_crop(img, params):
        x = int(params.get('x', 0))
        y = int(params.get('y', 0))
        w = int(params.get('w', 0))
        h = int(params.get('h', 0))
        ih, iw = img.shape[:2]
        if w <= 0 or h <= 0:
            return img  # no crop params = return original
        x, y = max(0, min(x, iw - 1)), max(0, min(y, ih - 1))
        w, h = min(w, iw - x), min(h, ih - y)
        if w <= 0 or h <= 0:
            return img
        return img[y:y+h, x:x+w]

    @staticmethod
    def _op_flip(img, params):
        code = 1 if params.get('direction', 'horizontal') == 'horizontal' else 0
        return cv2.flip(img, code)

    @staticmethod
    def _op_rotate(img, params):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, params.get('angle', 0), 1.0)
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def _op_shear(img, params):
        rows, cols = img.shape[:2]
        sx, sy = params.get('shear_x', 0.0), params.get('shear_y', 0.0)
        Mx = np.float32([[1, sx, 0], [0, 1, 0]])
        out = cv2.warpAffine(img, Mx, (cols, rows))
        My = np.float32([[1, 0, 0], [sy, 1, 0]])
        return cv2.warpAffine(out, My, (cols, rows))

    # ------------------------------------------------------------------
    # POINT OPERATIONS
    # ------------------------------------------------------------------
    @staticmethod
    def _op_negative(img, params):
        return cv2.bitwise_not(img)

    @staticmethod
    def _scratch_negative(img, params):
        return (255 - img.astype(np.uint16)).astype(np.uint8)

    @staticmethod
    def _op_gamma(img, params):
        gamma = params.get('gamma', 1.0)
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    @staticmethod
    def _scratch_gamma(img, params):
        gamma = params.get('gamma', 1.0)
        return (np.power(img.astype(np.float32) / 255.0, gamma) * 255).astype(np.uint8)

    @staticmethod
    def _op_threshold(img, params):
        val = params.get('value', 127)
        mode_str = params.get('mode', 'binary')
        mode_map = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
        }
        _, out = cv2.threshold(img, val, 255, mode_map.get(mode_str, cv2.THRESH_BINARY))
        return out

    @staticmethod
    def _op_otsu(img, params):
        _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out

    @staticmethod
    def _scratch_otsu(img, params):
        """Pure numpy Otsu: find threshold minimizing intra-class variance."""
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
        total = img.size
        sum_total = np.dot(np.arange(256), hist)
        sum_bg, weight_bg, max_var, best_t = 0.0, 0, 0.0, 0
        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var > max_var:
                max_var = var
                best_t = t
        return ((img > best_t) * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # COLOR OPERATIONS
    # ------------------------------------------------------------------
    @staticmethod
    def _op_grayscale(img, params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _op_brightness_contrast(img, params):
        b = params.get('brightness', 0)
        c = params.get('contrast', 1.0)
        return cv2.convertScaleAbs(img, alpha=c, beta=b)

    @staticmethod
    def _op_color_balance(img, params):
        b, g, r = cv2.split(img)
        r = cv2.convertScaleAbs(r, alpha=params.get('r', 1.0))
        g = cv2.convertScaleAbs(g, alpha=params.get('g', 1.0))
        b = cv2.convertScaleAbs(b, alpha=params.get('b', 1.0))
        return cv2.merge([b, g, r])

    @staticmethod
    def _op_hue_saturation(img, params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].astype(np.int16)
        hsv[:, :, 0] = np.mod(hue + params.get('hue_shift', 0), 180).astype(np.uint8)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=params.get('sat_factor', 1.0))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _op_temperature(img, params):
        warmth = float(params.get('warmth', 0.0))
        out = img.copy().astype(np.float32)
        out[:, :, 0] = np.clip(out[:, :, 0] + warmth * 30, 0, 255)  # R
        out[:, :, 2] = np.clip(out[:, :, 2] - warmth * 30, 0, 255)  # B
        return out.astype(np.uint8)

    # ------------------------------------------------------------------
    # SMOOTHING OPERATIONS
    # ------------------------------------------------------------------
    @staticmethod
    def _op_gaussian_blur(img, params):
        k = params.get('kernel', 5)
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def _op_mean_blur(img, params):
        k = params.get('kernel', 5)
        k = k if k % 2 == 1 else k + 1
        return cv2.blur(img, (k, k))

    @staticmethod
    def _op_median_blur(img, params):
        k = max(3, params.get('kernel', 5))  # min 3, NOT 1
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(img, k)

    # ------------------------------------------------------------------
    # SHARPENING OPERATIONS
    # ------------------------------------------------------------------
    @staticmethod
    def _op_unsharp_mask(img, params):
        s = params.get('strength', 1.5)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        return cv2.addWeighted(img, s, blur, -(s - 1), 0)

    @staticmethod
    def _op_high_boost(img, params):
        f = params.get('factor', 2.0)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        return cv2.addWeighted(img, f, blur, -(f - 1), 0)

    # ------------------------------------------------------------------
    # EDGE DETECTION
    # ------------------------------------------------------------------
    @staticmethod
    def _op_sobel(img, params):
        k = params.get('ksize', 3)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        return cv2.convertScaleAbs(cv2.magnitude(gx, gy))

    @staticmethod
    def _op_laplacian(img, params):
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)

    @staticmethod
    def _op_canny(img, params):
        t1 = params.get('threshold1', 100)
        t2 = params.get('threshold2', 200)
        return cv2.Canny(img, t1, t2)

    # ------------------------------------------------------------------
    # NOISE (with seeding for deterministic replay)
    # ------------------------------------------------------------------
    @staticmethod
    def _op_gaussian_noise(img, params):
        sigma = params.get('sigma', 25)
        seed = params.get('seed')
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _op_salt_pepper(img, params):
        amount = params.get('amount', 0.02)
        seed = params.get('seed')
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        rng = np.random.default_rng(seed)
        out = img.copy()
        # Salt
        n_salt = int(np.ceil(amount * img.size / 2))
        coords = tuple(rng.integers(0, i, n_salt) for i in img.shape)
        out[coords] = 255
        # Pepper
        n_pepper = int(np.ceil(amount * img.size / 2))
        coords = tuple(rng.integers(0, i, n_pepper) for i in img.shape)
        out[coords] = 0
        return out

    # ------------------------------------------------------------------
    # DENOISING
    # ------------------------------------------------------------------
    @staticmethod
    def _op_denoise_gaussian(img, params):
        k = params.get('kernel', 5)
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def _op_denoise_median(img, params):
        k = max(3, params.get('kernel', 5))
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(img, k)

    # ------------------------------------------------------------------
    # MORPHOLOGY
    # ------------------------------------------------------------------
    @staticmethod
    def _morph_op(img, params, op):
        k = max(3, min(10, params.get('kernel', 5)))
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(img, op, kernel)

    @staticmethod
    def _op_erosion(img, params):
        return ImageProcessor._morph_op(img, params, cv2.MORPH_ERODE)

    @staticmethod
    def _op_dilation(img, params):
        return ImageProcessor._morph_op(img, params, cv2.MORPH_DILATE)

    @staticmethod
    def _op_opening(img, params):
        return ImageProcessor._morph_op(img, params, cv2.MORPH_OPEN)

    @staticmethod
    def _op_closing(img, params):
        return ImageProcessor._morph_op(img, params, cv2.MORPH_CLOSE)

    # ------------------------------------------------------------------
    # FREQUENCY DOMAIN (uses rfft2 + complex64 for RAM savings)
    # ------------------------------------------------------------------
    @staticmethod
    def _fft_helper(img):
        """Downscale + rfft2 (real FFT, complex64) -- ~60% RAM vs fft2."""
        h, w = img.shape[:2]
        if max(h, w) > FFT_MAX_DIMENSION:
            scale = FFT_MAX_DIMENSION / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        f = np.fft.rfft2(img.astype(np.float32))
        fshift = np.fft.fftshift(f, axes=(0,))
        return fshift, img.shape

    @staticmethod
    def _ifft_helper(fshift, original_shape):
        """Inverse of _fft_helper."""
        f_ishift = np.fft.ifftshift(fshift, axes=(0,))
        result = np.fft.irfft2(f_ishift, s=original_shape)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _op_fft_spectrum(img, params):
        fshift, _ = ImageProcessor._fft_helper(img)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def _scratch_fft_spectrum(img, params):
        """Pure numpy FFT spectrum -- no cv2 at all."""
        h, w = img.shape[:2]
        if max(h, w) > FFT_MAX_DIMENSION:
            scale = FFT_MAX_DIMENSION / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            # Pure numpy resize via nearest-neighbor sampling
            row_idx = (np.arange(nh) * h / nh).astype(int)
            col_idx = (np.arange(nw) * w / nw).astype(int)
            img = img[np.ix_(row_idx, col_idx)]
        f = np.fft.rfft2(img.astype(np.float32))
        fshift = np.fft.fftshift(f, axes=(0,))
        mag = 20 * np.log(np.abs(fshift) + 1)
        mag = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)
        return mag

    @staticmethod
    def _make_freq_mask(rows, cols, radius, lowpass=True):
        """Create circular frequency mask for rfft2 output shape (rows, cols//2+1)."""
        fft_cols = cols // 2 + 1
        cy, cx = rows // 2, fft_cols // 2
        Y, X = np.ogrid[:rows, :fft_cols]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        mask = np.zeros((rows, fft_cols), dtype=np.float32)
        if lowpass:
            mask[dist <= radius] = 1.0
        else:
            mask[dist > radius] = 1.0
        return mask

    @staticmethod
    def _op_lowpass(img, params):
        fshift, shape = ImageProcessor._fft_helper(img)
        radius = params.get('radius', 30)
        mask = ImageProcessor._make_freq_mask(shape[0], shape[1], radius, lowpass=True)
        return ImageProcessor._ifft_helper(fshift * mask, shape)

    @staticmethod
    def _op_highpass(img, params):
        fshift, shape = ImageProcessor._fft_helper(img)
        radius = params.get('radius', 30)
        mask = ImageProcessor._make_freq_mask(shape[0], shape[1], radius, lowpass=False)
        return ImageProcessor._ifft_helper(fshift * mask, shape)

    # ------------------------------------------------------------------
    # HISTOGRAM
    # ------------------------------------------------------------------
    @staticmethod
    def _op_histogram_equalize(img, params):
        if len(img.shape) == 3:
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return cv2.equalizeHist(img)

    @staticmethod
    def _scratch_histogram_equalize(img, params):
        """Pure numpy histogram equalization via CDF."""
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        total = img.size
        lut = ((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
        return lut[img]

    @staticmethod
    def _op_histogram_compute(img, params):
        return img  # visual_only -- doesn't modify

    # ------------------------------------------------------------------
    # DETECTION (Application Demo -- bonus)
    # ------------------------------------------------------------------
    _face_cascade = None

    @staticmethod
    def _op_face_detection(img, params):
        try:
            if ImageProcessor._face_cascade is None:
                import os
                cascade_path = os.path.join(
                    os.path.dirname(__file__), 'haar',
                    'haarcascade_frontalface_default.xml'
                )
                ImageProcessor._face_cascade = cv2.CascadeClassifier(cascade_path)
                if ImageProcessor._face_cascade.empty():
                    raise RuntimeError("Failed to load cascade")

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = ImageProcessor._face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return img
        except (AttributeError, RuntimeError):
            # opencv-python-headless may not include CascadeClassifier
            # Return original image -- face detection unavailable
            return img

    # ------------------------------------------------------------------
    # HISTOGRAM DATA (for Chart.js -- no matplotlib)
    # ------------------------------------------------------------------
    @staticmethod
    def get_histogram_data(image: np.ndarray) -> Dict[str, list]:
        """Return histogram data as plain lists for Chart.js."""
        if len(image.shape) == 2:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return {'gray': hist.flatten().tolist()}

        data = {}
        for i, ch in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            data[ch] = hist.flatten().tolist()
        return data

    # ------------------------------------------------------------------
    # DISPLAY IMAGE (serves ~800px JPEG for HTMX fragments)
    # ------------------------------------------------------------------
    @staticmethod
    def get_display_bytes(image: np.ndarray, max_dim: int = 800) -> bytes:
        """Return JPEG bytes at display size for inline HTMX response."""
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        if len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()
