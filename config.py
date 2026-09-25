"""
PixelLab Configuration -- Corrected Architecture
All 6 bugs from review fixed. Single source of truth.
"""

# === DEPLOYMENT ======================================================
MAX_IMAGE_DIMENSION = 2048       # px on longest side (resize on upload)
FFT_MAX_DIMENSION = 1024         # px -- FFT auto-downscales to this
MAX_FILE_SIZE_MB = 16
MAX_FILTER_STACK = 30            # per session
SESSION_TTL_SECONDS = 1800       # 30min -- lazy GC on each request
MAX_CONCURRENT_SESSIONS = 5      # conservative for 512MB

SUPPORTED_FORMATS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

# === OPERATION REGISTRY ==============================================
# Shape guard: which image modes each op accepts
# 'any'   = works on both color and grayscale
# 'color' = requires 3-channel (auto-converts if needed)
# 'gray'  = requires single-channel (auto-converts if needed)
#
# Output: what the op produces after running
# 'color' = always outputs 3-channel (even if input was gray)
# 'gray'  = always outputs single-channel
# 'pass'  = output matches input

OPERATIONS = {
    # == Geometric (5) ==
    'resize': {
        'category': 'geometric', 'label': 'Resize',
        'input': 'any', 'output': 'pass',
        'params': {'width_pct': 100, 'height_pct': 100},
        'controls': 'controls/resize.html',
    },
    'crop': {
        'category': 'geometric', 'label': 'Crop',
        'input': 'any', 'output': 'pass',
        'params': {'x': 0, 'y': 0, 'w': 0, 'h': 0},
        'controls': 'controls/crop.html',  # JS-based canvas drag
    },
    'flip': {
        'category': 'geometric', 'label': 'Flip',
        'input': 'any', 'output': 'pass',
        'params': {'direction': 'horizontal'},  # horizontal|vertical
    },
    'rotate': {
        'category': 'geometric', 'label': 'Rotate',
        'input': 'any', 'output': 'pass',
        'params': {'angle': 0},  # 0-360
        'controls': 'controls/rotate.html',
    },
    'shear': {
        'category': 'geometric', 'label': 'Shear',
        'input': 'any', 'output': 'pass',
        'params': {'shear_x': 0.0, 'shear_y': 0.0},  # -0.5 to 0.5
        'controls': 'controls/shear.html',
    },

    # == Point Operations (4) ==
    'negative': {
        'category': 'point', 'label': 'Negative',
        'input': 'any', 'output': 'pass',
        'params': {},
        'scratch': True,  # has pure-numpy implementation for viva
    },
    'gamma': {
        'category': 'point', 'label': 'Gamma Correction',
        'input': 'any', 'output': 'pass',
        'params': {'gamma': 1.0},  # 0.1-5.0
        'scratch': True,
        'controls': 'controls/gamma.html',
    },
    'threshold': {
        'category': 'point', 'label': 'Threshold',
        'input': 'gray', 'output': 'gray',
        'params': {'value': 127, 'mode': 'binary'},  # binary|binary_inv|trunc|tozero|otsu
        'controls': 'controls/threshold.html',
    },
    'otsu': {
        'category': 'point', 'label': "Otsu's Auto-Threshold",
        'input': 'gray', 'output': 'gray',
        'params': {},
        'scratch': True,
    },

    # == Color Operations (4) ==
    'grayscale': {
        'category': 'color', 'label': 'Grayscale',
        'input': 'color', 'output': 'gray',
        'params': {},
    },
    'brightness_contrast': {
        'category': 'color', 'label': 'Brightness / Contrast',
        'input': 'any', 'output': 'pass',
        'params': {'brightness': 0, 'contrast': 1.0},  # b:-100..100, c:0.5..2.0
        'controls': 'controls/brightness.html',
    },
    'color_balance': {
        'category': 'color', 'label': 'Color Balance',
        'input': 'color', 'output': 'color',
        'params': {'r': 1.0, 'g': 1.0, 'b': 1.0},  # 0.0-2.0
        'controls': 'controls/color_balance.html',
    },
    'hue_saturation': {
        'category': 'color', 'label': 'Hue / Saturation',
        'input': 'color', 'output': 'color',
        'params': {'hue_shift': 0, 'sat_factor': 1.0},  # h:-30..30, s:0.0..2.0
        'controls': 'controls/hue_sat.html',
    },
    'temperature': {
        'category': 'color', 'label': 'Temperature',
        'input': 'color', 'output': 'color',
        'params': {'warmth': 0.0},  # -1.0 (cool) to 1.0 (warm)
    },

    # == Smoothing (3) ==
    'gaussian_blur': {
        'category': 'smoothing', 'label': 'Gaussian Blur',
        'input': 'any', 'output': 'pass',
        'params': {'kernel': 5},  # 1-101, odd
        'controls': 'controls/blur.html',
    },
    'mean_blur': {
        'category': 'smoothing', 'label': 'Mean Blur',
        'input': 'any', 'output': 'pass',
        'params': {'kernel': 5},  # 1-101, odd
    },
    'median_blur': {
        'category': 'smoothing', 'label': 'Median Blur',
        'input': 'any', 'output': 'pass',
        'params': {'kernel': 5},  # 3-101, odd (min 3, NOT 1)
    },

    # == Sharpening (2) ==
    'unsharp_mask': {
        'category': 'sharpening', 'label': 'Unsharp Mask',
        'input': 'any', 'output': 'pass',
        'params': {'strength': 1.5},  # 1.0-3.0
        'controls': 'controls/unsharp_mask.html',
    },
    'high_boost': {
        'category': 'sharpening', 'label': 'High-Boost',
        'input': 'any', 'output': 'pass',
        'params': {'factor': 2.0},  # 1.5-5.0
        'controls': 'controls/high_boost.html',
    },

    # == Edge Detection (3) ==
    'sobel': {
        'category': 'edge', 'label': 'Sobel Edge',
        'input': 'gray', 'output': 'gray',
        'params': {'ksize': 3},  # 1,3,5,7
        'controls': 'controls/sobel.html',
    },
    'laplacian': {
        'category': 'edge', 'label': 'Laplacian Edge',
        'input': 'gray', 'output': 'gray',
        'params': {},
    },
    'canny': {
        'category': 'edge', 'label': 'Canny Edge',
        'input': 'gray', 'output': 'gray',
        'params': {'threshold1': 100, 'threshold2': 200},  # 0-255
        'controls': 'controls/canny.html',
    },

    # == Noise (2) ==
    'gaussian_noise': {
        'category': 'noise', 'label': 'Gaussian Noise',
        'input': 'any', 'output': 'pass',
        'params': {'sigma': 25, 'seed': None},  # sigma: 1-100, seed: int for reproducibility
        'controls': 'controls/noise.html',
    },
    'salt_pepper': {
        'category': 'noise', 'label': 'Salt & Pepper Noise',
        'input': 'any', 'output': 'pass',
        'params': {'amount': 0.02, 'seed': None},  # 0.01-0.10
        'controls': 'controls/noise.html',
    },

    # == Denoising (2) ==
    'denoise_gaussian': {
        'category': 'denoise', 'label': 'Gaussian Denoise',
        'input': 'any', 'output': 'pass',
        'params': {'kernel': 5},  # 1-101, odd
        'controls': 'controls/denoise_gaussian.html',
    },
    'denoise_median': {
        'category': 'denoise', 'label': 'Median Denoise',
        'input': 'any', 'output': 'pass',
        'params': {'kernel': 5},  # 3-101, odd
        'controls': 'controls/denoise_median.html',
    },

    # == Morphology (4) ==
    'erosion': {
        'category': 'morphology', 'label': 'Erosion',
        'input': 'gray', 'output': 'gray',
        'params': {'kernel': 5},  # 3-10
        'controls': 'controls/morphology.html',
    },
    'dilation': {
        'category': 'morphology', 'label': 'Dilation',
        'input': 'gray', 'output': 'gray',
        'params': {'kernel': 5},
        'controls': 'controls/morphology.html',
    },
    'opening': {
        'category': 'morphology', 'label': 'Opening',
        'input': 'gray', 'output': 'gray',
        'params': {'kernel': 5},
        'controls': 'controls/morphology.html',
    },
    'closing': {
        'category': 'morphology', 'label': 'Closing',
        'input': 'gray', 'output': 'gray',
        'params': {'kernel': 5},
        'controls': 'controls/morphology.html',
    },

    # == Frequency Domain (3) ==
    'fft_spectrum': {
        'category': 'frequency', 'label': 'FFT Magnitude Spectrum',
        'input': 'gray', 'output': 'gray',
        'params': {},
        'scratch': True,
    },
    'lowpass': {
        'category': 'frequency', 'label': 'Low-Pass Filter',
        'input': 'gray', 'output': 'gray',
        'params': {'radius': 30},  # 5-100
        'controls': 'controls/frequency.html',
    },
    'highpass': {
        'category': 'frequency', 'label': 'High-Pass Filter',
        'input': 'gray', 'output': 'gray',
        'params': {'radius': 30},
        'controls': 'controls/frequency.html',
    },

    # == Histogram (2) -- was missing! ==
    'histogram_equalize': {
        'category': 'histogram', 'label': 'Histogram Equalization',
        'input': 'gray', 'output': 'gray',
        'params': {},
        'scratch': True,
    },
    'histogram_compute': {
        'category': 'histogram', 'label': 'Histogram (Display Only)',
        'input': 'any', 'output': 'pass',
        'params': {},
        'visual_only': True,  # doesn't modify image, just displays chart
    },

    # == Detection (1) -- labeled "Application Demo" ==
    'face_detection': {
        'category': 'application', 'label': 'Face Detection (Demo)',
        'input': 'color', 'output': 'color',
        'params': {},
        'bonus': True,  # marked as bonus, not core DIP
    },
}

# === SCRATCH IMPLEMENTATIONS ==
# These operations have pure-numpy implementations for viva demonstration.
# When scratch mode is active, these bypass cv2 entirely.

SCRATCH_OPS = {
    'negative': 'pure numpy: 255 - img',
    'gamma': 'pure numpy: np.power(img/255.0, gamma) * 255',
    'otsu': "pure numpy: compute histogram, find threshold that minimizes intra-class variance",
    'histogram_equalize': 'pure numpy: compute CDF, map intensities',
    'fft_spectrum': 'pure numpy: np.fft.fft2 + fftshift + magnitude',
}

# === INFO PANELS ==
THEORY = {
    'point': {
        'title': 'Point Operations',
        'content': """Point operations transform each pixel independently.

Negative: Inverts intensity. Useful for enhancing white detail in dark regions.
Gamma: Power-law transform. Gamma less than 1 brightens; greater than 1 darkens. Used in display calibration.
Threshold: Binarizes image. Pixels above threshold become white, below become black. Otsu's method finds optimal threshold automatically. WARNING: Converts color images to grayscale.
Histogram Equalization: Spreads pixel intensities uniformly across the range. Improves contrast. WARNING: Converts color images to grayscale.

IMPORTANT: Operations are cumulative and cannot be reversed. Each operation adds to the filter stack. Use the undo button to remove the last operation.""",
    },
    'color': {
        'title': 'Color Operations',
        'content': """Color space conversions and adjustments operate on individual channels.

Grayscale: Converts to luminance using weighted sum: Y = 0.299R + 0.587G + 0.114B
Brightness/Contrast: Scales pixel values and shifts offset. Alpha scales contrast, Beta shifts brightness.
Hue/Saturation: Works in HSV color space where Hue is the color angle, Saturation is intensity, and Value is brightness.""",
    },
    'smoothing': {
        'title': 'Spatial Smoothing Filters',
        'content': """Smoothing applies convolution with a kernel to reduce noise and detail.

Mean/Box filter: Simple averaging over a square kernel. Effective against salt noise.
Gaussian filter: Bell-curve weighted averaging. Smooths without creating sharp edges.
Median filter: Nonlinear filter that replaces each pixel with the median of its neighbors. Best for salt and pepper noise.

IMPORTANT: Operations are cumulative. Setting kernel=1 is a no-op (1x1 window does nothing). You cannot unblur by applying a smaller kernel after a larger one. Use the undo button to remove the last operation.""",
    },
    'sharpening': {
        'title': 'Sharpening Filters',
        'content': """Sharpening enhances edges by boosting high-frequency detail.

Unsharp Masking: Subtracts a blurred version from the original to enhance edges.
High-Boost: Same approach as unsharp masking but with stronger amplification for pronounced edge enhancement.""",
    },
    'edge': {
        'title': 'Edge Detection',
        'content': """Edge detection finds regions of rapid intensity change.

Sobel: Computes gradient magnitude using directional derivative kernels.
Laplacian: Uses second-order derivatives. Zero-crossings indicate edge locations.
Canny: Multi-stage detector. Gaussian smooth, gradient computation, non-maximum suppression, then hysteresis thresholding.""",
    },
    'noise': {
        'title': 'Noise Models',
        'content': """Noise adds unwanted variations to pixel values.

Gaussian noise: Additive noise from sensor thermal effects. Follows normal distribution.
Salt and Pepper noise: Random pixels set to 0 or 255. Caused by transmission errors.
Denoising: Gaussian blur reduces Gaussian noise. Median filter reduces salt and pepper noise.""",
    },
    'morphology': {
        'title': 'Morphological Operations',
        'content': """Morphological operations process binary images using a structuring element.

Erosion: Shrinks white regions by keeping only pixels fully inside the foreground.
Dilation: Expands white regions by adding pixels near the boundary.
Opening: Erosion followed by dilation. Removes small white spots and thin protrusions.
Closing: Dilation followed by erosion. Fills small holes and gaps in the foreground.""",
    },
    'frequency': {
        'title': 'Frequency Domain Filtering',
        'content': """Frequency domain methods transform the image using the Fast Fourier Transform.

Low-pass filter: Keeps low frequencies, removes high frequencies. Results in a blurred image.
High-pass filter: Keeps high frequencies, removes low frequencies. Highlights edges and fine detail.
FFT Spectrum: Shows the magnitude of frequency components. Low frequencies are at the center.""",
    },
    'histogram': {
        'title': 'Histogram Operations',
        'content': """Histograms show the distribution of pixel intensity values.

Histogram: Count of pixels at each intensity level from 0 to 255.
Equalization: Maps pixel values to achieve a uniform distribution. Improves contrast for under or over-exposed images.
CLAHE: Adaptive version that works on local regions. Prevents over-amplification of noise.""",
    },
    'application': {
        'title': 'Application Demos',
        'content': """Computer vision applications built on digital image processing fundamentals.

Face Detection: Uses Haar cascade classifiers trained on frontal faces. Employs integral images for fast detection.""",
    },
}

# === SESSION STATE TEMPLATE ==
def new_session():
    return {
        'original': None,       # np.ndarray, uint8, RGB
        'processed': None,      # np.ndarray, uint8, RGB or Gray
        'filter_stack': [],     # list of {id, name, type, params}
        'filename': '',
        'dimensions': (0, 0),
        'is_gray': False,       # track if grayscale was applied
        'created_at': 0,        # time.time() for TTL cleanup
        'render_count': 0,      # incremented on each render for cache-busting
    }
