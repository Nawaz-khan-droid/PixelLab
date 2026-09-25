"""
PixelLab Browser Flow Test
Full HTTP flow with cookie persistence, asserting image BYTES change per op.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hashlib
import io
import re
import urllib.request
import urllib.parse
import http.cookiejar
import numpy as np
import cv2

BASE = "http://127.0.0.1:8501"
PASS = 0
FAIL = 0
ERRORS = []

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def h(b):
    return hashlib.md5(b).hexdigest()


def get(url):
    try:
        return opener.open(url)
    except urllib.error.HTTPError as e:
        return e


def post_form(url, fields):
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    try:
        return opener.open(req)
    except urllib.error.HTTPError as e:
        return e


def post_multipart(url, files):
    boundary = '----PLBoundary'
    body = io.BytesIO()
    for name, (filename, fdata, ctype) in files.items():
        body.write(('--' + boundary + '\r\n').encode())
        body.write(('Content-Disposition: form-data; name="' + name + '"; filename="' + filename + '"\r\n').encode())
        body.write(('Content-Type: ' + ctype + '\r\n\r\n').encode())
        body.write(fdata)
        body.write(b'\r\n')
    body.write(('--' + boundary + '--\r\n').encode())
    data = body.getvalue()
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'multipart/form-data; boundary=' + boundary)
    return opener.open(req)


def get_image_md5():
    r = get(BASE + "/api/image/processed")
    return h(r.read())


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print("  [PASS] " + name)
    else:
        FAIL += 1
        msg = "  [FAIL] " + name
        if detail:
            msg += " -- " + detail
        print(msg)
        ERRORS.append(name)


print("=" * 60)
print("PIXELLAB BROWSER FLOW TEST")
print("=" * 60)

# 1. Home page
print("\n--- 1. Home page ---")
r = get(BASE + "/")
html = r.read().decode()
check("Home page loads", r.status == 200)
check("HTMX CDN present", 'htmx.org' in html)
check("Chart.js CDN present", 'chart.js' in html.lower() or 'chart.umd' in html.lower())
check("app.js loaded", 'app.js' in html)
check("viewport meta tag", 'viewport' in html)

# 2. Upload
print("\n--- 2. Upload flow ---")
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:, :] = [50, 100, 200]
img[50:150, 50:350] = [200, 50, 50]
_, buf = cv2.imencode('.png', img)
orig_bytes = buf.tobytes()

r = post_multipart(BASE + "/api/upload", {'image': ('test.png', orig_bytes, 'image/png')})
check("Upload redirects to editor", r.status in (200, 302) and '/editor' in r.url)

editor_html = get(r.url).read().decode()
check("Editor page renders", len(editor_html) > 1000)
check("#editor-content div exists", 'id="editor-content"' in editor_html)
check("Image tag present", '<img' in editor_html)

# 3. Image serves
print("\n--- 3. Image serving ---")
orig_md5 = get_image_md5()
check("Original image serves", orig_md5 is not None)

# 4. Simple ops that produce measurable byte changes
print("\n--- 4. Operations (byte change proof) ---")
# Reset first for clean state
post_form(BASE + "/api/reset", {})
base_md5 = get_image_md5()

# Ops that definitely change pixels
ops = [
    ('negative', {}),
    ('gamma', {'gamma': '2.0'}),
    ('threshold', {'value': '128', 'mode': 'binary'}),
    ('grayscale', {}),
    ('gaussian_blur', {'kernel': '15'}),
    ('mean_blur', {'kernel': '15'}),
    ('median_blur', {'kernel': '15'}),
    ('unsharp_mask', {}),
    ('high_boost', {}),
    ('sobel', {}),
    ('laplacian', {}),
    ('canny', {'threshold1': '50', 'threshold2': '150'}),
    ('gaussian_noise', {'sigma': '30'}),
    ('salt_pepper', {'amount': '0.05'}),
    ('brightness_contrast', {'brightness': '50', 'contrast': '1.5'}),
    ('color_balance', {'r': '1.5', 'g': '0.8', 'b': '1.2'}),
    ('hue_saturation', {'hue_shift': '15', 'sat_factor': '1.5'}),
    ('flip', {'direction': 'vertical'}),
    ('rotate', {'angle': '45'}),
    ('resize', {'width_pct': '50', 'height_pct': '50'}),
    ('erosion', {'kernel': '5'}),
    ('dilation', {'kernel': '5'}),
    ('opening', {'kernel': '5'}),
    ('closing', {'kernel': '5'}),
    ('lowpass', {'radius': '20'}),
    ('highpass', {'radius': '20'}),
    ('denoise_gaussian', {'strength': '10'}),
    ('denoise_median', {'kernel': '5'}),
]

for op, params in ops:
    post_form(BASE + "/api/reset", {})
    fields = {"operation": op}
    fields.update({k: str(v) for k, v in params.items()})
    r = post_form(BASE + "/api/apply", fields)
    check(op + " returns 200", r.status == 200)
    cur_md5 = get_image_md5()
    check(op + " changes image bytes", cur_md5 != base_md5,
          "same MD5: " + cur_md5[:8])

# 5. FFT and histogram (need larger image for FFT)
print("\n--- 5. FFT + Histogram ---")
post_form(BASE + "/api/reset", {})
# Upload a larger, more complex image
big_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
_, big_buf = cv2.imencode('.png', big_img)
r = post_multipart(BASE + "/api/upload", {'image': ('big.png', big_buf.tobytes(), 'image/png')})
get(r.url)
big_md5 = get_image_md5()

r = post_form(BASE + "/api/apply", {"operation": "fft_spectrum"})
check("fft_spectrum returns 200", r.status == 200)
check("fft_spectrum changes image bytes", get_image_md5() != big_md5)

post_form(BASE + "/api/reset", {})
r = post_form(BASE + "/api/apply", {"operation": "histogram_equalize"})
check("histogram_equalize returns 200", r.status == 200)
check("histogram_equalize changes image bytes", get_image_md5() != big_md5)

# 6. Filter stack management
print("\n--- 6. Filter stack ---")
post_form(BASE + "/api/reset", {})
post_form(BASE + "/api/apply", {"operation": "negative"})
r_html = get(BASE + "/editor").read().decode()
remove_match = re.search(r'hx-post="/api/remove/([^"]+)"', r_html)
if remove_match:
    fid = remove_match.group(1)
    r = post_form(BASE + "/api/remove/" + fid, {})
    check("Remove filter returns 200", r.status == 200)
else:
    check("Remove button found", False, "no hx-post remove in HTML")

# 7. 404
print("\n--- 7. Error pages ---")
r404 = get(BASE + "/nonexistent-page")
check("404 returns 404 status", r404.status == 404)
check("404 page has content", len(r404.read()) > 100)

# 8. Download
print("\n--- 8. Download ---")
post_form(BASE + "/api/reset", {})
rdl = get(BASE + "/download")
check("Download returns 200", rdl.status == 200)
check("Download is PNG", rdl.headers.get('Content-Type') == 'image/png')

# Summary
print("\n" + "=" * 60)
total = PASS + FAIL
print("RESULTS: " + str(PASS) + "/" + str(total) + " passed, " + str(FAIL) + " failed")
if ERRORS:
    print("\nFAILED:")
    for e in ERRORS:
        print("  - " + e)
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
