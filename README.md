# PixelLab - Digital Image Processing Studio

PixelLab is a comprehensive digital image processing web application built with Python, Flask, OpenCV, and modern web technologies. It provides a robust suite of image manipulation tools, real-time object detection, and deepfake analysis in a secure, efficient environment designed for both educational and practical use.

## Features

### 1. Digital Image Processing Engine
- **36 Supported Operations:** A full suite of geometric transforms, point operations, spatial smoothing, sharpening, edge detection, noise models, morphological operations, and frequency domain filtering.
- **Academic Rigor:** Built with educational panels (IIT-H Virtual Lab style) to explain the mathematics and algorithms behind each filter.
- **Scratch Implementations:** Pure-NumPy implementations available for core operations (Negative, Gamma, Otsu, Histogram Equalization, FFT) bypassing OpenCV for pedagogical demonstration.
- **Non-Destructive Stack:** Filter history is preserved and replayed losslessly from the original image using a deterministic parameter stack.

### 2. Live Camera & Real-Time Object Detection
- **Google MediaPipe Integration:** Blazing-fast, client-side object detection (EfficientDet-Lite0) running entirely in the browser (30-60 FPS).
- **Zero Server RAM Cost:** Real-time inferencing is completely offloaded from the server, making it perfectly suited for resource-constrained deployments like the Render Free Tier.

### 3. AI-Generated & Deepfake Image Detection
- **SightEngine API Integration:** Uses the industry-leading [SightEngine API](https://sightengine.com/) for detecting AI-generated images (Stable Diffusion, MidJourney, DALL-E), deepfakes (face swaps), and classifying photos vs. illustrations.
- **Secure Architecture:** The Flask backend acts as a secure proxy. API credentials remain safe on the server and are never exposed to the client.
- **Built-in Throttling & Caching:** A custom server-side caching mechanism (SHA-256) and a 1 request/second throttle protect the free-tier API limits (2,000 ops/month).

## Architecture & Tech Stack

- **Backend**: Python 3.11, Flask, Gunicorn
- **Image Processing**: OpenCV (headless), NumPy, Pillow
- **Frontend**: HTML5, Tailwind CSS, HTMX, Vanilla JS
- **Object Detection**: Google MediaPipe (WebAssembly)
- **AI Detection API**: SightEngine

## Deployment

The application is highly optimized for PaaS environments with strict resource constraints (e.g., Render.com's 512MB free tier).

- **Memory Efficiency:** Idle memory footprint is ~91MB. Under load, it operates safely within ~341MB, utilizing lazy session garbage collection and in-memory caches.
- **Deployment Ready:** Includes `Procfile`, `runtime.txt`, and a curated `requirements.txt` specifically designed to avoid graphical library conflicts on Linux environments (via `opencv-python-headless`).

## Setup Instructions

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your SightEngine API credentials in a `.env` file at the root:
   ```env
   api_user=YOUR_API_USER
   api_secret=YOUR_API_SECRET
   ```
4. Run the application locally: `python app.py`

## Testing

A comprehensive integration test suite covers all application routes, operations, and noise determinism.

```bash
python test_integration.py
```
*(All 47 tests pass successfully)*
