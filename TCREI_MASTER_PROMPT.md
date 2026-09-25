# TCREI Master Prompt — PixelLab Image Processing Virtual Lab

## TASK
Continue development of PixelLab, an Interactive Digital Image Processing Virtual Lab deployed on Render.com free tier. Fix remaining bugs, complete missing features, and polish the UI.

## CONTEXT

### What is PixelLab?
A Flask + HTMX + Tailwind web app for image processing. Two distinct workspaces:
- **Editor** (`/editor`): Clean photo editor (Google Photos style) with Adjust/Transform/Filters/History tabs
- **Lab** (`/lab`): IIT-H experiment structure with operations sidebar + Info Panel (Aim/Theory/Procedure/Observations)

### Tech Stack (FIXED — do not change)
- Backend: Python 3 + Flask 3.1.1 + Gunicorn (1 worker)
- Frontend: Tailwind CSS (CDN) + HTMX 1.9 (CDN) + Chart.js 4.4 (CDN)
- Image Processing: OpenCV-headless + NumPy
- Hosting: Render.com free tier (512MB RAM, 0.1 CPU, sleeps after 15min)
- Session: In-memory Python dict, cookie `sid`, lazy TTL cleanup (30min)
- Database: NONE
- Build step: NONE
- npm: NONE

### Current State (verified working)
- 36 operations across 12 categories (35 original + temperature)
- All operations have working controls/sliders
- Editor: Adjust/Transform/Filters/History tabs, Undo, Reset, Save
- Lab: Operations sidebar + Info Panel + Undo/Reset
- Dark/Light mode toggle (localStorage persisted, CSS variables)
- Nav bar highlights active page
- Download page works without redirect
- All 47 integration tests pass, 75/75 browser flow tests pass

### Key Architecture
- `app.py` — Flask routes (page routes + HTMX fragment routes)
- `config.py` — 36-operation registry with params, controls, theory panels
- `core/processor.py` — ImageProcessor with `_op_*` handlers (auto-discovered via getattr)
- `templates/base.html` — Tailwind CDN, CSS variables for dark/light, nav, theme toggle
- `templates/editor.html` — Photo editor layout (tabs, no sidebar)
- `templates/lab.html` — IIT-H experiment layout (sidebar + info panel)
- `templates/partials/editor_content.html` — HTMX swap target with image + toolbar
- `templates/partials/controls/*.html` — 18 control templates for parameterized ops
- `static/js/app.js` — HTMX error handler, Chart.js lifecycle, toast notifications

### Reference Implementation
A React/TypeScript version exists at `C:\Users\ks919\Downloads\PixelLab-Image-Studio\PixelLab-Image-Studio`. Key patterns worth referencing:
- Editor: Non-destructive sliders with instant preview + 20-state undo stack
- Lab: Sequential operations with explicit "Apply" to chain
- Before/After comparison toggle in both workspaces
- Presets as pill buttons (Soft light, Crisp daylight, Mono study)
- Operations grouped by: Tone, Filters, Edges, Detect
- Side-by-side layout: image preview (left) + controls sidebar (right, 340px)
- Responsive: sticky sidebar on XL, hamburger on mobile

## RULES

### Deployment Constraints
1. Render.com free tier: 512MB RAM, single worker. Memory is the hard limit.
2. No database. Images stored in-memory only. Sessions die on server restart.
3. Max 5 concurrent sessions. Max 2048px on upload. FFT auto-downscales to 1024px.
4. opencv-python-headless ONLY. Never plain opencv-python.
5. SECRET_KEY stays set in app.config. Removing it kills sessions.

### Code Discipline
1. ADDITIVE only — add files, routes, registry entries. Never rewrite working code.
2. One logical change at a time. App must run after each change.
3. Before editing ANY file, read it completely first.
4. Before renaming/renaming anything, grep the whole repo for usages.
5. Every `<img>` tag needs cache-buster: `?t={{ render_count }}` or `?v={{ state.ver }}`.
6. HTMX partials (fragments) return WITHOUT base.html. Page routes return WITH base.html.
7. Form field names must exactly match operation param keys in config.py.
8. Noise ops: `np.random.default_rng(seed)` for determinism on replay.
9. Chart.js: destroy instance before re-creating on HTMX swap.
10. Scripts in HTMX-swapped partials re-run on every swap — guard against double-binding.

### Testing
1. Run `python test_integration.py` after every change (47 tests).
2. Run `tests/browser_flow_test.py` for HTTP flow verification (75 tests).
3. Manual browser check: upload image, apply each tab's tools, verify visual change.
4. Check both dark AND light mode after any UI change.

### Git
1. Commit after every verified working state.
2. Broken >15 min? `git checkout` last working commit.
3. Never regenerate codebase as a "fix."

## EXAMPLES

### Adding a new operation
1. Add to OPERATIONS dict in `config.py`:
```python
'new_op': {
    'category': 'point', 'label': 'New Operation',
    'input': 'any', 'output': 'pass',
    'params': {'param1': default_value},
    'controls': 'controls/new_op.html',  # if parameterized
},
```
2. Add handler in `core/processor.py`:
```python
@staticmethod
def _op_new_op(img, params):
    val = params.get('param1', default_value)
    # processing logic
    return result
```
3. Create `templates/partials/controls/new_op.html` if needed.
4. Test: `POST /api/apply` with `operation=new_op&param1=value`.

### HTMX fragment route pattern
```python
@app.route('/api/apply', methods=['POST'])
def apply_operation():
    s = get_session()
    # ... process ...
    return render_template('partials/editor_content.html',
                           operations=OPERATIONS, theory=THEORY,
                           filter_stack=s['filter_stack'],
                           # ... other context ...
                           histogram=ImageProcessor.get_histogram_data(s['processed']))
```

### CSS variable theme pattern
```css
:root { --bg: #020617; --text: #e2e8f0; }  /* dark */
html:not(.dark) { --bg: #f8fafc; --text: #1e293b; }  /* light */
```
Body/nav use inline `style="background:var(--bg);color:var(--text)"`.
Internal components use Tailwind `bg-surface-*` classes (dark mode only, no dark: variants needed for internal UI).

## INTENT

### What the user wants
1. A working DIP virtual lab that looks professional and feels like a real tool
2. Editor = simple photo editor (like Google Photos), NOT an operations lab
3. Lab = IIT-H experiment structure with educational info panel
4. Both pages must be mobile-friendly
5. Dark and light mode must both work with readable text
6. All 36 operations must be functional with adjustable parameters
7. Deployed on Render.com free tier without crashing

### What NOT to do
1. Do NOT add npm, build steps, or databases
2. Do NOT add features that require >512MB RAM (batch processing, large file handling)
3. Do NOT break existing working features while adding new ones
4. Do NOT create "cosmetic fixes" that hide broken functionality
5. Do NOT claim "100% fixed" without running verification commands with output
6. Do NOT add matplotlib, plotly, or heavy frontend libraries
7. Do NOT rewrite working code while adding features

### Remaining work (priority order)
1. **Mobile responsiveness** — Editor toolbar overflows on small screens, Lab sidebar doesn't collapse
2. **Fix light mode for internal components** — Templates still use hardcoded `bg-surface-*` Tailwind classes
3. **Before/After comparison** — Split-screen slider in Editor (reference: React version has toggle)
4. **Dual histograms** — Show original vs processed pixel distribution
5. **Recipe export/import** — Filter stack is already JSON; export = 10 lines
6. **Pixel inspector** — Hover to see pixel values (needs `/api/pixels` endpoint for accuracy)
