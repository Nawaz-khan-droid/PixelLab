// PixelLab -- HTMX error handling + utilities
document.addEventListener('DOMContentLoaded', () => {
    // HTMX error handler -- show toast on non-2xx responses
    document.body.addEventListener('htmx:responseError', e => {
        const status = e.detail.xhr.status;
        const msg = e.detail.xhr.responseText || 'Request failed';
        showToast(status + ': ' + msg.substring(0, 120), 'error');
    });
    document.body.addEventListener('htmx:sendError', e => {
        showToast('Network error -- check connection', 'error');
    });

    // HTMX loading indicator on triggering element
    document.body.addEventListener('htmx:beforeRequest', e => {
        e.target.classList.add('htmx-request');
    });
    document.body.addEventListener('htmx:afterRequest', e => {
        e.target.classList.remove('htmx-request');
    });

    // Re-init Chart.js after HTMX swap
    document.body.addEventListener('htmx:afterSwap', e => {
        if (e.detail.target && e.detail.target.id === 'editor-content') {
            initHistogram();
            if (window.innerWidth <= 768 && typeof closeLabPanels === 'function') {
                closeLabPanels();
            }
        }
    });
});

function showToast(msg, type) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const el = document.createElement('div');
    const bg = type === 'error' ? 'bg-red-600' : type === 'success' ? 'bg-emerald-600' : 'bg-surface-700';
    el.className = bg + ' text-white px-4 py-2.5 rounded-lg shadow-lg text-sm toast-enter max-w-sm';
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, 4000);
}

let histChart = null;
function initHistogram() {
    const el = document.getElementById('histogramCanvas');
    if (!el) return;
    const data = el.dataset.hist;
    if (!data) return;
    let histData;
    try { histData = JSON.parse(data); } catch(e) { return; }
    if (!histData || Object.keys(histData).length === 0) return;

    if (histChart) { histChart.destroy(); histChart = null; }
    const ctx = el.getContext('2d');
    const labels = Array.from({length: 256}, (_, i) => i);
    const datasets = [];
    if (histData.gray) {
        datasets.push({ label: 'Gray', data: histData.gray, borderColor: 'rgba(148,163,184,0.8)', borderWidth: 1, fill: false, pointRadius: 0 });
    } else {
        if (histData.r) datasets.push({ label: 'R', data: histData.r, borderColor: 'rgba(239,68,68,0.8)', borderWidth: 1, fill: false, pointRadius: 0 });
        if (histData.g) datasets.push({ label: 'G', data: histData.g, borderColor: 'rgba(34,197,94,0.8)', borderWidth: 1, fill: false, pointRadius: 0 });
        if (histData.b) datasets.push({ label: 'B', data: histData.b, borderColor: 'rgba(59,130,246,0.8)', borderWidth: 1, fill: false, pointRadius: 0 });
    }
    if (datasets.length === 0) return;
    histChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: datasets.length > 1, labels: { color: '#94a3b8', font: { size: 10 } } } },
            scales: { x: { display: false }, y: { display: false, beginAtZero: true } }
        }
    });
}

// Init histogram on page load
document.addEventListener('DOMContentLoaded', () => initHistogram());
