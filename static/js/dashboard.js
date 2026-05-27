/**
 * SentinelAI — dashboard.js
 * Handles: image/video upload, webcam capture, detection API calls, results rendering
 */

/* ══════════════════════════════════════════
   Selectors
   ══════════════════════════════════════════ */
const modeTabs      = document.querySelectorAll('.mode-tab');
const panels        = document.querySelectorAll('.detection-panel');
const loadingOverlay= document.getElementById('loadingOverlay');
const loadingText   = document.getElementById('loadingText');
const resultsSection= document.getElementById('resultsSection');
const resultsGrid   = document.getElementById('resultsGrid');
const resultCount   = document.getElementById('resultCount');
const alertBanner   = document.getElementById('alertBanner');
const alertMessage  = document.getElementById('alertMessage');
const alertSound    = document.getElementById('alertSound');

/* ── Image elements ── */
const imageFile       = document.getElementById('imageFile');
const imageDropZone   = document.getElementById('imageDropZone');
const imagePreviewWrap= document.getElementById('imagePreviewWrap');
const imagePreview    = document.getElementById('imagePreview');
const imageOutput     = document.getElementById('imageOutput');
const imageOutputBox  = document.getElementById('imageOutputBox');
const detectImageBtn  = document.getElementById('detectImageBtn');
const clearImageBtn   = document.getElementById('clearImageBtn');

/* ── Video elements ── */
const videoFile        = document.getElementById('videoFile');
const videoDropZone    = document.getElementById('videoDropZone');
const videoPreviewWrap = document.getElementById('videoPreviewWrap');
const videoPreview     = document.getElementById('videoPreview');
const videoOutput      = document.getElementById('videoOutput');
const videoOutputBox   = document.getElementById('videoOutputBox');
const detectVideoBtn   = document.getElementById('detectVideoBtn');
const clearVideoBtn    = document.getElementById('clearVideoBtn');
const videoProgress    = document.getElementById('videoProgress');
const videoProgressFill= document.getElementById('videoProgressFill');

/* ── Webcam elements ── */
const webcamFeed       = document.getElementById('webcamFeed');
const webcamCanvas     = document.getElementById('webcamCanvas');
const webcamPlaceholder= document.getElementById('webcamPlaceholder');
const startCamBtn      = document.getElementById('startCamBtn');
const captureBtn       = document.getElementById('captureBtn');
const stopCamBtn       = document.getElementById('stopCamBtn');
const webcamOutputBox  = document.getElementById('webcamOutputBox');
const webcamOutput     = document.getElementById('webcamOutput');

let cameraStream = null;

/* ══════════════════════════════════════════
   Tab switching
   ══════════════════════════════════════════ */
modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    modeTabs.forEach(t => t.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`panel-${tab.dataset.mode}`).classList.add('active');

    // Stop camera when leaving webcam tab
    if (tab.dataset.mode !== 'webcam' && cameraStream) stopCamera();
    hideResults();
  });
});

/* ══════════════════════════════════════════
   Image upload & detection
   ══════════════════════════════════════════ */
imageDropZone.addEventListener('click', () => imageFile.click());
setupDragDrop(imageDropZone, handleImageFile);
imageFile.addEventListener('change', () => {
  if (imageFile.files[0]) handleImageFile(imageFile.files[0]);
});

function handleImageFile(file) {
  if (!file.type.startsWith('image/')) {
    return alert('Please select a valid image file.');
  }
  const url = URL.createObjectURL(file);
  imagePreview.src = url;
  imageOutputBox.style.display = 'none';
  imagePreviewWrap.classList.remove('hidden');
  imageDropZone.style.display = 'none';
  hideResults();
}

detectImageBtn.addEventListener('click', async () => {
  if (!imageFile.files[0]) return alert('No image selected.');

  const fd = new FormData();
  fd.append('file', imageFile.files[0]);
  fd.append('type', 'image');

  showLoading('Analysing image with YOLOv8…');
  try {
    const data = await postDetect(fd);
    imageOutput.src = data.output_url;
    imageOutputBox.style.display = 'block';
    renderResults(data);
  } catch(e) {
    alert('Detection failed: ' + e.message);
  } finally {
    hideLoading();
  }
});

clearImageBtn.addEventListener('click', () => {
  imageFile.value = '';
  imagePreviewWrap.classList.add('hidden');
  imageDropZone.style.display = '';
  hideResults();
});

/* ══════════════════════════════════════════
   Video upload & detection
   ══════════════════════════════════════════ */
videoDropZone.addEventListener('click', () => videoFile.click());
setupDragDrop(videoDropZone, handleVideoFile);
videoFile.addEventListener('change', () => {
  if (videoFile.files[0]) handleVideoFile(videoFile.files[0]);
});

function handleVideoFile(file) {
  if (!file.type.startsWith('video/')) {
    return alert('Please select a valid video file.');
  }
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  videoOutputBox.style.display = 'none';
  videoPreviewWrap.classList.remove('hidden');
  videoDropZone.style.display = 'none';
  videoProgress.classList.add('hidden');
  hideResults();
}

detectVideoBtn.addEventListener('click', async () => {
  if (!videoFile.files[0]) return alert('No video selected.');

  const fd = new FormData();
  fd.append('file', videoFile.files[0]);
  fd.append('type', 'video');

  // Show progress bar with fake progress animation
  videoProgress.classList.remove('hidden');
  animateProgress(videoProgressFill, 90, 15000); // ~15s fake

  showLoading('Processing video frames — this may take a moment…');
  try {
    const data = await postDetect(fd);
    videoProgressFill.style.width = '100%';
    videoOutput.src = data.output_url;
    videoOutputBox.style.display = 'block';
    renderResults(data);
    setTimeout(() => videoProgress.classList.add('hidden'), 800);
  } catch(e) {
    alert('Detection failed: ' + e.message);
  } finally {
    hideLoading();
  }
});

clearVideoBtn.addEventListener('click', () => {
  videoFile.value = '';
  videoPreviewWrap.classList.add('hidden');
  videoDropZone.style.display = '';
  videoProgress.classList.add('hidden');
  hideResults();
});

/* ══════════════════════════════════════════
   Webcam
   ══════════════════════════════════════════ */
startCamBtn.addEventListener('click', async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamFeed.srcObject = cameraStream;
    webcamPlaceholder.style.display = 'none';
    captureBtn.disabled = false;
    stopCamBtn.disabled  = false;
    startCamBtn.disabled = true;
  } catch(e) {
    alert('Cannot access camera: ' + e.message);
  }
});

stopCamBtn.addEventListener('click', stopCamera);

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  webcamFeed.srcObject = null;
  webcamPlaceholder.style.display = '';
  captureBtn.disabled  = true;
  stopCamBtn.disabled  = true;
  startCamBtn.disabled = false;
}

captureBtn.addEventListener('click', async () => {
  // Draw current frame to canvas → base64
  webcamCanvas.width  = webcamFeed.videoWidth;
  webcamCanvas.height = webcamFeed.videoHeight;
  webcamCanvas.getContext('2d').drawImage(webcamFeed, 0, 0);
  const dataURL = webcamCanvas.toDataURL('image/jpeg', 0.9);

  const fd = new FormData();
  fd.append('type',       'webcam');
  fd.append('frame_data', dataURL);

  showLoading('Analysing captured frame…');
  try {
    const data = await postDetect(fd);
    webcamOutput.src = data.output_url;
    webcamOutputBox.classList.remove('hidden');
    renderResults(data);
  } catch(e) {
    alert('Detection failed: ' + e.message);
  } finally {
    hideLoading();
  }
});

/* ══════════════════════════════════════════
   API helper
   ══════════════════════════════════════════ */
async function postDetect(formData) {
  const resp = await fetch('/detect', { method: 'POST', body: formData });
  const json = await resp.json();
  if (!resp.ok) throw new Error(json.error || 'Server error');
  return json;
}

/* ══════════════════════════════════════════
   Results rendering
   ══════════════════════════════════════════ */
function renderResults(data) {
  const { detections = [], alerts = [], count = 0 } = data;

  resultsSection.classList.remove('hidden');
  resultCount.textContent = `${count} object${count !== 1 ? 's' : ''} detected`;

  resultsGrid.innerHTML = '';

  if (detections.length === 0) {
    resultsGrid.innerHTML = '<p style="color:var(--c-muted);font-size:.9rem;">No objects detected.</p>';
  } else {
    detections.forEach(d => {
      const isAlert = alerts.some(a => a.class === d.class && a.confidence === d.confidence);
      const item = document.createElement('div');
      item.className = 'result-item' + (isAlert ? ' is-alert' : '');
      item.innerHTML = `
        <div class="result-class">${d.class}</div>
        <div class="result-conf ${isAlert ? 'alert' : ''}">${(d.confidence * 100).toFixed(1)}%</div>
        ${isAlert ? '<span class="result-alert-tag">⚠ Alert</span>' : ''}
      `;
      resultsGrid.appendChild(item);
    });
  }

  // Alert banner & sound
  if (alerts.length > 0) {
    const names = [...new Set(alerts.map(a => a.class))].join(', ');
    alertMessage.textContent = `Detected alert-class object(s): ${names}`;
    alertBanner.classList.remove('hidden');

    // Play alert sound (ignore if file missing)
    if (alertSound) {
      alertSound.currentTime = 0;
      alertSound.play().catch(() => {});
    }
  } else {
    alertBanner.classList.add('hidden');
  }

  // Scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
  resultsSection.classList.add('hidden');
  alertBanner.classList.add('hidden');
  resultsGrid.innerHTML = '';
}

/* ══════════════════════════════════════════
   Loading helpers
   ══════════════════════════════════════════ */
function showLoading(text = 'Processing…') {
  loadingText.textContent = text;
  loadingOverlay.classList.remove('hidden');
}
function hideLoading() {
  loadingOverlay.classList.add('hidden');
}

/* ══════════════════════════════════════════
   Drag-and-drop helper
   ══════════════════════════════════════════ */
function setupDragDrop(zone, handler) {
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handler(file);
  });
}

/* ══════════════════════════════════════════
   Progress animation
   ══════════════════════════════════════════ */
function animateProgress(el, targetPct, durationMs) {
  const start   = Date.now();
  const current = parseFloat(el.style.width) || 0;

  function tick() {
    const elapsed  = Date.now() - start;
    const progress = Math.min(elapsed / durationMs, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    el.style.width = (current + (targetPct - current) * eased) + '%';
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
