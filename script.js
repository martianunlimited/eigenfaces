/**
 * Eigenfaces & PCA Face Reconstruction Studio
 * Interactive Principal Component Analysis, Dynamic Linear Combinations & Bias Explorer
 */

(function () {
  'use strict';

  // --- APPLICATION STATE ---
  const state = {
    dataset: null,
    currentPersonId: 'george_w_bush',
    k: 20,
    isAnimating: false,
    animInterval: null,
    mixerWeights: [0, 0, 0, 0, 0, 0],
    mixerDescriptions: [
      'Horizontal Illumination / Key Light',
      'Vertical Ambient Light / Forehead',
      'Face Elongation & Jawline Shape',
      'Eye Sockets & Nose Bridge Depth',
      'Mouth Curvature & Smile Feature',
      'Hairline & Upper Contour Boundary'
    ]
  };

  // --- DOM REFERENCES ---
  const el = {
    // Header & Theme
    themeToggleBtn: document.getElementById('themeToggleBtn'),
    themeMoonIcon: document.getElementById('themeMoonIcon'),
    themeSunIcon: document.getElementById('themeSunIcon'),

    // Person Selector & Profile
    personSelect: document.getElementById('personSelect'),
    profileFlag: document.getElementById('profileFlag'),
    profileName: document.getElementById('profileName'),
    profileTrainingBadge: document.getElementById('profileTrainingBadge'),
    profileRole: document.getElementById('profileRole'),
    profileBio: document.getElementById('profileBio'),
    profileDemoTag: document.getElementById('profileDemoTag'),
    profileRegionTag: document.getElementById('profileRegionTag'),

    // K Controls
    kSlider: document.getElementById('kSlider'),
    kValueBubble: document.getElementById('kValueBubble'),
    animToggleBtn: document.getElementById('animToggleBtn'),
    animIcon: document.getElementById('animIcon'),
    animText: document.getElementById('animText'),
    resetKBtn: document.getElementById('resetKBtn'),
    snapButtons: document.querySelectorAll('.snap-btn'),

    // Canvases
    canvasOriginal: document.getElementById('canvasOriginal'),
    canvasRecon: document.getElementById('canvasRecon'),
    canvasDiff: document.getElementById('canvasDiff'),
    reconBadge: document.getElementById('reconBadge'),
    reconEquationDisplay: document.getElementById('reconEquationDisplay'),
    reconStatusText: document.getElementById('reconStatusText'),

    // Telemetry
    metricCompression: document.getElementById('metricCompression'),
    metricCompressionSub: document.getElementById('metricCompressionSub'),
    metricVariance: document.getElementById('metricVariance'),
    metricMSE: document.getElementById('metricMSE'),
    metricMSESub: document.getElementById('metricMSESub'),
    metricPSNR: document.getElementById('metricPSNR'),
    metricCosine: document.getElementById('metricCosine'),

    // Weights Chart
    weightsChart: document.getElementById('weightsChart'),

    // Mixer
    mixerSlidersContainer: document.getElementById('mixerSlidersContainer'),
    canvasMixerPreview: document.getElementById('canvasMixerPreview'),
    mixerPreviewDesc: document.getElementById('mixerPreviewDesc'),
    mixerResetBtn: document.getElementById('mixerResetBtn'),
    mixerZeroBtn: document.getElementById('mixerZeroBtn'),
    mixerRandomBtn: document.getElementById('mixerRandomBtn'),

    // Gallery
    galleryGrid: document.getElementById('galleryGrid'),

    // Bias Comparison Canvases & Chart
    canvasBushComp: document.getElementById('canvasBushComp'),
    canvasMandelaComp: document.getElementById('canvasMandelaComp'),
    canvasMegawatiComp: document.getElementById('canvasMegawatiComp'),
    canvasSerenaComp: document.getElementById('canvasSerenaComp'),
    chartMSE: document.getElementById('chartMSE')
  };

  // --- INITIALIZATION ---
  async function init() {
    setupThemeToggle();
    try {
      const res = await fetch('data/faces_data.json');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.dataset = await res.json();
      console.log('Loaded faces_data.json successfully with', state.dataset.people.length, 'people.');

      setupEventListeners();
      renderGallery();
      initMixerSliders();
      updatePersonProfile();
      updateReconstruction();
      renderBiasComparisonCanvases();
      renderMSEChart();

    } catch (err) {
      console.error('Failed to load faces_data.json:', err);
      showErrorNotice(err);
    }
  }

  function showErrorNotice(err) {
    const banner = document.createElement('div');
    banner.style.cssText = 'background:#f43f5e; color:#fff; padding:1rem; border-radius:8px; margin:1rem 0; text-align:center; font-weight:600;';
    banner.textContent = `Could not load faces_data.json (${err.message}). Ensure you are serving via HTTP (e.g. npx serve or python3 -m http.server).`;
    document.querySelector('.main-layout').prepend(banner);
  }

  // --- THEME SWITCHER ---
  function setupThemeToggle() {
    el.themeToggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', newTheme);
      if (newTheme === 'light') {
        el.themeMoonIcon.classList.add('hidden');
        el.themeSunIcon.classList.remove('hidden');
      } else {
        el.themeSunIcon.classList.add('hidden');
        el.themeMoonIcon.classList.remove('hidden');
      }
      renderMSEChart(); // Re-render chart with appropriate grid lines
    });
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    // Person selection change
    el.personSelect.addEventListener('change', (e) => {
      state.currentPersonId = e.target.value;
      updatePersonProfile();
      updateReconstruction();
      syncMixerWithCurrentPerson();
    });

    // K Slider Input
    el.kSlider.addEventListener('input', (e) => {
      stopAnimation();
      setK(parseInt(e.target.value, 10));
    });

    // Preset snapshot buttons
    el.snapButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        stopAnimation();
        const targetK = parseInt(btn.getAttribute('data-k'), 10);
        setK(targetK);
      });
    });

    // Reset K Button
    el.resetKBtn.addEventListener('click', () => {
      stopAnimation();
      setK(0);
    });

    // Animate button
    el.animToggleBtn.addEventListener('click', toggleAnimation);

    // Mixer action buttons
    el.mixerResetBtn.addEventListener('click', syncMixerWithCurrentPerson);
    el.mixerZeroBtn.addEventListener('click', () => {
      state.mixerWeights = [0, 0, 0, 0, 0, 0];
      updateMixerSlidersUI();
      renderMixerCanvas();
    });
    el.mixerRandomBtn.addEventListener('click', () => {
      // Draw weights from realistic Gaussian range for the first 6 components
      state.mixerWeights = state.mixerWeights.map(() => {
        return (Math.random() * 6 - 3).toFixed(2);
      });
      updateMixerSlidersUI();
      renderMixerCanvas();
    });
  }

  function setK(newK) {
    state.k = Math.max(0, Math.min(80, newK));
    el.kSlider.value = state.k;

    // Update slider progress background
    const progress = (state.k / 80) * 100;
    el.kSlider.style.setProperty('--progress', `${progress}%`);

    // Update active snapshot pill
    el.snapButtons.forEach(btn => {
      const match = parseInt(btn.getAttribute('data-k'), 10) === state.k;
      btn.classList.toggle('active', match);
    });

    updateReconstruction();
  }

  function toggleAnimation() {
    if (state.isAnimating) {
      stopAnimation();
    } else {
      startAnimation();
    }
  }

  function startAnimation() {
    state.isAnimating = true;
    el.animIcon.textContent = '⏸';
    el.animText.textContent = 'Pause Animation';
    el.animToggleBtn.classList.add('active');

    if (state.k >= 80) {
      setK(0);
    }

    state.animInterval = setInterval(() => {
      if (state.k < 80) {
        setK(state.k + 1);
      } else {
        stopAnimation();
      }
    }, 45);
  }

  function stopAnimation() {
    state.isAnimating = false;
    el.animIcon.textContent = '▶';
    el.animText.textContent = 'Animate Emergence';
    el.animToggleBtn.classList.remove('active');
    if (state.animInterval) {
      clearInterval(state.animInterval);
      state.animInterval = null;
    }
  }

  // --- PERSON PROFILE UPDATE ---
  function getCurrentPerson() {
    return state.dataset.people.find(p => p.id === state.currentPersonId) || state.dataset.people[0];
  }

  function updatePersonProfile() {
    const person = getCurrentPerson();
    el.profileFlag.textContent = person.flag;
    el.profileName.textContent = person.name;
    el.profileRole.textContent = person.role;
    el.profileBio.textContent = person.bio;
    el.profileDemoTag.textContent = person.demographic;
    el.profileRegionTag.textContent = `${person.region} • ${person.country}`;

    if (person.inTrainingSet) {
      el.profileTrainingBadge.textContent = `In scikit-learn Training Set (${person.trainingImagesCount} photos)`;
      el.profileTrainingBadge.classList.remove('tag-ood');
    } else {
      el.profileTrainingBadge.textContent = `Out-of-Distribution (${person.trainingImagesCount} total photos in LFW)`;
      el.profileTrainingBadge.classList.add('tag-ood');
    }
  }

  // --- CORE PCA RECONSTRUCTION ENGINE ---
  function updateReconstruction() {
    if (!state.dataset) return;

    const person = getCurrentPerson();
    const meanFace = state.dataset.mean_face; // Float array (1850)
    const eigenfaces = state.dataset.eigenfaces; // 80 arrays of (1850)
    const weights = person.weights; // Float array (80)
    const originalPixels = person.pixels; // Float array (1850)
    const K = state.k;

    // 1. Compute Linear Combination: recon = mean_face + sum_{k=0}^{K-1} w_k * ef_k
    const recon = new Float32Array(1850);
    for (let i = 0; i < 1850; i++) {
      let val = meanFace[i];
      for (let k = 0; k < K; k++) {
        val += weights[k] * eigenfaces[k][i];
      }
      recon[i] = Math.max(0, Math.min(1.0, val));
    }

    // 2. Render Canvases
    renderGrayscaleCanvas(el.canvasOriginal, originalPixels);
    renderGrayscaleCanvas(el.canvasRecon, recon);
    renderDifferenceHeatmap(el.canvasDiff, originalPixels, recon);

    // 3. Compute Metrics
    let sumSqErr = 0;
    let dotProd = 0;
    let normOrig = 0;
    let normRecon = 0;

    for (let i = 0; i < 1850; i++) {
      const diff = originalPixels[i] - recon[i];
      sumSqErr += diff * diff;
      dotProd += originalPixels[i] * recon[i];
      normOrig += originalPixels[i] * originalPixels[i];
      normRecon += recon[i] * recon[i];
    }

    const mse = sumSqErr / 1850;
    const psnr = mse > 1e-9 ? (10 * Math.log10(1.0 / mse)) : 99.9;
    const cosine = (normOrig > 0 && normRecon > 0) ? (dotProd / (Math.sqrt(normOrig) * Math.sqrt(normRecon))) : 1.0;
    const compressionRatio = K > 0 ? (1850 / K).toFixed(1) : '∞';
    const varExplained = K > 0 ? (state.dataset.cumulative_variance[K - 1] * 100).toFixed(1) + '%' : '0.0%';

    // 4. Update Telemetry UI
    el.kValueBubble.textContent = `${K} ${K === 1 ? 'component' : 'components'}`;
    el.reconBadge.textContent = K === 0 ? 'Mean Face (K = 0)' : `K = ${K} Components`;
    el.reconEquationDisplay.innerHTML = K === 0
      ? `$\\hat{\\mathbf{x}} = \\boldsymbol{\\mu}$ (Average Face)`
      : `$\\hat{\\mathbf{x}} = \\boldsymbol{\\mu} + \\sum_{k=1}^{${K}} w_k \\mathbf{v}_k$`;

    if (K === 0) {
      el.reconStatusText.textContent = 'Generic Centroid';
      el.reconStatusText.style.color = 'var(--text-muted)';
    } else if (K < 10) {
      el.reconStatusText.textContent = 'Coarse Ghost Contours';
      el.reconStatusText.style.color = 'var(--amber)';
    } else if (K < 30) {
      el.reconStatusText.textContent = 'Facial Features Forming';
      el.reconStatusText.style.color = 'var(--accent)';
    } else {
      el.reconStatusText.textContent = 'Recognizable Identity';
      el.reconStatusText.style.color = 'var(--emerald)';
    }

    el.metricCompression.textContent = `${compressionRatio}×`;
    el.metricCompressionSub.textContent = K === 0 ? 'Infinite compression' : `1,850 px → ${K} weights (${((1 - K / 1850) * 100).toFixed(1)}% saved)`;
    el.metricVariance.textContent = varExplained;
    el.metricMSE.textContent = mse.toFixed(5);
    el.metricPSNR.textContent = `${psnr.toFixed(1)} dB`;
    el.metricCosine.textContent = cosine.toFixed(3);

    // Color code MSE based on fidelity
    if (mse < 0.004) {
      el.metricMSE.style.color = 'var(--emerald)';
    } else if (mse < 0.009) {
      el.metricMSE.style.color = 'var(--amber)';
    } else {
      el.metricMSE.style.color = 'var(--rose)';
    }

    // 5. Render Weights Bar Chart
    renderWeightsChart(weights, K);
  }

  // --- CANVAS RENDERING HELPERS ---
  function renderGrayscaleCanvas(canvas, pixelArray) {
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(37, 50);
    const data = imgData.data;

    for (let i = 0; i < 1850; i++) {
      const idx = i * 4;
      const gray = Math.round(pixelArray[i] * 255.0);
      data[idx] = gray;     // R
      data[idx + 1] = gray; // G
      data[idx + 2] = gray; // B
      data[idx + 3] = 255;  // Alpha
    }

    ctx.putImageData(imgData, 0, 0);
  }

  function renderDifferenceHeatmap(canvas, original, recon) {
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(37, 50);
    const data = imgData.data;

    for (let i = 0; i < 1850; i++) {
      const idx = i * 4;
      const diff = Math.abs(original[i] - recon[i]); // Range [0.0, 1.0]

      // Heatmap colormap: Black -> Amber -> Bright Red -> White
      let r = 0, g = 0, b = 0;
      if (diff < 0.05) {
        // Low difference (almost exact match): black to faint warm amber
        const t = diff / 0.05;
        r = Math.round(t * 120);
        g = Math.round(t * 60);
        b = 0;
      } else if (diff < 0.18) {
        // Medium difference: rich amber to vibrant red
        const t = (diff - 0.05) / 0.13;
        r = Math.round(120 + t * 135); // up to 255
        g = Math.round(60 + t * 40);
        b = Math.round(t * 30);
      } else {
        // High difference: fiery red to bright white/yellow
        const t = Math.min(1.0, (diff - 0.18) / 0.3);
        r = 255;
        g = Math.round(100 + t * 155);
        b = Math.round(30 + t * 225);
      }

      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }

    ctx.putImageData(imgData, 0, 0);
  }

  // --- WEIGHTS BAR CHART ---
  function renderWeightsChart(weights, activeK) {
    el.weightsChart.innerHTML = '';
    const displayCount = Math.min(40, weights.length);

    for (let i = 0; i < displayCount; i++) {
      const col = document.createElement('div');
      col.className = 'weight-bar-col';
      col.title = `Eigenface v_${i + 1}: weight = ${weights[i].toFixed(2)} ${i < activeK ? '(Active)' : '(Inactive)'}`;

      const w = weights[i];
      const maxRange = 5.0; // scale relative to standard eigenface weight ranges
      const barHeightPct = Math.min(100, (Math.abs(w) / maxRange) * 100);

      const bar = document.createElement('div');
      if (w >= 0) {
        bar.className = 'weight-bar-pos';
        bar.style.height = `${barHeightPct}%`;
      } else {
        bar.className = 'weight-bar-neg';
        bar.style.height = `${barHeightPct}%`;
      }

      if (i >= activeK) {
        bar.style.opacity = '0.18';
        bar.style.boxShadow = 'none';
      }

      col.appendChild(bar);
      el.weightsChart.appendChild(col);
    }
  }

  // --- EIGENFACE GALLERY RENDERING ---
  function renderGallery() {
    const fragment = document.createDocumentFragment();

    for (let i = 0; i < 16; i++) {
      const card = document.createElement('div');
      card.className = 'ef-card';
      const varPct = (state.dataset.variance_ratio[i] * 100).toFixed(1);

      const featureHints = [
        'Horizontal Illumination / Key Light',
        'Vertical Ambient Light / Forehead',
        'Face Elongation & Jawline Shape',
        'Eye Sockets & Nose Bridge Prominence',
        'Mouth Curvature & Smile Expression',
        'Hairline & Upper Contour Boundary',
        'Eyebrow Contrast & Brow Ridge',
        'Nasolabial Folds & Laugh Lines',
        'Cheek Shadow & Skin Specularity',
        'Nostril Depth & Lip Thickness',
        'Temple Width & Ear Angle',
        'Chin Cleft & Lower Shadow',
        'Eye Aperture & Gaze Intensity',
        'Facial Wrinkles & Skin Texture',
        'Flash Reflection & Specular Sheen',
        'High-Frequency Facial Contours'
      ];

      card.innerHTML = `
        <div class="ef-card-top">
          <span class="ef-tag">EF #${i + 1}</span>
          <span class="ef-var">${varPct}% var</span>
        </div>
        <img src="data/eigenfaces/eigenface_${i + 1}.png" alt="Eigenface ${i + 1}" class="ef-img" loading="lazy">
        <div class="ef-card-footer">
          <span class="ef-name">Eigenface $\\mathbf{v}_{${i + 1}}$</span>
          <span class="ef-desc">${featureHints[i]}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        // Set K to include this eigenface
        setK(i + 1);
      });

      fragment.appendChild(card);
    }

    el.galleryGrid.appendChild(fragment);
  }

  // --- INTERACTIVE SYNTHESIZER / MIXER ---
  function initMixerSliders() {
    el.mixerSlidersContainer.innerHTML = '';
    const person = getCurrentPerson();

    for (let i = 0; i < 6; i++) {
      state.mixerWeights[i] = person.weights[i];

      const card = document.createElement('div');
      card.className = 'mixer-slider-card';
      card.innerHTML = `
        <div class="mixer-slider-header">
          <div>
            <span class="mixer-ef-label">Eigenface $\\mathbf{v}_{${i + 1}}$</span>
            <span class="mixer-ef-desc">${state.mixerDescriptions[i]}</span>
          </div>
          <span class="mixer-val-text" id="mixerValText_${i}">${person.weights[i].toFixed(2)}</span>
        </div>
        <input type="range" id="mixerSlider_${i}" min="-6" max="6" step="0.1" value="${person.weights[i]}" class="range-slider">
      `;

      el.mixerSlidersContainer.appendChild(card);

      const slider = card.querySelector(`#mixerSlider_${i}`);
      const valText = card.querySelector(`#mixerValText_${i}`);

      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        state.mixerWeights[i] = val;
        valText.textContent = val.toFixed(2);
        renderMixerCanvas();
      });
    }

    renderMixerCanvas();
  }

  function syncMixerWithCurrentPerson() {
    const person = getCurrentPerson();
    for (let i = 0; i < 6; i++) {
      state.mixerWeights[i] = person.weights[i];
    }
    updateMixerSlidersUI();
    renderMixerCanvas();
  }

  function updateMixerSlidersUI() {
    for (let i = 0; i < 6; i++) {
      const slider = document.getElementById(`mixerSlider_${i}`);
      const valText = document.getElementById(`mixerValText_${i}`);
      if (slider && valText) {
        slider.value = state.mixerWeights[i];
        valText.textContent = parseFloat(state.mixerWeights[i]).toFixed(2);
        const progress = ((parseFloat(state.mixerWeights[i]) + 6) / 12) * 100;
        slider.style.setProperty('--progress', `${progress}%`);
      }
    }
  }

  function renderMixerCanvas() {
    if (!state.dataset) return;
    const meanFace = state.dataset.mean_face;
    const eigenfaces = state.dataset.eigenfaces;

    const synth = new Float32Array(1850);
    for (let i = 0; i < 1850; i++) {
      let val = meanFace[i];
      for (let k = 0; k < 6; k++) {
        val += state.mixerWeights[k] * eigenfaces[k][i];
      }
      synth[i] = Math.max(0, Math.min(1.0, val));
    }

    renderGrayscaleCanvas(el.canvasMixerPreview, synth);
  }

  // --- BIAS COMPARISON CANVASES ---
  function renderBiasComparisonCanvases() {
    if (!state.dataset) return;
    const targetPeople = [
      { id: 'george_w_bush', canvas: el.canvasBushComp },
      { id: 'nelson_mandela', canvas: el.canvasMandelaComp },
      { id: 'megawati_sukarnoputri', canvas: el.canvasMegawatiComp },
      { id: 'serena_williams', canvas: el.canvasSerenaComp }
    ];

    const meanFace = state.dataset.mean_face;
    const eigenfaces = state.dataset.eigenfaces;
    const K = 20; // Fixed comparison depth

    targetPeople.forEach(({ id, canvas }) => {
      const person = state.dataset.people.find(p => p.id === id);
      if (!person || !canvas) return;

      const recon = new Float32Array(1850);
      for (let i = 0; i < 1850; i++) {
        let val = meanFace[i];
        for (let k = 0; k < K; k++) {
          val += person.weights[k] * eigenfaces[k][i];
        }
        recon[i] = Math.max(0, Math.min(1.0, val));
      }

      renderGrayscaleCanvas(canvas, recon);
    });
  }

  // --- MSE GENERALIZATION GAP CHART (HTML5 CANVAS) ---
  function renderMSEChart() {
    if (!state.dataset) return;
    const canvas = el.chartMSE;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const gridColor = isLight ? 'rgba(100, 116, 139, 0.15)' : 'rgba(255, 255, 255, 0.08)';
    const textColor = isLight ? '#64748b' : '#94a3b8';

    ctx.clearRect(0, 0, width, height);

    // Padding
    const pLeft = 60;
    const pRight = 20;
    const pTop = 20;
    const pBottom = 40;

    const chartW = width - pLeft - pRight;
    const chartH = height - pTop - pBottom;

    // Max MSE across curves for Y scaling
    const maxMSE = 0.025; // fixed scale for easy reading
    const kSteps = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80];

    // Helper coordinates
    function getX(kVal) {
      return pLeft + (kVal / 80) * chartW;
    }
    function getY(mseVal) {
      return pTop + chartH - (mseVal / maxMSE) * chartH;
    }

    // 1. Draw Grid Lines & Y-axis labels
    ctx.lineWidth = 1;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = textColor;
    ctx.textAlign = 'right';

    for (let mse = 0; mse <= maxMSE; mse += 0.005) {
      const y = getY(mse);
      ctx.strokeStyle = gridColor;
      ctx.beginPath();
      ctx.moveTo(pLeft, y);
      ctx.lineTo(width - pRight, y);
      ctx.stroke();

      ctx.fillText(mse.toFixed(3), pLeft - 8, y + 4);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    for (let k = 0; k <= 80; k += 10) {
      const x = getX(k);
      ctx.strokeStyle = gridColor;
      ctx.beginPath();
      ctx.moveTo(x, pTop);
      ctx.lineTo(x, pTop + chartH);
      ctx.stroke();

      ctx.fillText(`K=${k}`, x, pTop + chartH + 18);
    }

    // 2. Draw Curves for 4 People
    const series = [
      { id: 'george_w_bush', color: '#10b981', width: 2.8 },
      { id: 'nelson_mandela', color: '#06b6d4', width: 2.2 },
      { id: 'megawati_sukarnoputri', color: '#f43f5e', width: 2.8 },
      { id: 'serena_williams', color: '#f59e0b', width: 2.2 }
    ];

    series.forEach(({ id, color, width: strokeW }) => {
      const person = state.dataset.people.find(p => p.id === id);
      if (!person) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = strokeW;
      ctx.beginPath();

      kSteps.forEach((k, idx) => {
        const mse = person.mse_curve[k.toString()];
        const x = getX(k);
        const y = getY(mse);

        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw dots
      ctx.fillStyle = color;
      kSteps.forEach(k => {
        const mse = person.mse_curve[k.toString()];
        const x = getX(k);
        const y = getY(mse);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    // 3. Current K Marker Line
    if (state.k > 0) {
      const currX = getX(state.k);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(currX, pTop);
      ctx.lineTo(currX, pTop + chartH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px "Outfit", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Current K = ${state.k}`, currX, pTop - 6);
    }
  }

  // --- START WHEN DOM READY ---
  document.addEventListener('DOMContentLoaded', init);

})();
