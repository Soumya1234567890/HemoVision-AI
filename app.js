/**
 * HemoVision AI — Clinical Hemoglobinopathy Diagnostic & XAI Application
 * Pure Vanilla JavaScript Client Logic
 */

// Application State
const state = {
  modelInfo: null,
  patientData: {},
  latestPrediction: null,
  cohortData: [],
  boundaryData: null,
  rbcChart: null,
  rdwChart: null
};

// DOM Elements
const patientForm = document.getElementById('patientForm');
const mentzerDisplay = document.getElementById('mentzerDisplay');
const mentzerPill = document.getElementById('mentzerPill');
const btnRunInference = document.getElementById('btnRunInference');
const btnResetInputs = document.getElementById('btnResetInputs');
const presetSelect = document.getElementById('presetSelect');
const btnRandomCohort = document.getElementById('btnRandomCohort');
const btnExportReport = document.getElementById('btnExportReport');
const reportModal = document.getElementById('reportModal');
const btnCloseModal = document.getElementById('btnCloseModal');

// Canvas Elements
const canvas = document.getElementById('decisionCanvas');
const canvasWrapper = document.getElementById('canvasWrapper');
const canvasTooltip = document.getElementById('canvasTooltip');
const axisXSelect = document.getElementById('axisX');
const axisYSelect = document.getElementById('axisY');
const btnRecalculateBoundary = document.getElementById('btnRecalculateBoundary');

// Initialize Application
async function initApp() {
  try {
    // 1. Fetch Model Metadata
    const infoRes = await fetch('/api/info');
    state.modelInfo = await infoRes.json();

    // 2. Build Patient Input Form
    buildInputControls(state.modelInfo.features);

    // 3. Load Clinical Presets
    loadPresets();

    // 4. Fetch Cohort Data
    loadCohort('ida_btt');

    // 5. Setup Tab Navigation
    setupTabs();

    // 6. Setup Canvas Event Listeners
    setupCanvas();

    // 7. Initial Inference Run
    await runInference();

  } catch (err) {
    console.error('Initialization error:', err);
  }
}

// Dynamically construct input sliders & numeric inputs
function buildInputControls(features) {
  patientForm.innerHTML = '';
  
  for (const [key, meta] of Object.entries(features)) {
    // Skip Mentzer Index in manual sliders because it's auto-computed from MCV / RBC
    if (key === 'Mentzer Index') continue;

    state.patientData[key] = meta.default;

    const item = document.createElement('div');
    item.className = 'feature-item';
    item.id = `item_${key.replace(/[^a-zA-Z0-9]/g, '_')}`;

    item.innerHTML = `
      <div class="feature-header">
        <div>
          <span class="feature-label">${meta.label}</span>
          <span class="feature-unit">(${meta.unit})</span>
        </div>
        <span id="range_badge_${key.replace(/[^a-zA-Z0-9]/g, '_')}" style="font-size: 0.65rem; color: var(--text-muted);">
          Ref: ${meta.normal_range[0]} - ${meta.normal_range[1]}
        </span>
      </div>
      <div class="feature-input-group">
        <input type="range" class="feature-slider" 
               id="slider_${key.replace(/[^a-zA-Z0-9]/g, '_')}"
               min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${meta.default}">
        <input type="number" class="feature-num-input" 
               id="input_${key.replace(/[^a-zA-Z0-9]/g, '_')}"
               min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${meta.default}">
      </div>
      <div class="ref-range-info">
        <span>Min: ${meta.min}</span>
        <span>Max: ${meta.max}</span>
      </div>
    `;

    patientForm.appendChild(item);

    const slider = item.querySelector('.feature-slider');
    const numInput = item.querySelector('.feature-num-input');

    // Sync Slider and Number Input
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      numInput.value = val;
      state.patientData[key] = val;
      updateCalculatedMentzer();
    });

    numInput.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val)) {
        slider.value = val;
        state.patientData[key] = val;
        updateCalculatedMentzer();
      }
    });

    // Auto-inference on slider release for smooth interactive experience
    slider.addEventListener('change', () => {
      runInference();
    });
  }

  updateCalculatedMentzer();
}

// Compute Mentzer Index (MCV / RBC) in real time
function updateCalculatedMentzer() {
  const mcv = state.patientData['MCV'] || 64.0;
  const rbc = state.patientData['RBC Count'] || 5.5;
  const mentzer = rbc > 0 ? (mcv / rbc).toFixed(1) : '13.0';

  state.patientData['Mentzer Index'] = parseFloat(mentzer);
  mentzerDisplay.textContent = mentzer;

  if (parseFloat(mentzer) < 13.0) {
    mentzerPill.className = 'mentzer-status-pill btt';
    mentzerPill.textContent = 'Rule: <13 (Suggests BTT)';
  } else {
    mentzerPill.className = 'mentzer-status-pill ida';
    mentzerPill.textContent = 'Rule: ≥13 (Suggests IDA)';
  }
}

// Run Diagnostic Inference & Update Explainability Suite
async function runInference() {
  try {
    updateCalculatedMentzer();

    const payload = {
      Age_years: state.patientData['Age_years'],
      "RBC Count": state.patientData['RBC Count'],
      HCT: state.patientData['HCT'],
      Hemoglobin: state.patientData['Hemoglobin'],
      MCV: state.patientData['MCV'],
      MCH: state.patientData['MCH'],
      MCHC: state.patientData['MCHC'],
      "RDW-CV": state.patientData['RDW-CV'],
      "Mentzer Index": state.patientData['Mentzer Index']
    };

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Inference API call failed');
    const result = await res.json();
    state.latestPrediction = result;

    // Update Verdict Hero Card
    updateVerdictUI(result);

    // Update Tab 1: SHAP Waterfall
    renderShapWaterfall(result.shap_analysis, result.clinical_narratives);

    // Update Tab 3: Support Vector Exemplars
    renderExemplars(result.nearest_support_vectors);

    // Update Tab 4: What-If Sensitivity Charts
    renderSensitivity(result.sensitivity, result.diagnosis);

    // Update Tab 2: 2D Decision Space Surface
    renderDecisionBoundary();

  } catch (err) {
    console.error('Inference error:', err);
  }
}

// Update Verdict Card UI
function updateVerdictUI(pred) {
  const hero = document.getElementById('verdictHero');
  const title = document.getElementById('verdictTitle');
  const conf = document.getElementById('verdictConfidence');
  const margin = document.getElementById('verdictMargin');
  const zoneBadge = document.getElementById('marginZoneBadge');
  const probBTT = document.getElementById('probValBTT');
  const probIDA = document.getElementById('probValIDA');
  const barBTT = document.getElementById('probBarBTT');
  const barIDA = document.getElementById('probBarIDA');
  const discordanceBanner = document.getElementById('discordanceBanner');
  const discordanceText = document.getElementById('discordanceText');

  const isBTT = pred.predicted_class === 1;

  hero.className = `verdict-hero ${isBTT ? 'btt-verdict' : 'ida-verdict'}`;
  title.textContent = pred.diagnosis;
  conf.textContent = `${pred.confidence}%`;
  conf.style.color = isBTT ? '#c4b5fd' : '#67e8f9';

  const sign = pred.decision_margin >= 0 ? '+' : '';
  margin.textContent = `${sign}${pred.decision_margin}`;

  zoneBadge.textContent = pred.margin_zone;
  zoneBadge.style.background = isBTT ? 'rgba(139, 92, 246, 0.2)' : 'rgba(6, 182, 212, 0.2)';
  zoneBadge.style.color = isBTT ? '#c4b5fd' : '#67e8f9';

  probBTT.textContent = `${pred.probabilities.btt}%`;
  probIDA.textContent = `${pred.probabilities.ida}%`;
  barBTT.style.width = `${pred.probabilities.btt}%`;
  barIDA.style.width = `${pred.probabilities.ida}%`;

  // Mentzer Concordance Banner
  const m = pred.mentzer_comparison;
  if (m.status === 'CONCORDANT') {
    discordanceBanner.className = 'discordance-banner concordant';
    discordanceBanner.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
      <div><strong>Clinical Concordance:</strong> ${m.explanation}</div>
    `;
  } else {
    discordanceBanner.className = 'discordance-banner discordant';
    discordanceBanner.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <div><strong>Clinical Discordance Warning:</strong> ${m.explanation}</div>
    `;
  }
}

// Render SHAP Waterfall Force Bars
function renderShapWaterfall(shapData, narratives) {
  const container = document.getElementById('waterfallContainer');
  const baselineElem = document.getElementById('shapBaseline');
  const narrativeList = document.getElementById('clinicalNarrativeList');

  baselineElem.textContent = `${shapData.baseline_prob}%`;
  container.innerHTML = '';

  shapData.waterfall.forEach(item => {
    const isBtt = item.direction === 'btt';
    const absPct = Math.abs(item.attribution_pct);
    const barWidth = Math.min(absPct * 2.2, 48); // scale bar for visual clarity

    const row = document.createElement('div');
    row.className = 'waterfall-bar-item';
    row.innerHTML = `
      <div class="waterfall-feat-name">${item.label}</div>
      <div class="waterfall-feat-val">${item.patient_value} ${item.unit}</div>
      <div class="waterfall-visual-track">
        <div class="waterfall-center-line"></div>
        <div class="waterfall-bar ${isBtt ? 'push-btt' : 'push-ida'}" 
             style="width: ${barWidth}%;"></div>
      </div>
      <div class="waterfall-score ${isBtt ? 'push-btt' : 'push-ida'}">
        ${isBtt ? '+' : ''}${item.attribution_pct}%
      </div>
    `;
    container.appendChild(row);
  });

  // Clinical Narratives
  narrativeList.innerHTML = '';
  narratives.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    narrativeList.appendChild(li);
  });
}

// Render Nearest Support Vector Exemplar Cards
function renderExemplars(exemplars) {
  const grid = document.getElementById('exemplarGrid');
  grid.innerHTML = '';

  exemplars.forEach((sv, idx) => {
    const isBtt = sv.class_code === 1;
    const card = document.createElement('div');
    card.className = `exemplar-card ${isBtt ? 'btt' : 'ida'}`;

    card.innerHTML = `
      <div class="exemplar-header">
        <div>
          <span class="exemplar-tag">${sv.diagnosis}</span>
          <div style="font-size: 0.7rem; color: var(--text-muted);">${sv.sv_id} (Rank #${idx + 1} Neighbor)</div>
        </div>
        <div class="exemplar-sim">
          ${sv.similarity_pct}% Similarity
        </div>
      </div>
      <div class="exemplar-metrics">
        <span>RBC: <strong>${sv.features['RBC Count']}</strong> 10¹²/L</span>
        <span>Hb: <strong>${sv.features['Hemoglobin']}</strong> g/dL</span>
        <span>MCV: <strong>${sv.features['MCV']}</strong> fL</span>
        <span>RDW-CV: <strong>${sv.features['RDW-CV']}</strong>%</span>
        <span>Mentzer: <strong>${sv.features['Mentzer Index']}</strong></span>
        <span>Distance: <strong>${sv.distance}</strong></span>
      </div>
    `;
    grid.appendChild(card);
  });
}

// Render What-If Sensitivity Charts (Chart.js)
function renderSensitivity(sensitivity, currentDiagnosis) {
  const rbcCtx = document.getElementById('chartRbcSensitivity').getContext('2d');
  const rdwCtx = document.getElementById('chartRdwSensitivity').getContext('2d');
  const cfText = document.getElementById('counterfactualText');

  // RBC Chart
  const rbcLabels = sensitivity.rbc_sweep.map(d => `${d.rbc} (${d.delta > 0 ? '+' : ''}${d.delta})`);
  const rbcProbs = sensitivity.rbc_sweep.map(d => d.new_btt_prob);

  if (state.rbcChart) state.rbcChart.destroy();
  state.rbcChart = new Chart(rbcCtx, {
    type: 'line',
    data: {
      labels: rbcLabels,
      datasets: [{
        label: 'BTT Probability (%)',
        data: rbcProbs,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.15)',
        tension: 0.3,
        fill: true,
        pointBackgroundColor: '#8b5cf6',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });

  // RDW Chart
  const rdwLabels = sensitivity.rdw_sweep.map(d => `${d.rdw} (${d.delta > 0 ? '+' : ''}${d.delta}%)`);
  const rdwProbs = sensitivity.rdw_sweep.map(d => d.new_btt_prob);

  if (state.rdwChart) state.rdwChart.destroy();
  state.rdwChart = new Chart(rdwCtx, {
    type: 'line',
    data: {
      labels: rdwLabels,
      datasets: [{
        label: 'BTT Probability (%)',
        data: rdwProbs,
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.15)',
        tension: 0.3,
        fill: true,
        pointBackgroundColor: '#06b6d4',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });

  // Narrative
  if (currentDiagnosis.includes('Beta Thalassemia')) {
    cfText.innerHTML = `
      Current status is <strong>Beta Thalassemia Trait</strong>. Based on sensitivity gradient analysis, to invert the classification toward <strong>Iron Deficiency Anemia</strong>, the patient's RBC Count would need to decrease significantly below 4.5 10¹²/L, or the RDW-CV would need to exceed 22.5% (demonstrating severe anisocytosis).
    `;
  } else {
    cfText.innerHTML = `
      Current status is <strong>Iron Deficiency Anemia</strong>. To invert the classification toward <strong>Beta Thalassemia Trait</strong>, the patient's RBC Count would need to increase above 5.4 10¹²/L (microcytic erythrocytosis) or RDW-CV would need to decrease below 16.0% (homogeneous red cell sizing).
    `;
  }
}

// Setup Canvas and Render 2D Decision Space Visualizer
function setupCanvas() {
  function resizeCanvas() {
    canvas.width = canvasWrapper.clientWidth;
    canvas.height = canvasWrapper.clientHeight;
    if (state.boundaryData) drawBoundaryCanvas(state.boundaryData);
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  axisXSelect.addEventListener('change', () => renderDecisionBoundary());
  axisYSelect.addEventListener('change', () => renderDecisionBoundary());
  btnRecalculateBoundary.addEventListener('click', () => renderDecisionBoundary());

  // Interactive Hover Inspection on Canvas
  canvas.addEventListener('mousemove', (e) => {
    if (!state.boundaryData) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const b = state.boundaryData;
    const padding = 50;
    const plotW = canvas.width - padding * 2;
    const plotH = canvas.height - padding * 2;

    if (mx < padding || mx > canvas.width - padding || my < padding || my > canvas.height - padding) {
      canvasTooltip.style.display = 'none';
      return;
    }

    const normX = (mx - padding) / plotW;
    const normY = 1.0 - ((my - padding) / plotH);

    const valX = (b.x_range[0] + normX * (b.x_range[1] - b.x_range[0])).toFixed(2);
    const valY = (b.y_range[0] + normY * (b.y_range[1] - b.y_range[0])).toFixed(2);

    // Check if near any support vector
    let nearbyText = '';
    b.support_vectors.forEach(sv => {
      const svPx = padding + ((sv.x - b.x_range[0]) / (b.x_range[1] - b.x_range[0])) * plotW;
      const svPy = padding + (1.0 - (sv.y - b.y_range[0]) / (b.y_range[1] - b.y_range[0])) * plotH;
      const dist = Math.hypot(mx - svPx, my - svPy);
      if (dist < 8) {
        nearbyText = `<br><span style="color: #c4b5fd;">★ Support Vector: ${sv.id} (${sv.class_label})</span>`;
      }
    });

    canvasTooltip.style.display = 'block';
    canvasTooltip.style.left = `${mx + 15}px`;
    canvasTooltip.style.top = `${my - 15}px`;
    canvasTooltip.innerHTML = `
      <strong>${b.x_feature}:</strong> ${valX}<br>
      <strong>${b.y_feature}:</strong> ${valY}
      ${nearbyText}
    `;
  });

  canvas.addEventListener('mouseleave', () => {
    canvasTooltip.style.display = 'none';
  });
}

// Fetch 2D Decision Boundary Mesh
async function renderDecisionBoundary() {
  try {
    const xFeat = axisXSelect.value;
    const yFeat = axisYSelect.value;

    const url = `/api/decision-boundary?x_feature=${encodeURIComponent(xFeat)}&y_feature=${encodeURIComponent(yFeat)}&age=${state.patientData['Age_years']}&hb=${state.patientData['Hemoglobin']}&mch=${state.patientData['MCH']}&mchc=${state.patientData['MCHC']}&rdw=${state.patientData['RDW-CV']}&grid_size=28`;

    const res = await fetch(url);
    if (!res.ok) throw new Error('Boundary grid API error');
    const bData = await res.json();
    state.boundaryData = bData;

    drawBoundaryCanvas(bData);
  } catch (err) {
    console.error('Error rendering boundary:', err);
  }
}

// Draw Canvas with smooth RBF decision contours and points
function drawBoundaryCanvas(data) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const padding = 50;

  ctx.clearRect(0, 0, w, h);

  const plotW = w - padding * 2;
  const plotH = h - padding * 2;

  const xMin = data.x_range[0];
  const xMax = data.x_range[1];
  const yMin = data.y_range[0];
  const yMax = data.y_range[1];

  const toPxX = (val) => padding + ((val - xMin) / (xMax - xMin)) * plotW;
  const toPxY = (val) => padding + (1.0 - (val - yMin) / (yMax - yMin)) * plotH;

  // 1. Render Contour Grid Cells
  const gridN = data.grid_size;
  const cellW = plotW / (gridN - 1);
  const cellH = plotH / (gridN - 1);

  for (let i = 0; i < gridN - 1; i++) {
    for (let j = 0; j < gridN - 1; j++) {
      const prob = data.probs_btt[j][i]; // [row][col]

      // Interpolate colors: BTT purple (#8b5cf6) vs IDA cyan (#06b6d4)
      const r = Math.round(6 + prob * (139 - 6));
      const g = Math.round(182 + prob * (92 - 182));
      const bColor = Math.round(212 + prob * (246 - 212));
      const alpha = 0.35 + Math.abs(prob - 0.5) * 0.45;

      ctx.fillStyle = `rgba(${r}, ${g}, ${bColor}, ${alpha})`;

      const px = toPxX(data.x_vals[i]);
      const py = toPxY(data.y_vals[j + 1]);

      ctx.fillRect(px, py, cellW + 1, cellH + 1);
    }
  }

  // 2. Draw Decision Boundary Line (prob ~ 0.50)
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);

  for (let i = 0; i < gridN - 1; i++) {
    for (let j = 0; j < gridN - 1; j++) {
      const p1 = data.probs_btt[j][i];
      const p2 = data.probs_btt[j][i + 1];
      const p3 = data.probs_btt[j + 1][i];

      if ((p1 - 0.5) * (p2 - 0.5) < 0) {
        const xMid = (data.x_vals[i] + data.x_vals[i + 1]) / 2;
        const yPos = data.y_vals[j];
        ctx.beginPath();
        ctx.arc(toPxX(xMid), toPxY(yPos), 1.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    }
  }
  ctx.setLineDash([]);

  // 3. Draw Grid Lines and Axes
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padding, padding, plotW, plotH);

  // Axis Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`${data.x_feature}`, padding + plotW / 2, h - 15);

  ctx.save();
  ctx.translate(15, padding + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`${data.y_feature}`, 0, 0);
  ctx.restore();

  // Range text
  ctx.textAlign = 'left';
  ctx.fillText(xMin.toFixed(1), padding, h - 35);
  ctx.textAlign = 'right';
  ctx.fillText(xMax.toFixed(1), padding + plotW, h - 35);

  ctx.textAlign = 'left';
  ctx.fillText(yMax.toFixed(1), padding + 5, padding + 15);
  ctx.fillText(yMin.toFixed(1), padding + 5, padding + plotH - 5);

  // 4. Render Historical Cohort Points
  data.cohort_points.forEach(pt => {
    const px = toPxX(pt.x);
    const py = toPxY(pt.y);
    const isBtt = pt.true_diagnosis === 'beta_thalassemia_heterozygous';

    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = isBtt ? 'rgba(196, 181, 253, 0.75)' : 'rgba(103, 232, 249, 0.75)';
    ctx.fill();
  });

  // 5. Render Support Vectors (Halo Rings)
  data.support_vectors.forEach(sv => {
    const px = toPxX(sv.x);
    const py = toPxY(sv.y);
    const isBtt = sv.class_code === 1;

    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = isBtt ? '#8b5cf6' : '#06b6d4';
    ctx.fill();
  });

  // 6. Render Active Patient Pin (Red Pulsing Pin)
  const patientX = state.patientData[data.x_feature];
  const patientY = state.patientData[data.y_feature];

  if (patientX !== undefined && patientY !== undefined) {
    const px = toPxX(patientX);
    const py = toPxY(patientY);

    // Glowing Outer Halo
    ctx.beginPath();
    ctx.arc(px, py, 14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(244, 63, 94, 0.25)';
    ctx.fill();

    // Intermediate Ring
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Solid Center
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    // Label
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Patient Target', px + 12, py + 4);
  }
}

// Load and Render HPLC Cohort Table
async function loadCohort(filter) {
  try {
    const res = await fetch(`/api/cohort?filter_type=${filter}`);
    const data = await res.json();
    state.cohortData = data.records;

    const tbody = document.getElementById('cohortTableBody');
    tbody.innerHTML = '';

    data.records.forEach(row => {
      const tr = document.createElement('tr');
      const isBtt = row.hplc_diagnosis === 'beta_thalassemia_heterozygous';
      const isIda = row.hplc_diagnosis === 'ida';
      const badgeClass = isBtt ? 'color: #c4b5fd;' : (isIda ? 'color: #67e8f9;' : 'color: #94a3b8;');

      tr.innerHTML = `
        <td><strong>${row.id}</strong></td>
        <td>${row.age}</td>
        <td>${row.rbc}</td>
        <td>${row.hb}</td>
        <td>${row.mcv}</td>
        <td>${row.mch}</td>
        <td>${row.rdw}</td>
        <td>${row.mentzer}</td>
        <td style="${badgeClass} font-weight: 600;">${row.hplc_diagnosis}</td>
        <td>${row.svm_prob_btt !== null ? `${row.svm_prob_btt}% BTT` : '-'}</td>
        <td>
          <button class="cohort-btn-load" data-id="${row.id}">Load into Analyzer</button>
        </td>
      `;

      tr.querySelector('.cohort-btn-load').addEventListener('click', () => {
        loadPatientFromRow(row);
      });

      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Error loading cohort:', err);
  }
}

// Load patient into form
function loadPatientFromRow(row) {
  state.patientData['Age_years'] = row.age_years;
  state.patientData['RBC Count'] = row.rbc;
  state.patientData['HCT'] = row.hct;
  state.patientData['Hemoglobin'] = row.hb;
  state.patientData['MCV'] = row.mcv;
  state.patientData['MCH'] = row.mch;
  state.patientData['MCHC'] = row.mchc;
  state.patientData['RDW-CV'] = row.rdw;

  // Update inputs
  for (const [key, val] of Object.entries(state.patientData)) {
    if (key === 'Mentzer Index') continue;
    const cleanId = key.replace(/[^a-zA-Z0-9]/g, '_');
    const slider = document.getElementById(`slider_${cleanId}`);
    const input = document.getElementById(`input_${cleanId}`);
    if (slider) slider.value = val;
    if (input) input.value = val;
  }

  updateCalculatedMentzer();
  runInference();

  // Scroll smoothly to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Load Clinical Presets
async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    const presets = await res.json();

    presetSelect.addEventListener('change', (e) => {
      const presetId = e.target.value;
      if (!presetId || !presets[presetId]) return;

      const pVals = presets[presetId].values;
      for (const [key, val] of Object.entries(pVals)) {
        state.patientData[key] = val;
        const cleanId = key.replace(/[^a-zA-Z0-9]/g, '_');
        const slider = document.getElementById(`slider_${cleanId}`);
        const input = document.getElementById(`input_${cleanId}`);
        if (slider) slider.value = val;
        if (input) input.value = val;
      }
      updateCalculatedMentzer();
      runInference();
    });

    btnRandomCohort.addEventListener('click', () => {
      if (!state.cohortData.length) return;
      const randIdx = Math.floor(Math.random() * state.cohortData.length);
      loadPatientFromRow(state.cohortData[randIdx]);
    });

  } catch (err) {
    console.error('Presets error:', err);
  }
}

// Tab navigation handler
function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const targetId = tab.getAttribute('data-tab');
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.getElementById(targetId).classList.add('active');

      if (targetId === 'tab-boundary') {
        setTimeout(() => {
          canvas.width = canvasWrapper.clientWidth;
          canvas.height = canvasWrapper.clientHeight;
          if (state.boundaryData) drawBoundaryCanvas(state.boundaryData);
        }, 50);
      }
    });
  });

  // Reset button
  btnResetInputs.addEventListener('click', () => {
    if (state.modelInfo && state.modelInfo.features) {
      buildInputControls(state.modelInfo.features);
      runInference();
    }
  });

  // Run Inference button
  btnRunInference.addEventListener('click', () => {
    runInference();
  });

  // Filter cohort
  document.getElementById('filterCohort').addEventListener('change', (e) => {
    loadCohort(e.target.value);
  });

  // Modal Report handlers
  btnExportReport.addEventListener('click', () => {
    openReportModal();
  });
  btnCloseModal.addEventListener('click', () => {
    reportModal.classList.remove('active');
  });
  window.addEventListener('click', (e) => {
    if (e.target === reportModal) reportModal.classList.remove('active');
  });
}

// Generate Printable Clinical Report Modal
function openReportModal() {
  if (!state.latestPrediction) return;

  const pred = state.latestPrediction;
  document.getElementById('reportDate').textContent = new Date().toLocaleString();
  document.getElementById('reportDiagnosis').textContent = pred.diagnosis;
  document.getElementById('reportConfidence').textContent = `Model Confidence: ${pred.confidence}% | Decision Margin: ${pred.decision_margin >= 0 ? '+' : ''}${pred.decision_margin}`;

  const tbody = document.getElementById('reportTableBody');
  tbody.innerHTML = '';

  pred.shap_analysis.waterfall.forEach(item => {
    const tr = document.createElement('tr');
    const isBtt = item.direction === 'btt';
    const color = isBtt ? '#c084fc' : '#38bdf8';
    tr.innerHTML = `
      <td>${item.label}</td>
      <td><strong>${item.patient_value} ${item.unit}</strong></td>
      <td>${state.modelInfo.features[item.feature]?.normal_range.join(' - ') || '-'}</td>
      <td style="color: ${color}; font-weight: 700;">${isBtt ? '+' : ''}${item.attribution_pct}%</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('reportMentzerText').textContent = pred.mentzer_comparison.explanation;
  reportModal.classList.add('active');
}

// Start application
document.addEventListener('DOMContentLoaded', initApp);
