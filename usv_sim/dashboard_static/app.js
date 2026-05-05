const state = {
  samples: [],
  counts: { easy: 0, medium: 0, hard: 0 },
  filter: "all",
  selectedId: null,
  selected: null,
  playing: false,
  step: 0,
  playhead: 0,
  lastFrame: 0,
  speed: 1,
  balancedJobId: null,
  balancedPollTimer: null,
};

const els = {
  canvas: document.getElementById("trajectoryCanvas"),
  emptyState: document.getElementById("emptyState"),
  sampleList: document.getElementById("sampleList"),
  totalCount: document.getElementById("totalCount"),
  filteredCount: document.getElementById("filteredCount"),
  counts: document.getElementById("difficultyCounts"),
  filter: document.getElementById("difficultyFilter"),
  selectedDifficulty: document.getElementById("selectedDifficulty"),
  selectedTitle: document.getElementById("selectedTitle"),
  metrics: document.getElementById("metrics"),
  playButton: document.getElementById("playButton"),
  resetButton: document.getElementById("resetButton"),
  timeSlider: document.getElementById("timeSlider"),
  timeText: document.getElementById("timeText"),
  stepText: document.getElementById("stepText"),
  speedSelect: document.getElementById("speedSelect"),
  refreshButton: document.getElementById("refreshButton"),
  generateForm: document.getElementById("generateForm"),
  generateButton: document.getElementById("generateButton"),
  generateDifficulty: document.getElementById("generateDifficulty"),
  generateCount: document.getElementById("generateCount"),
  generateSeed: document.getElementById("generateSeed"),
  generationState: document.getElementById("generationState"),
  balancedGenerateForm: document.getElementById("balancedGenerateForm"),
  balancedGenerateButton: document.getElementById("balancedGenerateButton"),
  balancedGenerateCount: document.getElementById("balancedGenerateCount"),
  balancedGenerateSeed: document.getElementById("balancedGenerateSeed"),
  balancedGenerationState: document.getElementById("balancedGenerationState"),
  balancedProgressFill: document.getElementById("balancedProgressFill"),
  balancedProgressText: document.getElementById("balancedProgressText"),
  showAllTrajectories: document.getElementById("showAllTrajectories"),
};

const ctx = els.canvas.getContext("2d");
const difficultyLabel = { easy: "简单", medium: "中等", hard: "困难" };
const difficultyColor = { easy: "#2f8f61", medium: "#c77a29", hard: "#c84e4e" };

function filteredSamples() {
  if (state.filter === "all") return state.samples;
  return state.samples.filter((sample) => sample.difficulty === state.filter);
}

async function loadSamples(preferredId = state.selectedId) {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error("无法加载轨迹数据");
  const data = await response.json();
  state.samples = data.samples || [];
  state.counts = data.counts || { easy: 0, medium: 0, hard: 0 };

  renderSidebar();

  const available = filteredSamples();
  const next = available.find((sample) => sample.id === preferredId) || available[0] || null;
  if (next) {
    await selectSample(next.id);
  } else {
    state.selectedId = null;
    state.selected = null;
    state.step = 0;
    state.playhead = 0;
    state.playing = false;
    renderAll();
  }
}

async function selectSample(sampleId) {
  state.selectedId = sampleId;
  state.playing = false;
  state.step = 0;
  state.playhead = 0;
  setPlaybackIcon();
  renderSidebar();

  const response = await fetch(`/api/samples/${encodeURIComponent(sampleId)}`);
  if (!response.ok) throw new Error("无法加载轨迹详情");
  state.selected = await response.json();
  els.timeSlider.max = Math.max(0, state.selected.states.length - 1);
  els.timeSlider.value = "0";
  renderAll();
}

function renderSidebar() {
  const samples = filteredSamples();
  els.totalCount.textContent = `${state.samples.length} 条`;
  els.filteredCount.textContent = `${samples.length} 条`;
  els.counts.innerHTML = `
    <span>简单 ${state.counts.easy || 0}</span>
    <span>中等 ${state.counts.medium || 0}</span>
    <span>困难 ${state.counts.hard || 0}</span>
  `;

  els.sampleList.innerHTML = "";
  for (const sample of samples) {
    const button = document.createElement("button");
    button.className = `sample-card ${sample.id === state.selectedId ? "active" : ""}`;
    button.type = "button";
    button.addEventListener("click", () => selectSample(sample.id).catch(showError));
    button.innerHTML = `
      <div class="sample-top">
        <span class="sample-name">${escapeHtml(sample.name)} #${sample.index}</span>
        <span class="badge ${sample.difficulty}">${sample.difficulty_label}</span>
      </div>
      <div class="sample-meta">
        <span>障碍 ${sample.static_obstacle_count}+${sample.dynamic_obstacle_count}</span>
        <span>距离 ${formatNumber(sample.initial_distance, 1)} m</span>
        <span>余距 ${formatNumber(sample.min_clearance, 2)} m</span>
        <span>${sample.success ? "成功" : "未通过"}</span>
        <span>${formatNumber(sample.duration, 1)} s</span>
        <span>${formatNumber(sample.trajectory_length, 1)} m</span>
      </div>
    `;
    els.sampleList.appendChild(button);
  }

  els.emptyState.classList.toggle("hidden", samples.length > 0);
}

function renderAll() {
  resizeCanvas();
  renderHeader();
  drawScene();
  updatePlaybackReadout();
}

function renderHeader() {
  const selected = state.selected;
  const step = currentStep();
  if (!selected) {
    els.selectedDifficulty.textContent = "未选择";
    els.selectedTitle.textContent = state.samples.length ? "请选择一条轨迹" : "等待轨迹数据";
    els.metrics.innerHTML = "";
    return;
  }

  els.selectedDifficulty.textContent = `${selected.difficulty_label}障碍物`;
  els.selectedTitle.textContent = `${selected.name} #${selected.index}`;
  const speed = selected.states[step] ? Math.hypot(selected.states[step][4], selected.states[step][5]) : 0;
  const trace = selected.trace || [];
  const item = trace[Math.min(step, Math.max(0, trace.length - 1))] || {};
  const metrics = [
    ["成功状态", selected.success ? "成功" : "未通过"],
    ["当前位置速度", `${formatNumber(speed, 2)} m/s`],
    ["当前安全余距", `${formatNumber(item.clearance ?? selected.min_clearance, 2)} m`],
    ["距终点", `${formatNumber(item.dist_goal ?? selected.final_distance, 2)} m`],
  ];
  els.metrics.innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * ratio));
  const height = Math.max(320, Math.floor(rect.height * ratio));
  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawScene() {
  const rect = els.canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfdfd";
  ctx.fillRect(0, 0, width, height);

  const viewport = makeViewport(width, height);
  drawGrid(viewport);

  if (els.showAllTrajectories.checked) {
    const samples = filteredSamples();
    for (const sample of samples) {
      if (state.selected && sample.id === state.selected.id) continue;
      drawPolyline(sample.polyline, viewport, difficultyColor[sample.difficulty] || "#75838a", 0.18, 1.5);
    }
  }

  if (state.selected) {
    drawObstacles(state.selected, viewport);
    drawPlannerPath(state.selected, viewport);
    drawPolyline(state.selected.states.map((s) => [s[0], s[1]]), viewport, "#1976a2", 0.95, 3);
    drawTrace(state.selected, viewport);
    drawEndpoints(state.selected, viewport);
    drawVessel(state.selected, viewport);
    drawLegend(width, height);
  }
}

function makeViewport(width, height) {
  const pad = Math.max(34, Math.min(width, height) * 0.065);
  const side = Math.max(1, Math.min(width - 2 * pad, height - 2 * pad));
  const ox = (width - side) / 2;
  const oy = (height - side) / 2;
  return { ox, oy, side, minX: 0, maxX: 100, minY: 0, maxY: 100 };
}

function worldToScreen(point, viewport) {
  const x = viewport.ox + ((point[0] - viewport.minX) / (viewport.maxX - viewport.minX)) * viewport.side;
  const y = viewport.oy + viewport.side - ((point[1] - viewport.minY) / (viewport.maxY - viewport.minY)) * viewport.side;
  return [x, y];
}

function radiusToScreen(radius, viewport) {
  return (radius / (viewport.maxX - viewport.minX)) * viewport.side;
}

function drawGrid(viewport) {
  ctx.save();
  ctx.strokeStyle = "#dbe6e8";
  ctx.lineWidth = 1;
  for (let v = 0; v <= 100; v += 10) {
    const [x0, y0] = worldToScreen([v, 0], viewport);
    const [x1, y1] = worldToScreen([v, 100], viewport);
    const [x2, y2] = worldToScreen([0, v], viewport);
    const [x3, y3] = worldToScreen([100, v], viewport);
    line(x0, y0, x1, y1);
    line(x2, y2, x3, y3);
  }
  ctx.strokeStyle = "#284650";
  ctx.lineWidth = 2;
  ctx.strokeRect(viewport.ox, viewport.oy, viewport.side, viewport.side);
  ctx.restore();
}

function drawPolyline(points, viewport, color, alpha = 1, width = 2) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  points.forEach((point, index) => {
    const [x, y] = worldToScreen(point, viewport);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawPlannerPath(sample, viewport) {
  drawPolyline(sample.path, viewport, "#75838a", 0.7, 1.4);
  ctx.save();
  ctx.fillStyle = "#75838a";
  for (const point of sample.path || []) {
    const [x, y] = worldToScreen(point, viewport);
    circle(x, y, 3);
    ctx.fill();
  }
  ctx.restore();
}

function drawTrace(sample, viewport) {
  const points = sample.states.slice(0, currentStep() + 1).map((s) => [s[0], s[1]]);
  drawPolyline(points, viewport, "#0d5d83", 1, 4);
}

function drawEndpoints(sample, viewport) {
  const start = sample.scenario.start;
  const goal = sample.scenario.goal;
  const [sx, sy] = worldToScreen(start, viewport);
  const [gx, gy] = worldToScreen(goal, viewport);
  ctx.save();
  ctx.strokeStyle = "#0d5d83";
  ctx.lineWidth = 3;
  line(sx - 7, sy - 7, sx + 7, sy + 7);
  line(sx - 7, sy + 7, sx + 7, sy - 7);
  ctx.strokeStyle = "#2f8f61";
  circle(gx, gy, 10);
  ctx.stroke();
  line(gx - 14, gy, gx + 14, gy);
  line(gx, gy - 14, gx, gy + 14);
  ctx.restore();
}

function drawObstacles(sample, viewport) {
  const t = currentStep() * sample.dt;
  const obstacles = [
    ...(sample.scenario.static_obstacles || []).map((o) => ({ ...o, dynamic: false })),
    ...(sample.scenario.dynamic_obstacles || []).map((o) => ({ ...o, dynamic: true })),
  ];
  ctx.save();
  for (const obstacle of obstacles) {
    const center = obstaclePosition(obstacle, t);
    const [x, y] = worldToScreen(center, viewport);
    const r = radiusToScreen(obstacle.radius, viewport);
    const clearanceMargin = sample.obstacle_clearance_margin ?? sample.safety_margin ?? 0;
    const safety = radiusToScreen(obstacle.radius + clearanceMargin, viewport);
    ctx.fillStyle = obstacle.dynamic ? "#c77a29" : "#75838a";
    ctx.strokeStyle = obstacle.dynamic ? "#e2a565" : "#aeb8bd";
    ctx.globalAlpha = 0.95;
    circle(x, y, r);
    ctx.fill();
    ctx.globalAlpha = 0.75;
    circle(x, y, safety);
    ctx.stroke();

    if (obstacle.dynamic) {
      const future = obstaclePosition(obstacle, t + 6);
      const [fx, fy] = worldToScreen(future, viewport);
      ctx.strokeStyle = "#c77a29";
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 2;
      line(x, y, fx, fy);
      drawArrowHead(x, y, fx, fy);
    }
  }
  ctx.restore();
}

function obstaclePosition(obstacle, t) {
  const center = obstacle.center || [0, 0];
  const velocity = obstacle.velocity || [0, 0];
  if (obstacle.active === false) return center;
  return [center[0] + velocity[0] * t, center[1] + velocity[1] * t];
}

function drawVessel(sample, viewport) {
  const stateRow = sample.states[Math.min(currentStep(), sample.states.length - 1)];
  if (!stateRow) return;
  const x = stateRow[0];
  const y = stateRow[1];
  const heading = Math.atan2(stateRow[2], stateRow[3]);
  const length = radiusToScreen(5.2, viewport);
  const width = length * 0.54;
  const collisionRadius = radiusToScreen(sample.vessel_collision_radius ?? 3.3, viewport);
  const [sx, sy] = worldToScreen([x, y], viewport);

  ctx.save();
  ctx.strokeStyle = "rgba(13, 93, 131, 0.28)";
  ctx.lineWidth = 1.5;
  circle(sx, sy, collisionRadius);
  ctx.stroke();
  ctx.translate(sx, sy);
  ctx.rotate(-heading);
  ctx.fillStyle = "#176d8f";
  ctx.strokeStyle = "#082f3c";
  ctx.lineWidth = 2;
  const pontoonWidth = width * 0.22;
  const yOffset = width * 0.34;
  for (const side of [-1, 1]) {
    roundedVesselPart(-length * 0.48, side * yOffset - pontoonWidth / 2, length * 0.9, pontoonWidth);
  }
  ctx.fillStyle = "#d4e0e4";
  ctx.strokeStyle = "#082f3c";
  ctx.beginPath();
  ctx.rect(-length * 0.24, -width * 0.16, length * 0.48, width * 0.32);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f4f8f9";
  ctx.beginPath();
  ctx.rect(-length * 0.04, -width * 0.11, length * 0.2, width * 0.22);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#1f88af";
  ctx.beginPath();
  ctx.moveTo(length * 0.62, 0);
  ctx.lineTo(length * 0.34, -width * 0.18);
  ctx.lineTo(length * 0.34, width * 0.18);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function roundedVesselPart(x, y, width, height) {
  const radius = Math.min(5, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawLegend(width, height) {
  const items = [
    ["#1976a2", "完整轨迹"],
    ["#0d5d83", "已回放"],
    ["#75838a", "静态障碍"],
    ["#c77a29", "动态障碍"],
  ];
  ctx.save();
  ctx.font = "12px Segoe UI, Microsoft YaHei, Arial";
  ctx.textBaseline = "middle";
  let x = 20;
  const y = height - 22;
  for (const [color, label] of items) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 5, 18, 10);
    ctx.fillStyle = "#17313b";
    ctx.fillText(label, x + 24, y);
    x += ctx.measureText(label).width + 58;
  }
  ctx.restore();
}

function drawArrowHead(x0, y0, x1, y1) {
  const angle = Math.atan2(y1 - y0, x1 - x0);
  const size = 7;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - size * Math.cos(angle - Math.PI / 6), y1 - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x1 - size * Math.cos(angle + Math.PI / 6), y1 - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function line(x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

function circle(x, y, radius) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
}

function updatePlaybackReadout() {
  const selected = state.selected;
  const maxStep = selected ? selected.states.length - 1 : 0;
  state.step = Math.min(currentStep(), maxStep);
  const time = selected ? state.step * selected.dt : 0;
  els.timeSlider.max = maxStep;
  els.timeSlider.value = state.step;
  els.timeText.textContent = `${formatNumber(time, 1)} s`;
  els.stepText.textContent = `${state.step} / ${maxStep}`;
  els.playButton.disabled = !selected;
  els.resetButton.disabled = !selected;
  els.timeSlider.disabled = !selected;
}

function animationLoop(timestamp) {
  if (!state.lastFrame) state.lastFrame = timestamp;
  const elapsed = timestamp - state.lastFrame;
  state.lastFrame = timestamp;

  if (state.playing && state.selected) {
    const dt = Math.max(0.001, state.selected.dt || 0.5);
    const increment = (elapsed / 1000) * (1 / dt) * state.speed;
    state.playhead = Math.min(state.selected.states.length - 1, state.playhead + increment);
    state.step = currentStep();
    if (state.playhead >= state.selected.states.length - 1) {
      state.playhead = state.selected.states.length - 1;
      state.step = state.selected.states.length - 1;
      state.playing = false;
      setPlaybackIcon();
    }
    renderAll();
  }

  requestAnimationFrame(animationLoop);
}

function setPlaybackIcon() {
  els.playButton.textContent = state.playing ? "Ⅱ" : "▶";
}

function currentStep() {
  return Math.max(0, Math.floor(state.playhead));
}

function showError(error) {
  console.error(error);
  els.generationState.textContent = error.message || "操作失败";
}

function showBalancedError(error) {
  console.error(error);
  els.balancedGenerationState.textContent = error.message || "批量生成失败";
  els.balancedGenerateButton.disabled = false;
}

function updateBalancedProgress(job) {
  if (!job || job.status === "idle") {
    els.balancedGenerationState.textContent = "空闲";
    els.balancedProgressFill.style.width = "0%";
    els.balancedProgressText.textContent = "简单 0 / 中等 0 / 困难 0";
    els.balancedGenerateButton.disabled = false;
    return;
  }

  const created = Number(job.created || 0);
  const requested = Math.max(1, Number(job.requested || 1));
  const percent = Math.min(100, (created / requested) * 100);
  const targets = job.targets || {};
  const byDifficulty = job.created_by_difficulty || {};
  els.balancedProgressFill.style.width = `${percent.toFixed(1)}%`;
  const workers = Number(job.workers || 1);
  const savedCount = Number(job.saved_count || 0);
  els.balancedProgressText.textContent = [
    `简单 ${byDifficulty.easy || 0}/${targets.easy || 0}`,
    `中等 ${byDifficulty.medium || 0}/${targets.medium || 0}`,
    `困难 ${byDifficulty.hard || 0}/${targets.hard || 0}`,
    `并行 ${workers}`,
    `已保存 ${savedCount}`,
  ].join("  ");

  if (job.status === "running") {
    els.balancedGenerationState.textContent = `生成中 ${created}/${job.requested}`;
    els.balancedGenerateButton.disabled = true;
  } else if (job.status === "completed") {
    els.balancedGenerationState.textContent = `完成 ${created}/${job.requested}`;
    els.balancedGenerateButton.disabled = false;
  } else if (job.status === "failed") {
    els.balancedGenerationState.textContent = job.message || "生成失败";
    els.balancedGenerateButton.disabled = false;
  }
}

function stopBalancedPolling() {
  if (state.balancedPollTimer) {
    clearInterval(state.balancedPollTimer);
    state.balancedPollTimer = null;
  }
}

async function pollBalancedJob(jobId) {
  const response = await fetch(`/api/generate/balanced/${encodeURIComponent(jobId)}`);
  const job = await response.json();
  if (!response.ok) throw new Error(job.error || "无法读取批量生成状态");
  updateBalancedProgress(job);
  if (job.status === "completed") {
    stopBalancedPolling();
    await loadSamples(job.first_generated_id || state.selectedId);
  } else if (job.status === "failed") {
    stopBalancedPolling();
  }
}

function startBalancedPolling(jobId) {
  state.balancedJobId = jobId;
  stopBalancedPolling();
  state.balancedPollTimer = setInterval(() => {
    pollBalancedJob(jobId).catch(showBalancedError);
  }, 2000);
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0";
  return number.toFixed(digits);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

els.filter.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-difficulty]");
  if (!button) return;
  state.filter = button.dataset.difficulty;
  for (const item of els.filter.querySelectorAll("button")) item.classList.toggle("active", item === button);
  const next = filteredSamples()[0];
  if (next) {
    selectSample(next.id).catch(showError);
  } else {
    state.selectedId = null;
    state.selected = null;
    state.step = 0;
    state.playhead = 0;
    state.playing = false;
    setPlaybackIcon();
    renderSidebar();
    renderAll();
  }
});

els.playButton.addEventListener("click", () => {
  if (!state.selected) return;
  if (state.step >= state.selected.states.length - 1) {
    state.step = 0;
    state.playhead = 0;
  }
  state.playing = !state.playing;
  setPlaybackIcon();
});

els.resetButton.addEventListener("click", () => {
  state.step = 0;
  state.playhead = 0;
  state.playing = false;
  setPlaybackIcon();
  renderAll();
});

els.timeSlider.addEventListener("input", () => {
  state.step = Number(els.timeSlider.value);
  state.playhead = state.step;
  state.playing = false;
  setPlaybackIcon();
  renderAll();
});

els.speedSelect.addEventListener("change", () => {
  state.speed = Number(els.speedSelect.value);
});

els.showAllTrajectories.addEventListener("change", renderAll);

els.refreshButton.addEventListener("click", () => {
  els.generationState.textContent = "刷新中";
  loadSamples()
    .then(() => {
      els.generationState.textContent = "空闲";
    })
    .catch(showError);
});

els.generateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    difficulty: els.generateDifficulty.value,
    count: Number(els.generateCount.value),
    seed: els.generateSeed.value,
  };
  els.generateButton.disabled = true;
  els.generationState.textContent = "生成中";
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "生成失败");
    els.generationState.textContent = `已生成 ${data.created}/${data.requested}`;
    await loadSamples(data.generated_ids && data.generated_ids[0]);
  } catch (error) {
    showError(error);
  } finally {
    els.generateButton.disabled = false;
  }
});

els.balancedGenerateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    count: Number(els.balancedGenerateCount.value),
    seed: els.balancedGenerateSeed.value,
  };
  els.balancedGenerateButton.disabled = true;
  els.balancedGenerationState.textContent = "启动中";
  try {
    const response = await fetch("/api/generate/balanced", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "批量生成启动失败");
    updateBalancedProgress(job);
    startBalancedPolling(job.job_id);
  } catch (error) {
    showBalancedError(error);
  }
});

window.addEventListener("resize", renderAll);

loadSamples().catch(showError);
fetch("/api/generate/balanced/latest")
  .then((response) => response.json())
  .then((job) => {
    updateBalancedProgress(job);
    if (job.status === "running") startBalancedPolling(job.job_id);
  })
  .catch(showBalancedError);
requestAnimationFrame(animationLoop);
