// 前端显示与后端交互（主题/参数编辑/轨迹显示；控制与仿真在后端）

// 车辆参数（后端字段，仅用于显示与表单绑定）；控制与仿真均在后端
let params = {
  m: 35000.0, Iz: 500000.0, a: 8.0, b: 8.0,
  width: 3.4, track: 3.4,
  kf: 450000.0, kr: 450000.0,
  U: 2.2, mu: 0.85, g: 9.81, U_min: 0.3, U_blend: 0.3,
  tire_model: 'linear',
};
let state = { x: 0, y: 0, psi: 0, beta: 0, r: 0 };
let ctrl = { U: 2.2, df: 0, dr: 0, running: false };
let showObstacles = false;
// [新增] 地图与规划状态
let mapInfo = null;      // 存储后端传回的地图参数 (origin, resolution等)
let goalMarker = null;   // 用户点击的终点坐标 {x, y}
let globalPath = [];     // A* 全局路径 (绿色)
let localPath = [];      // Lattice 局部路径 (红色)

// 轨迹缓冲与设置
let track = [];
let plan = [];
const trackCfg = { enabled: true, retentionSec: 30, maxPoints: 20000 };
// 遥测缓冲与设置
let telemetry = [];
const telemetryCfg = { enabled: false, retentionSec: 30, timer: null };
let lastPerf = null;
// 到达终点检测与动画状态
let arrived = false;
let arriveTick = 0;
const arrivalAnim = { active: false, t0: 0, duration: 2000 };

// Canvas 与视图
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
let view = { scale: 20, offsetX: 0, offsetY: 0, follow: true, showGrid: true };

// 绘图（Chart.js）：最右侧独立栏，每个变量一个图
let charts = { epsi: null, ey: null, df: null, dr: null, beta: null, u: null };
let plotsCfg = { keepSec: 60, maxPoints: 600 }; // 10Hz 下约 60s
let plotsStartMs = performance.now();

function makeLineChart(elId, label, color) {
  const el = document.getElementById(elId);
  if (!el || typeof Chart === 'undefined') return null;
  // 固定像素尺寸，避免随容器宽度自动拉伸
  try { el.width = 320; el.height = 180; } catch (e) {}
  const textColor = getCSSVar('--text') || '#e7eaf6';
  return new Chart(el, {
    type: 'line',
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.2, pointRadius: 0 }] },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { mode: 'nearest', intersect: false } },
      scales: {
        x: { ticks: { color: textColor }, title: { display: true, text: 't (s)', color: textColor } },
        y: { ticks: { color: textColor }, title: { display: true, text: label, color: textColor } },
      },
    },
  });
}

function makeDualLineChart(elId, labels, colors) {
  const el = document.getElementById(elId);
  if (!el || typeof Chart === 'undefined') return null;
  try { el.width = 320; el.height = 180; } catch (e) {}
  const textColor = getCSSVar('--text') || '#e7eaf6';
  return new Chart(el, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: labels[0], data: [], borderColor: colors[0], tension: 0.2, pointRadius: 0 },
        { label: labels[1], data: [], borderColor: colors[1], tension: 0.2, pointRadius: 0 },
      ],
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { mode: 'nearest', intersect: false } },
      scales: {
        x: { ticks: { color: textColor }, title: { display: true, text: 't (s)', color: textColor } },
        y: { ticks: { color: textColor }, title: { display: true, text: 'δ (deg)', color: textColor } },
      },
    },
  });
}

function setupPlots() {
  try {
    const accent = getCSSVar('--accent') || '#6aa0ff';
    const ok = getCSSVar('--ok') || '#4caf50';
    const muted = getCSSVar('--muted') || '#9aa3bf';
    const danger = getCSSVar('--danger') || '#e74c3c';
    charts.epsi = makeLineChart('chart_epsi', 'eψ (deg)', danger);
    charts.ey   = makeLineChart('chart_ey',   'ey (m)',   muted);
    charts.df   = makeDualLineChart('chart_df', ['δf (deg)','δr (deg)'], [accent, '#ffa500']);
    charts.beta = makeLineChart('chart_beta', 'β (deg)',  ok);
    charts.u    = makeLineChart('chart_u',    'U (m/s)',  accent);
    bindPlotControls();
  } catch (e) { /* ignore */ }
}

function updatePlots() {
  const t_ms = performance.now() - plotsStartMs;
  const t_s = Math.max(0, t_ms / 1000);
  const err = computePlanError();
  const epsi_deg = deg(err.e_psi || 0);
  const ey_m = (err.e_y || 0);
  const df_deg = deg((typeof state.df === 'number') ? state.df : (ctrl.df || 0));
  const dr_deg = deg((typeof state.dr === 'number') ? state.dr : (ctrl.dr || 0));
  const beta_deg = deg((typeof state.beta === 'number') ? state.beta : 0);
  const U_mps = (typeof state.U === 'number') ? state.U : (typeof ctrl.U === 'number' ? ctrl.U : (params.U || 0));
  const keepN = Math.max(10, Math.min(plotsCfg.maxPoints, Math.floor(plotsCfg.keepSec * 10)));
  function pushUpdate(key, chart, val) {
    if (!chart) return;
    if (plotPaused[key]) return;
    chart.data.labels.push(t_s.toFixed(1));
    chart.data.datasets[0].data.push(val);
    while (chart.data.labels.length > keepN) chart.data.labels.shift();
    while (chart.data.datasets[0].data.length > keepN) chart.data.datasets[0].data.shift();
    chart.update('none');
  }
  function pushUpdateDual(key, chart, v0, v1) {
    if (!chart) return;
    if (plotPaused[key]) return;
    chart.data.labels.push(t_s.toFixed(1));
    chart.data.datasets[0].data.push(v0);
    chart.data.datasets[1].data.push(v1);
    while (chart.data.labels.length > keepN) chart.data.labels.shift();
    while (chart.data.datasets[0].data.length > keepN) chart.data.datasets[0].data.shift();
    while (chart.data.datasets[1].data.length > keepN) chart.data.datasets[1].data.shift();
    chart.update('none');
  }
  pushUpdate('epsi', charts.epsi, epsi_deg);
  pushUpdate('ey',   charts.ey,   ey_m);
  pushUpdateDual('df', charts.df, df_deg, dr_deg);
  pushUpdate('beta', charts.beta, beta_deg);
  pushUpdate('u',    charts.u,    U_mps);
}

function computePlanError() {
  try {
    if (!Array.isArray(plan) || plan.length < 2) return { e_y: 0, e_psi: 0, ok: false };
    let best_i = 0, best_d2 = Infinity;
    for (let i = 0; i < plan.length; i++) {
      const px = Number(plan[i].x) || 0;
      const py = Number(plan[i].y) || 0;
      const dx = px - (Number(state.x) || 0);
      const dy = py - (Number(state.y) || 0);
      const d2 = dx*dx + dy*dy;
      if (d2 < best_d2) { best_d2 = d2; best_i = i; }
    }
    const i_seg = Math.max(0, Math.min(plan.length - 2, best_i));
    const p0 = plan[i_seg]; const p1 = plan[i_seg + 1];
    const dx = Number(p1.x) - Number(p0.x);
    const dy = Number(p1.y) - Number(p0.y);
    const ds = Math.hypot(dx, dy);
    const psi_ref = ds > 1e-6 ? Math.atan2(dy, dx) : (Number(p0.psi) || Number(state.psi) || 0);
    const ex = (Number(state.x) || 0) - Number(p0.x);
    const ey = (Number(state.y) || 0) - Number(p0.y);
    const e_y = -ex * Math.sin(psi_ref) + ey * Math.cos(psi_ref);
    const e_psi = angleDiff(psi_ref, Number(state.psi) || 0);
    return { e_y, e_psi, ok: true };
  } catch (e) { return { e_y: 0, e_psi: 0, ok: false }; }
}

// CSS 变量读取
function getCSSVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// 后端连通指示器
let backendConnected = false;
function setConnStatus(ok) {
  backendConnected = ok;
  const dot = document.getElementById('conn_dot');
  if (!dot) return;
  dot.classList.toggle('ok', ok);
  dot.classList.toggle('bad', !ok);
}
async function pingBackend() {
  try { const res = await fetch('/api/params', { cache: 'no-store' }); setConnStatus(res.ok); }
  catch { setConnStatus(false); }
}

// 设备指示器（MPPI 使用的设备：mps/cuda/cpu）；active 表示是否已初始化并在用
function setDeviceIndicator(dev, active) {
  const dot = document.getElementById('dev_dot');
  const label = document.getElementById('dev_label');
  if (!dot || !label) return;
  const d = (dev || '').toLowerCase();
  let cls = 'muted';
  let text = 'N/A';
  if (d === 'mps') { cls = 'ok'; text = 'MPS'; }
  else if (d === 'cuda') { cls = 'accent'; text = 'CUDA'; }
  else if (d === 'cpu') { cls = 'muted'; text = 'CPU'; }
  else if (d) { cls = 'bad'; text = d.toUpperCase(); }
  dot.classList.remove('ok','bad','accent','muted');
  dot.classList.add(cls);
  dot.classList.toggle('inactive', !active);
  label.classList.toggle('inactive', !active);
  label.textContent = text;
}

// 控制器类型指示器
function setControllerIndicator(type, enabled) {
  const dot = document.getElementById('ctl_dot');
  const label = document.getElementById('ctl_label');
  if (!dot || !label) return;
  const t = (type || '').toLowerCase();
  let cls = 'muted';
  let text = 'N/A';
  
  if (!enabled) { 
    cls = 'muted'; 
    text = '手动'; 
  } else if (t === 'mpc') { 
    cls = 'ok'; 
    text = 'MPC'; 
  } else { 
    cls = 'muted'; 
    text = (type || 'N/A'); 
  }
  dot.classList.remove('ok','bad','accent','muted');
  dot.classList.add(cls);
  label.textContent = text;
}
async function pollAutopDevice() {
  try {
    const res = await fetch('/api/autop', { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    setDeviceIndicator(data.device || null, !!data.device_active);
    setControllerIndicator(data.controller || data.mode || null, !!data.enabled);
  } catch {}
}

// 绑定 UI 控件
function bindUI() {
  // 主题与外观（右上角）
  const themeSel = document.getElementById('theme_select');
  const customThemeRow = document.getElementById('custom_theme');
  const accentPicker = document.getElementById('accent_color');
  const okPicker = document.getElementById('ok_color');
  // 左/右侧栏折叠/展开（图标）
  const gridEl = document.querySelector('.app-grid');
  const collapseLeftIcon = document.getElementById('collapse_left_icon');
  const collapsePlotsIcon = document.getElementById('collapse_plots_icon');
  try {
    const leftSaved = localStorage.getItem('leftCollapsed');
    const plotsSaved = localStorage.getItem('plotsCollapsed');
    if (leftSaved === '1') { gridEl && gridEl.classList.add('collapsed-left'); }
    if (plotsSaved === '1') { gridEl && gridEl.classList.add('collapsed-plots'); }
  } catch {}
  if (collapseLeftIcon) {
    collapseLeftIcon.addEventListener('click', () => {
      if (!gridEl) return;
      const collapsed = gridEl.classList.toggle('collapsed-left');
      try { localStorage.setItem('leftCollapsed', collapsed ? '1' : '0'); } catch {}
      draw();
    });
  }
  if (collapsePlotsIcon) {
    collapsePlotsIcon.addEventListener('click', () => {
      if (!gridEl) return;
      const collapsed = gridEl.classList.toggle('collapsed-plots');
      try { localStorage.setItem('plotsCollapsed', collapsed ? '1' : '0'); } catch {}
      draw();
    });
  }

  const prefersDarkMQ = window.matchMedia('(prefers-color-scheme: dark)');
  let systemListener = null;
  function applyTheme(v) {
    localStorage.setItem('theme', v);
    customThemeRow.style.display = v === 'custom' ? 'flex' : 'none';
    // tear down system listener if leaving system mode
    if (systemListener && v !== 'system') { try { prefersDarkMQ.removeEventListener('change', systemListener); } catch {} systemListener = null; }
    if (v === 'light') {
      document.documentElement.classList.add('light');
      document.documentElement.style.removeProperty('--accent');
      document.documentElement.style.removeProperty('--ok');
    } else if (v === 'dark') {
      document.documentElement.classList.remove('light');
      document.documentElement.style.removeProperty('--accent');
      document.documentElement.style.removeProperty('--ok');
    } else if (v === 'system') {
      const isDark = !!prefersDarkMQ.matches;
      document.documentElement.classList.toggle('light', !isDark);
      document.documentElement.style.removeProperty('--accent');
      document.documentElement.style.removeProperty('--ok');
      if (!systemListener) {
        systemListener = (e) => { document.documentElement.classList.toggle('light', !e.matches); };
        try { prefersDarkMQ.addEventListener('change', systemListener); } catch {}
      }
    } else if (v === 'custom') {
      // keep current light/dark class as-is; apply custom colors
      document.documentElement.style.setProperty('--accent', accentPicker.value);
      document.documentElement.style.setProperty('--ok', okPicker.value);
    }
  }
  const savedTheme = localStorage.getItem('theme') || 'light';
  const savedAccent = localStorage.getItem('accent') || getCSSVar('--accent');
  const savedOk = localStorage.getItem('ok') || getCSSVar('--ok');
  themeSel.value = savedTheme;
  accentPicker.value = savedAccent || '#6aa0ff';
  okPicker.value = savedOk || '#4caf50';
  applyTheme(savedTheme);
  themeSel.addEventListener('change', () => { applyTheme(themeSel.value); });

  // 顶部全局图表控件
  bindGlobalPlotControls();

  // 模式与场景切换
  const modeSel = document.getElementById('mode_select');
  async function loadMode() {
    try {
      const mRes = await fetch('/api/mode', { cache: 'no-store' });
      if (mRes.ok) { const m = await mRes.json(); if (m && m.mode) modeSel.value = m.mode; }
    } catch {}
  }
  loadMode();
  modeSel.addEventListener('change', async () => {
    try { await fetch('/api/mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: modeSel.value }) }); }
    catch {}
  });
  accentPicker.addEventListener('input', () => {
    document.documentElement.style.setProperty('--accent', accentPicker.value);
    localStorage.setItem('accent', accentPicker.value);
  });
  okPicker.addEventListener('input', () => {
    document.documentElement.style.setProperty('--ok', okPicker.value);
    localStorage.setItem('ok', okPicker.value);
  });

  // 控制按钮
  const startPauseBtn = document.getElementById('start_pause_btn');
  const resetBtn = document.getElementById('reset_btn');
  startPauseBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sim/start_pause', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ running: !ctrl.running }) });
      const data = await res.json();
      ctrl.running = !!data.running; startPauseBtn.textContent = ctrl.running ? '暂停' : '开始';
    } catch (e) { console.warn('POST /api/sim/start_pause 失败', e); }
  });
  resetBtn.addEventListener('click', async () => {
    try {
      // 1) 关闭自动跟踪并暂停仿真
      await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: false }) });
      await fetch('/api/sim/start_pause', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ running: false }) });
      // 2) 后端复位（状态、轨迹、运行标志）
      await fetch('/api/sim/reset', { method: 'POST' });
      // 3) 归零前后轮角，避免残留角度影响
      const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ df_deg: 0, dr_deg: 0 }) });
      ctrl = await res.json();
      // 4) 同步 UI 与本地状态
      ctrl.running = false; startPauseBtn.textContent = '开始';
      arrived = false; arriveTick = 0; arrivalAnim.active = false; arrivalAnim.t0 = 0;
      // 停止遥测采样
      try { stopTelemetry(); } catch {}
      // 清空并暂停所有图表记录，重置时间轴
      try {
        for (const k of Object.keys(charts)) clearChart(k);
        plotsStartMs = performance.now();
        globalPaused = true;
        for (const k of Object.keys(plotPaused)) {
          plotPaused[k] = true;
          const pbtn = document.getElementById(`pause_${k}`);
          if (pbtn) pbtn.textContent = '开始记录';
        }
        const gPauseBtn = document.getElementById('global_pause');
        if (gPauseBtn) gPauseBtn.textContent = '开始记录';
      } catch {}
      // 同步自动跟踪按钮文案
      try {
        const autopStartEl = document.getElementById('autop_start');
        if (autopStartEl) autopStartEl.textContent = '开始自动跟踪';
      } catch {}
      pollAutopDevice();
      draw();
    } catch (e) { console.warn('重置失败', e); }
  });

  // 起始位姿
  const applyInitPoseBtn = document.getElementById('apply_init_pose');
  const initX = document.getElementById('init_x');
  const initY = document.getElementById('init_y');
  const initPsi = document.getElementById('init_psi');
  applyInitPoseBtn.addEventListener('click', async () => {
    const x0 = parseFloat(initX.value) || 0;
    const y0 = parseFloat(initY.value) || 0;
    const psi0deg = parseFloat(initPsi.value) || 0;
    try { await fetch('/api/init_pose', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ x: x0, y: y0, psi: psi0deg }) }); } catch (e) {}
    view.offsetX = 0; view.offsetY = 0; if (view.follow) centerOnVehicle();
  });

  // 终点与规划
  const goalX = document.getElementById('goal_x');
  const goalY = document.getElementById('goal_y');
  const goalPsi = document.getElementById('goal_psi');
  const autopStartBtn = document.getElementById('autop_start');
  const planMethodSel = document.getElementById('plan_method_select');
  // 规划时长 T 已移除，由后端自动估算
  const autopModeSel = document.getElementById('autop_mode_select');
  const exportPlanBtn = document.getElementById('export_plan');
  const importPlanBtn = document.getElementById('import_plan');
  const importPlanFile = document.getElementById('import_plan_file');
  // 初始化自动模式选择器与后端同步
  try {
    fetch('/api/autop', { cache: 'no-store' }).then(res => res.ok ? res.json() : null).then(data => {
      if (data && data.mode && autopModeSel) autopModeSel.value = data.mode;
      if (autopStartBtn) autopStartBtn.textContent = (data && data.enabled) ? '停止自动跟踪' : '开始自动跟踪';
    }).catch(() => {});
  } catch {}
  // 初始化设备指示器并轮询更新
  pollAutopDevice();
  setInterval(pollAutopDevice, 5000);
  if (autopModeSel) {
    autopModeSel.addEventListener('change', async () => {
      try { await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: autopModeSel.value }) }); pollAutopDevice(); } catch {}
    });
  }
  // --- [新增] 按钮：加载测试场景 (一键复位) ---
  const btnLoadScene = document.getElementById('btn_load_scene');
  if (btnLoadScene) {
    btnLoadScene.addEventListener('click', async () => {
      if (!mapInfo) { alert('地图数据未就绪'); return; }
      
      // 1. 开启障碍物显示
      showObstacles = true;
      
      // 2. 设置默认终点 (读取 map_server 配置)
      if (mapInfo.default_end) {
        goalMarker = { x: mapInfo.default_end.x, y: mapInfo.default_end.y };
        
        // 同步更新 UI 输入框
        const gx = document.getElementById('goal_x');
        const gy = document.getElementById('goal_y');
        if(gx) gx.value = goalMarker.x.toFixed(1);
        if(gy) gy.value = goalMarker.y.toFixed(1);
      }
      
      // 3. 瞬移车辆到默认起点 (调用后端接口)
      if (mapInfo.default_start) {
        const s = mapInfo.default_start;
        try {
           await fetch('/api/init_pose', { 
             method: 'POST', 
             headers: { 'Content-Type': 'application/json' }, 
             body: JSON.stringify({ x: s.x, y: s.y, psi: 0 }) // 默认朝东
           });
           
           // 强制视角跟随并居中，体验更好
           view.follow = true;
           centerOnVehicle();
        } catch(e) { console.error('重置车辆位置失败', e); }
      }
      
      // 4. 重置规划状态 (清空旧路径，防止逻辑冲突)
      globalPath = [];
      localPath = [];
      plan = []; // 清空旧的 plan 以兼容图表
      
      // 禁用局部规划按钮，强制用户重新点击生成
      const btnLocal = document.getElementById('btn_local_plan');
      if(btnLocal) btnLocal.disabled = true;
      
      // 立即重绘 (显示障碍物、终点和新车位)
      draw();
      
      console.log('场景加载完成：障碍物已显示，起点/终点已重置');
    });
  }
  if (canvas) {
      canvas.addEventListener('mousedown', (e) => {
        // 简单判断：仅左键点击且没有按下 Shift 键(防止与拖拽冲突)触发
        if (e.button === 0 && !e.shiftKey) {
           const rect = canvas.getBoundingClientRect();
           
           // 计算相对于 Canvas 的像素坐标
           // 注意：这里需要处理 Canvas 可能的缩放样式
           const scaleX = canvas.width / rect.width;
           const scaleY = canvas.height / rect.height;
           const sx = (e.clientX - rect.left) * scaleX;
           const sy = (e.clientY - rect.top) * scaleY;
           
           // 转换为世界坐标 (调用之前定义的工具函数)
           const wPos = screenToWorld(sx, sy);
           
           // 更新全局状态
           goalMarker = { x: wPos.x, y: wPos.y };
           
           // 更新 UI 显示 (只读输入框)
           const gx = document.getElementById('goal_x');
           const gy = document.getElementById('goal_y');
           if(gx) gx.value = wPos.x.toFixed(1);
           if(gy) gy.value = wPos.y.toFixed(1);
           
           // 重置规划状态：一旦改变终点，之前的路径失效
           globalPath = [];
           localPath = [];
           plan = []; // 清空旧的 plan 以兼容图表
           
           // 禁用局部规划按钮，强制用户重新跑 A*
           const btnLocal = document.getElementById('btn_local_plan');
           if(btnLocal) btnLocal.disabled = true;
           
           // 立即重绘，显示红色的终点标记
           draw(); 
        }
      });
  }

  // --- [新增] 2. 按钮：全局规划 (A*) ---
  const btnGlobal = document.getElementById('btn_global_plan');
  if (btnGlobal) {
    btnGlobal.addEventListener('click', async () => {
      if (!goalMarker) { alert('请先在地图上左键点击设置终点'); return; }
      
      // 构造请求，start 不传默认使用当前车位
      const body = { end: goalMarker }; 
      
      try {
        // 给按钮加个 loading 状态反馈 (可选)
        btnGlobal.textContent = "规划中...";
        
        const res = await fetch('/api/plan/global', { 
            method: 'POST', 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        const data = await res.json();
        
        if (data.ok) {
           globalPath = data.path; // 存入全局变量
           console.log(`A* 成功: ${data.count} 点`);
           
           // 启用局部规划按钮
           const btnLocal = document.getElementById('btn_local_plan');
           if(btnLocal) btnLocal.disabled = false;
           
           draw();
        } else {
           alert('A* 失败: ' + (data.error || '未知错误'));
        }
      } catch(e) { 
          console.error(e); 
          alert('请求异常');
      } finally {
          btnGlobal.textContent = "1. 全局路径 (A*)";
      }
    });
  }

  // --- [新增] 3. 按钮：局部规划 (Lattice) ---
  const btnLocal = document.getElementById('btn_local_plan');
  if (btnLocal) {
    btnLocal.addEventListener('click', async () => {
      try {
        btnLocal.textContent = "计算中...";
        
        // 直接请求，无需参数，后端会使用之前 A* 算出的缓存路径
        const res = await fetch('/api/plan/local', { method: 'POST' });
        const data = await res.json();
        
        if (data.ok) {
           localPath = data.path; 
           
           // [关键] 同步给 plan 变量，让原来的图表逻辑也能工作
           plan = data.path; 
           
           console.log(`Lattice 成功: ${data.count} 点`);
           draw();
        } else {
           alert('Lattice 失败: ' + (data.error || '未知错误'));
        }
      } catch(e) { 
          console.error(e); 
      } finally {
          btnLocal.textContent = "2. 局部规划 (Lattice)";
      }
    });
  }
  
  // --- [新增] 4. 按钮：清除路径 ---
  const btnClear = document.getElementById('btn_clear_map');
  if (btnClear) {
    btnClear.addEventListener('click', () => {
       // 清空所有状态
       goalMarker = null;
       globalPath = [];
       localPath = [];
       plan = [];
       
       // 重置 UI
       document.getElementById('goal_x').value = '0';
       document.getElementById('goal_y').value = '0';
       const btnLocal = document.getElementById('btn_local_plan');
       if(btnLocal) btnLocal.disabled = true;
       
       draw();
    });
  }
  if (autopStartBtn) {
    autopStartBtn.addEventListener('click', async () => {
      try {
        // 读取当前自动状态以决定切换方向
        let enabled = false;
        try {
          const res = await fetch('/api/autop', { cache: 'no-store' });
          if (res.ok) {
            const info = await res.json(); enabled = !!info.enabled;
          }
        } catch {}
        if (enabled) {
          // 当前已启用：执行停止逻辑
          await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: false }) });
          await fetch('/api/sim/start_pause', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ running: false }) });
          const res2 = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ df_deg: 0, dr_deg: 0 }) });
          ctrl = await res2.json();
          startPauseBtn.textContent = '开始';
          autopStartBtn.textContent = '开始自动跟踪';
          arrived = false; arriveTick = 0; arrivalAnim.active = false; arrivalAnim.t0 = 0;
          pollAutopDevice(); draw();
        } else {
          // 当前未启用：执行启动逻辑
          const mode = (autopModeSel && autopModeSel.value) || 'mpc';
          await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: true, start: true, mode }) });
          ctrl.running = true; startPauseBtn.textContent = '暂停';
          autopStartBtn.textContent = '停止自动跟踪';
          // 清空所有图表并从零开始绘制，取消暂停
          try {
            for (const k of Object.keys(charts)) clearChart(k);
            plotsStartMs = performance.now();
            globalPaused = false;
            for (const k of Object.keys(plotPaused)) {
              plotPaused[k] = false;
              const pbtn = document.getElementById(`pause_${k}`);
              if (pbtn) pbtn.textContent = '暂停记录';
            }
            const gPauseBtn = document.getElementById('global_pause');
            if (gPauseBtn) gPauseBtn.textContent = '暂停记录';
          } catch {}
          view.follow = true; centerOnVehicle();
          pollAutopDevice();
        }
      } catch (e) { console.warn('自动跟踪切换失败', e); }
    });
  }
  if (exportPlanBtn) {
    exportPlanBtn.addEventListener('click', () => { try { exportPlanCSV(); } catch (e) {} });
  }
  if (importPlanBtn && importPlanFile) {
    importPlanBtn.addEventListener('click', () => { importPlanFile.click(); });
    importPlanFile.addEventListener('change', async () => {
      const file = importPlanFile.files && importPlanFile.files[0];
      if (!file) return;
      if (file.size > 10 * 1024 * 1024) { alert('文件超过 10MB 上限'); importPlanFile.value = ''; return; }
      try {
        const fd = new FormData(); fd.append('file', file);
        let res = await fetch('/api/plan/import_csv', { method: 'POST', body: fd });
        if (!res.ok && res.status === 405) {
          res = await fetch('/api/plan/import', { method: 'POST', body: fd });
        }
        if (!res.ok) { const err = await res.json().catch(() => ({ error: '导入失败' })); alert(err.error || '导入失败'); return; }
        await Promise.all([
          fetch('/api/plan?max=2000', { cache: 'no-store' }).then(r => r.ok ? r.json() : { points: [] }).then(d => { plan = Array.isArray(d.points) ? d.points : []; }),
        ]);
        view.follow = true; centerOnVehicle(); draw();
        alert('导入成功');
      } catch (e) {
        console.warn('上传并导入规划 CSV 失败', e);
        alert('导入失败，请稍后重试');
      } finally {
        importPlanFile.value = '';
      }
    });
  }

  // 跟随与网格
  const followChk = document.getElementById('follow_mode');
  const gridChk = document.getElementById('show_grid');
  followChk.addEventListener('change', () => { view.follow = followChk.checked; if (view.follow) centerOnVehicle(); });
  gridChk.addEventListener('change', () => { view.showGrid = gridChk.checked; draw(); });

  // 轨迹设置
  const showTrackChk = document.getElementById('show_track');
  const trackKeepInput = document.getElementById('track_keep_sec');
  if (showTrackChk) {
    trackCfg.enabled = showTrackChk.checked;
    showTrackChk.addEventListener('change', async () => {
      trackCfg.enabled = showTrackChk.checked;
      try { await fetch('/api/track/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: trackCfg.enabled }) }); } catch (e) {}
    });
  }
  if (trackKeepInput) {
    const v = parseFloat(trackKeepInput.value);
    if (!Number.isNaN(v)) trackCfg.retentionSec = Math.max(0, v);
    trackKeepInput.addEventListener('change', async () => {
      const n = parseFloat(trackKeepInput.value);
      trackCfg.retentionSec = Number.isNaN(n) ? trackCfg.retentionSec : Math.max(0, n);
      try { await fetch('/api/track/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ retentionSec: trackCfg.retentionSec }) }); } catch (e) {}
    });
  }

  // 轨迹导出为 CSV
  const exportBtn = document.getElementById('export_track');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => { try { exportTrackCSV(); } catch (e) {} });
  }

  // 遥测：保留时长设置与控制按钮
  const telKeepInput = document.getElementById('telemetry_keep_sec');
  const telStartBtn = document.getElementById('telemetry_start');
  const telStopBtn = document.getElementById('telemetry_stop');
  const telExportBtn = document.getElementById('export_telemetry');
  const telStats = document.getElementById('telemetry_stats');
  if (telKeepInput) {
    const v = parseFloat(telKeepInput.value);
    if (!Number.isNaN(v)) telemetryCfg.retentionSec = Math.max(0, v);
    telKeepInput.addEventListener('change', () => {
      const n = parseFloat(telKeepInput.value);
      telemetryCfg.retentionSec = Number.isNaN(n) ? telemetryCfg.retentionSec : Math.max(0, n);
    });
  }
  function updateTelStats() {
    if (telStats) telStats.textContent = `${telemetry.length} 条`;
  }
  async function pollTelemetryOnce() {
    try {
      const res = await fetch('/api/telemetry', { cache: 'no-store' });
      if (!res.ok) return;
      const d = await res.json();
      const now = performance.now();
      telemetry.push({
        t_ms: now,
        U: Number(d.U ?? 0),
        x: Number(d.x ?? 0),
        y: Number(d.y ?? 0),
        psi: Number(d.psi ?? 0),
        df: Number(d.df ?? 0),
        dr: Number(d.dr ?? 0),
        beta_dot: Number(d.beta_dot ?? 0),
        r_dot: Number(d.r_dot ?? 0),
        speed: Number(d.speed ?? (d.U ?? 0)),
        radius: (typeof d.radius === 'number') ? Number(d.radius) : null,
        autop_ms: Number(d.autop_ms ?? 0),
        loop_ms: Number(d.loop_ms ?? 0),
        sleep_ms: Number(d.sleep_ms ?? 0),
        tick_hz: Number(d.tick_hz ?? 0),
        dt_cfg: Number(d.dt_cfg ?? 0),
      });
      // 保留最近时长
      const keepMs = telemetryCfg.retentionSec * 1000;
      const cutoff = now - keepMs;
      let i = 0;
      while (i < telemetry.length && telemetry[i].t_ms < cutoff) i++;
      if (i > 0) telemetry.splice(0, i);
      updateTelStats();
    } catch (e) {}
  }
  function startTelemetry() {
    if (telemetryCfg.timer) return;
    telemetryCfg.enabled = true;
    telemetryCfg.timer = setInterval(pollTelemetryOnce, 100);
  }
  function stopTelemetry() {
    telemetryCfg.enabled = false;
    if (telemetryCfg.timer) { clearInterval(telemetryCfg.timer); telemetryCfg.timer = null; }
  }
  if (telStartBtn) telStartBtn.addEventListener('click', startTelemetry);
  if (telStopBtn) telStopBtn.addEventListener('click', stopTelemetry);
  if (telExportBtn) telExportBtn.addEventListener('click', () => { try { exportTelemetryCSV(); } catch (e) {} });

  // 已移除：自动跟踪调度（Stanley & PID），仅保留 MPC 选项

  // 手动设置：车速与前后轮角
  const speedInput = document.getElementById('speed_input');
  const dfInput = document.getElementById('df_input');
  const drInput = document.getElementById('dr_input');
  speedInput.addEventListener('change', async () => {
    const U = parseFloat(speedInput.value);
    try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ U }) }); ctrl = await res.json(); } catch (e) {}
  });
  dfInput.addEventListener('change', async () => {
    const df = parseFloat(dfInput.value);
    try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ df_deg: df }) }); ctrl = await res.json(); } catch (e) {}
  });
  drInput.addEventListener('change', async () => {
    const dr = parseFloat(drInput.value);
    try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dr_deg: dr }) }); ctrl = await res.json(); } catch (e) {}
  });

  // 键盘控制（保持与手动输入同步）
  const KEY_INC = { speed: 0.5, steer: 0.5 };
  window.addEventListener('keydown', async (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    let patch = null;
    switch(e.key.toLowerCase()) {
      case 'w': patch = { U: (ctrl.U + KEY_INC.speed) }; break;
      case 's': patch = { U: (ctrl.U - KEY_INC.speed) }; break;
      case 'a': patch = { df_deg: (deg(ctrl.df) + KEY_INC.steer) }; break;
      case 'd': patch = { df_deg: (deg(ctrl.df) - KEY_INC.steer) }; break;
      case 'q': patch = { dr_deg: (deg(ctrl.dr) - KEY_INC.steer) }; break;
      case 'e': patch = { dr_deg: (deg(ctrl.dr) + KEY_INC.steer) }; break;
      default: break;
    }
    if (patch) {
      try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) }); ctrl = await res.json(); } catch (err) {}
    }
  });

  // 车辆参数输入（回写后端）
  const pInputs = {
    m: document.getElementById('p_m'), Iz: document.getElementById('p_Iz'),
    a: document.getElementById('p_a'), b: document.getElementById('p_b'),
    width: document.getElementById('p_width'), track: document.getElementById('p_track'),
    kf: document.getElementById('p_kf'), kr: document.getElementById('p_kr'),
    mu: document.getElementById('p_mu'), g: document.getElementById('p_g'), U_min: document.getElementById('p_Umin'), U_blend: document.getElementById('p_Ublend'),
  };
  for (const key of Object.keys(pInputs)) {
    pInputs[key].addEventListener('change', () => {
      const val = parseFloat(pInputs[key].value);
      if (!Number.isNaN(val)) { params[key] = val; schedulePostParams(); updateDerivedStats(); }
    });
  }
  // 轮胎模型选择（字符串）
  const tireSel = document.getElementById('p_tire_model');
  if (tireSel) {
    tireSel.value = params.tire_model || 'pacejka';
    tireSel.addEventListener('change', () => {
      params.tire_model = tireSel.value;
      schedulePostParams();
    });
  }

  // 画布交互 + 自适应
  setupCanvasInteractions();
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();
  setupPlots();

  // 工具栏
  document.getElementById('zoom_in').addEventListener('click', () => { view.scale *= 1.2; draw(); });
  document.getElementById('zoom_out').addEventListener('click', () => { view.scale /= 1.2; draw(); });
  document.getElementById('reset_view').addEventListener('click', () => { view.scale = 20; view.offsetX = 0; view.offsetY = 0; centerOnVehicle(); draw(); });

  // 后端心跳
  pingBackend(); setInterval(pingBackend, 5000);

  // 显示循环：轮询后端状态与轨迹，仅负责显示
  async function pollOnce() {
    try {
      const [stRes, trRes, ctRes, plRes] = await Promise.all([
        fetch('/api/state', { cache: 'no-store' }),
        fetch('/api/track', { cache: 'no-store' }),
        fetch('/api/control', { cache: 'no-store' }),
        fetch('/api/plan?max=2000', { cache: 'no-store' }),
      ]);
      if (stRes.ok) { state = await stRes.json(); }
      if (trRes.ok) { const td = await trRes.json(); track = Array.isArray(td.points) ? td.points : []; }
      if (ctRes.ok) { ctrl = await ctRes.json(); }
      if (plRes.ok) {
        const pd = await plRes.json();
        plan = Array.isArray(pd.points) ? pd.points : [];
        // 若需要完整轨迹且当前只抽样，尝试分片一次性加载完整
        try {
          const metaRes = await fetch('/api/plan/meta', { cache: 'no-store' });
          if (metaRes.ok) {
            const meta = await metaRes.json(); const total = Number(meta.count || 0);
            if (total > plan.length && total > 0) {
              const chunkSize = 4000; let full = [];
              for (let s = 0; s < total; s += chunkSize) {
                const chRes = await fetch(`/api/plan/chunk?start=${s}&count=${chunkSize}`, { cache: 'no-store' });
                if (chRes.ok) { const ch = await chRes.json(); if (Array.isArray(ch.points)) full = full.concat(ch.points); }
              }
              if (full.length === total) plan = full;
            }
          }
        } catch {}
      }
      // 同步按钮文字与输入框显示（避免覆盖正在编辑的元素）
      startPauseBtn.textContent = ctrl.running ? '暂停' : '开始';
      const safeSet = (el, v) => { if (!el || document.activeElement === el) return; el.value = String(v); };
      safeSet(speedInput, (ctrl.U ?? params.U).toFixed(2));
      safeSet(dfInput, deg(ctrl.df ?? 0).toFixed(1));
      safeSet(drInput, deg(ctrl.dr ?? 0).toFixed(1));
      // 同步模式与场景显示
      if (ctrl.mode && modeSel && document.activeElement !== modeSel) modeSel.value = ctrl.mode;
      try {
        const perfRes = await fetch('/api/telemetry', { cache: 'no-store' });
        if (perfRes.ok) lastPerf = await perfRes.json();
      } catch {}
      updateDashboard();
      if (view.follow) centerOnVehicle();
      // 到达终点判定与处理（触发一次）
      const tgt = (Array.isArray(plan) && plan.length > 0) ? plan[plan.length - 1] : null;
      if (tgt) {
        const dx = state.x - (tgt.x || 0);
        const dy = state.y - (tgt.y || 0);
        const dist = Math.hypot(dx, dy);
        const dpsi = angleDiff(state.psi || 0, tgt.psi || 0);
        const distTh = 0.8; // m
        const angTh = 10 * Math.PI / 180; // rad
        if (dist <= distTh && Math.abs(dpsi) <= angTh) {
          arriveTick++;
        } else {
          arriveTick = 0;
        }
        if (!arrived && arriveTick >= 5) { // ~0.5s 稳定到达
          arrived = true;
          arrivalAnim.active = true; arrivalAnim.t0 = performance.now();
          // 停止自动跟踪，并将前后轮角设为 0 保持直行
          try { await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: false }) }); } catch (e) {}
          try { await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ df_deg: 0, dr_deg: 0 }) }); } catch (e) {}
          ctrl.df = 0; ctrl.dr = 0;
        }
      }
      draw();
      updatePlots();
    } catch (e) {}
  }
  setInterval(pollOnce, 100); // 10Hz 轮询
}

// 前端不再进行积分与复位，复位由后端处理
function resetSim() { /* no-op; 后端负责 */ }

// 交互：滚轮缩放、拖拽平移
function setupCanvasInteractions() {
  let dragging = false; let last = { x: 0, y: 0 };
  canvas.addEventListener('mousedown', (e) => { dragging = true; last.x = e.clientX; last.y = e.clientY; });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', (e) => {
    if (!dragging || view.follow) return;
    const dx = e.clientX - last.x; const dy = e.clientY - last.y;
    last.x = e.clientX; last.y = e.clientY;
    view.offsetX += dx; view.offsetY += dy; draw();
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault(); const factor = e.deltaY < 0 ? 1.1 : 0.9; view.scale *= factor; draw();
  }, { passive: false });
}

// 已迁移到后端：积分与控制

function centerOnVehicle() {
  const rect = canvas.getBoundingClientRect();
  const cx = rect.width / 2; const cy = rect.height / 2;
  const sx = worldToScreenX(state.x); const sy = worldToScreenY(state.y);
  view.offsetX += (cx - sx); view.offsetY += (cy - sy);
}

// 仪表盘与派生量更新
  function updateDashboard() {
  document.getElementById('stat_x').textContent = state.x.toFixed(2);
  document.getElementById('stat_y').textContent = state.y.toFixed(2);
  document.getElementById('stat_psi').textContent = (state.psi * 180 / Math.PI).toFixed(2);
  document.getElementById('stat_u').textContent = (ctrl.U ?? params.U).toFixed(2);
  document.getElementById('stat_df').textContent = deg(ctrl.df ?? 0).toFixed(1);
  document.getElementById('stat_dr').textContent = deg(ctrl.dr ?? 0).toFixed(1);
  const v = (typeof state.speed === 'number') ? state.speed : (ctrl.U ?? params.U);
  document.getElementById('stat_v').textContent = v.toFixed(2);
  const R = (typeof state.radius === 'number') ? state.radius : null;
  document.getElementById('stat_R').textContent = (R !== null) ? R.toFixed(2) : '—';
  const dfDot = (typeof state.df_dot === 'number') ? state.df_dot : 0;
  const drDot = (typeof state.dr_dot === 'number') ? state.dr_dot : 0;
  document.getElementById('stat_df_dot').textContent = deg(dfDot).toFixed(1);
  document.getElementById('stat_dr_dot').textContent = deg(drDot).toFixed(1);
  try {
    const p = lastPerf || (telemetry.length > 0 ? telemetry[telemetry.length - 1] : null);
    if (p) {
      const autopEl = document.getElementById('perf_autop_ms');
      const tickEl = document.getElementById('perf_tick_hz');
      const stepEl = document.getElementById('stat_step_ms');
      const hzEl = document.getElementById('stat_tick_hz');
      if (autopEl) autopEl.textContent = `${(p.autop_ms || 0).toFixed(1)} ms`;
      if (tickEl) tickEl.textContent = `${(p.tick_hz || 0).toFixed(1)} Hz`;
      if (stepEl) stepEl.textContent = `${(p.loop_ms || 0).toFixed(2)}`;
      if (hzEl) hzEl.textContent = `${(p.tick_hz || 0).toFixed(2)}`;
    }
  } catch {}
}
function updateDerivedStats() {
  // L 与 K 在后端计算亦可，这里简易前端计算与展示
  const L = params.a + params.b;
  const K = (params.m / L) * (params.b / params.kr - params.a / params.kf);
  const pL = document.getElementById('p_L'); const pK = document.getElementById('p_K');
  if (pL) pL.textContent = L.toFixed(3);
  if (pK) pK.textContent = K.toFixed(6);
}

// 绘制
function worldToScreenX(x) { return x * view.scale + view.offsetX; }
function worldToScreenY(y) { return -y * view.scale + view.offsetY + canvas.height / (window.devicePixelRatio || 1); }
function drawGrid() {
  if (!view.showGrid) return;
  const rect = canvas.getBoundingClientRect(); const step = view.scale;
  ctx.strokeStyle = getCSSVar('--grid') || 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1; ctx.beginPath();
  for (let x = view.offsetX % step; x < rect.width; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); }
  for (let y = view.offsetY % step; y < rect.height; y += step) { ctx.moveTo(0, y); ctx.lineTo(rect.width, y); }
  ctx.stroke();
}
function drawTrack() {
  if (!trackCfg.enabled || track.length < 2) return;
  const rect = canvas.getBoundingClientRect();
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = getCSSVar('--accent') || '#6aa0ff';
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  // 只绘制当前视窗范围内的点以提升性能（简化，仍遍历一次）
  for (let i = 0; i < track.length; i++) {
    const sx = worldToScreenX(track[i].x);
    const sy = worldToScreenY(track[i].y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.restore();
}
function drawPlan() {
  if (!Array.isArray(plan) || plan.length < 2) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = getCSSVar('--accent') || '#6aa0ff';
  ctx.globalAlpha = 0.35;
  ctx.beginPath();
  for (let i = 0; i < plan.length; i++) {
    const sx = worldToScreenX(plan[i].x);
    const sy = worldToScreenY(plan[i].y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.restore();
}
function drawVehicle() {
  const x = worldToScreenX(state.x); const y = worldToScreenY(state.y);
  const L = (params.a + params.b);
  const W = (params.width || 1.8);
  const TR = (params.track || Math.max(0.5, W - 0.3));
  const w = L * view.scale; const h = W * view.scale;
  ctx.save(); ctx.translate(x, y); ctx.rotate(-state.psi);
  // 车身：矩形（后部） + 梯形（前部）组合
  const aPix = params.a * view.scale;
  const bPix = params.b * view.scale;
  const kFront = 0.65; // 前端收窄比例（0~1）
  const tPixRaw = L * 0.10 * view.scale; // 梯形深度为总长的10%
  const tPix = Math.min(aPix, Math.max(0, tPixRaw));
  ctx.fillStyle = 'rgba(106,160,255,0.9)'; ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1.5;
  // 后部矩形：从 -b 到 0，宽度为 W
  // 调整为从 -b 到 (a - tPix)，预留前10%长度画梯形
  ctx.beginPath();
  ctx.rect(-bPix, -h/2, (aPix - tPix) + bPix, h);
  ctx.fill(); ctx.stroke();
  // 前部梯形：从 0 到 +a，前端收窄为 kFront*W
  ctx.beginPath();
  ctx.moveTo(aPix - tPix, -h/2);                    // 接矩形上边
  ctx.lineTo(+aPix, -h/2 * kFront);                 // 前左（收窄）
  ctx.lineTo(+aPix, +h/2 * kFront);                 // 前右（收窄）
  ctx.lineTo(aPix - tPix, +h/2);                    // 接矩形下边
  ctx.closePath();
  ctx.fill(); ctx.stroke();

  // 车轴线
  ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.2;
  const trPix = TR * view.scale;
  // 轴线位置
  ctx.beginPath();
  ctx.moveTo(+aPix, -trPix/2); ctx.lineTo(+aPix, +trPix/2);
  ctx.moveTo(-bPix, -trPix/2); ctx.lineTo(-bPix, +trPix/2);
  ctx.stroke();

  // 轮胎矩形（按轮角旋转）
  const wheelLen = 0.6 * view.scale;
  const wheelWid = 0.25 * view.scale;
  const df = (typeof state.df === 'number') ? state.df : (ctrl.df || 0);
  const dr = (typeof state.dr === 'number') ? state.dr : (ctrl.dr || 0);
  const drawWheel = (cx, cy, ang) => {
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(-ang);
    ctx.fillStyle = 'rgba(30,30,30,0.9)'; ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1.0;
    ctx.beginPath(); ctx.rect(-wheelLen/2, -wheelWid/2, wheelLen, wheelWid); ctx.fill(); ctx.stroke();
    ctx.restore();
  };
  drawWheel(+aPix, +trPix/2, df);
  drawWheel(+aPix, -trPix/2, df);
  drawWheel(-bPix, +trPix/2, dr);
  drawWheel(-bPix, -trPix/2, dr);

  ctx.restore();
  // 质心标记
  ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI*2); ctx.fill();
}
function drawGoalVehicle() {
  const tgt = (Array.isArray(plan) && plan.length > 0) ? plan[plan.length - 1] : null;
  if (!tgt) return;
  const x = worldToScreenX(tgt.x); const y = worldToScreenY(tgt.y);
  const L = (params.a + params.b);
  const W = (params.width || 1.8);
  const TR = (params.track || Math.max(0.5, W - 0.3));
  const w = L * view.scale; const h = W * view.scale;
  const psi = tgt.psi || 0;
  ctx.save(); ctx.translate(x, y); ctx.rotate(-psi);
  // 浅色车身：后矩形 + 前梯形
  const aPix = params.a * view.scale;
  const bPix = params.b * view.scale;
  const kFront = 0.65;
  const tPixRaw = L * 0.10 * view.scale;
  const tPix = Math.min(aPix, Math.max(0, tPixRaw));
  ctx.fillStyle = 'rgba(106,160,255,0.35)'; ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1.2;
  // 后部矩形
  ctx.beginPath();
  ctx.rect(-bPix, -h/2, (aPix - tPix) + bPix, h);
  ctx.fill(); ctx.stroke();
  // 前部梯形
  ctx.beginPath();
  ctx.moveTo(aPix - tPix, -h/2);
  ctx.lineTo(+aPix, -h/2 * kFront);
  ctx.lineTo(+aPix, +h/2 * kFront);
  ctx.lineTo(aPix - tPix, +h/2);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  // 车轴线
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1.0;
  const trPix = TR * view.scale;
  ctx.beginPath();
  ctx.moveTo(+aPix, -trPix/2); ctx.lineTo(+aPix, +trPix/2);
  ctx.moveTo(-bPix, -trPix/2); ctx.lineTo(-bPix, +trPix/2);
  ctx.stroke();
  // 轮胎（不显示转角，作为终点标识）
  const wheelLen = 0.6 * view.scale;
  const wheelWid = 0.25 * view.scale;
  const drawWheel = (cx, cy) => {
    ctx.save(); ctx.translate(cx, cy);
    ctx.fillStyle = 'rgba(30,30,30,0.6)'; ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.rect(-wheelLen/2, -wheelWid/2, wheelLen, wheelWid); ctx.fill(); ctx.stroke();
    ctx.restore();
  };
  drawWheel(+aPix, +trPix/2);
  drawWheel(+aPix, -trPix/2);
  drawWheel(-bPix, +trPix/2);
  drawWheel(-bPix, -trPix/2);
  ctx.restore();
}
function drawArrivalEffect() {
  if (!arrivalAnim.active) return;
  const t = performance.now() - (arrivalAnim.t0 || 0);
  if (t > arrivalAnim.duration) { arrivalAnim.active = false; return; }
  const tgt = (Array.isArray(plan) && plan.length > 0) ? plan[plan.length - 1] : null;
  if (!tgt) return;
  const x = worldToScreenX(tgt.x || 0);
  const y = worldToScreenY(tgt.y || 0);
  const ok = getCSSVar('--ok') || '#4caf50';
  ctx.save();
  // 脉冲圆环
  const cycles = 3;
  for (let i = 0; i < cycles; i++) {
    const phase = Math.max(0, Math.min(1, (t / arrivalAnim.duration) + i * 0.25));
    const r = (params.width || 1.8) * view.scale * (0.6 + 0.8 * phase);
    const alpha = 1.0 - phase;
    ctx.beginPath();
    ctx.strokeStyle = ok;
    ctx.globalAlpha = 0.7 * alpha;
    ctx.lineWidth = 3;
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  // 成功提示
  ctx.globalAlpha = 0.95;
  ctx.fillStyle = ok;
  ctx.font = `${Math.round(14 * (window.devicePixelRatio || 1))}px sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText('到达终点', x, y - (params.width || 1.8) * view.scale * 0.8);
  ctx.textBaseline = 'top';
  ctx.fillText('✓', x, y + (params.width || 1.8) * view.scale * 0.1);
  ctx.restore();
}
function draw() { 
  const rect = canvas.getBoundingClientRect(); 
  ctx.clearRect(0, 0, rect.width, rect.height); 
  
  drawGrid(); 
  
  // [新增] 绘制地图边界
  drawMapBounds();
  drawObstacles();

  // [新增] 绘制 A* 全局路径 (绿色虚线)
  // 参数: points, color, width, dashArray
  drawPath(globalPath, '#4caf50', 2, [6, 6]); 

  // [新增] 绘制 Lattice 局部路径 (红色实线)
  // Lattice 路径通常较密，画细一点
  drawPath(localPath, '#e74c3c', 2, []);

  // [新增] 绘制手动设置的终点
  if (goalMarker) {
    drawTarget(goalMarker);
  }

  // 保留原有的历史轨迹绘制
  drawTrack(); 
  
  // 绘制车辆
  drawVehicle(); 
  
  // 绘制到达特效
  drawArrivalEffect(); 
}
// 导出当前保留的轨迹为 CSV 文件
function exportTrackCSV() {
  try {
    if (!Array.isArray(track) || track.length === 0) { alert('当前无可导出的轨迹'); return; }
    const hasTime = typeof track[0].t === 'number';
    const t0 = hasTime ? track[0].t : null;
    const rows = [];
    rows.push('t,x,y');
    for (let i = 0; i < track.length; i++) {
      const p = track[i] || {};
      const t = (hasTime && typeof p.t === 'number') ? (t0 !== null ? (p.t - t0) : p.t) : '';
      const x = (typeof p.x === 'number') ? p.x : '';
      const y = (typeof p.y === 'number') ? p.y : '';
      rows.push(`${t},${x},${y}`);
    }
    const csv = rows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = `track_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  } catch (e) {
    console.warn('导出 CSV 失败', e);
    alert('导出失败，请稍后重试');
  }
}

// 导出当前保留的遥测为 CSV 文件
function exportTelemetryCSV() {
  try {
    if (!Array.isArray(telemetry) || telemetry.length === 0) { alert('当前无可导出的遥测'); return; }
    const t0 = telemetry[0].t_ms;
    const rows = [];
    rows.push('t_ms,U,x,y,psi,df,dr,beta_dot,r_dot,speed,radius');
    for (let i = 0; i < telemetry.length; i++) {
      const p = telemetry[i] || {};
      const t = (typeof p.t_ms === 'number') ? (p.t_ms - t0).toFixed(0) : '';
      const U = (typeof p.U === 'number') ? p.U : '';
      const x = (typeof p.x === 'number') ? p.x : '';
      const y = (typeof p.y === 'number') ? p.y : '';
      const psi = (typeof p.psi === 'number') ? p.psi : '';
      const df = (typeof p.df === 'number') ? p.df : '';
      const dr = (typeof p.dr === 'number') ? p.dr : '';
      const beta_dot = (typeof p.beta_dot === 'number') ? p.beta_dot : '';
      const r_dot = (typeof p.r_dot === 'number') ? p.r_dot : '';
      const speed = (typeof p.speed === 'number') ? p.speed : '';
      const radius = (typeof p.radius === 'number') ? p.radius : '';
      rows.push(`${t},${U},${x},${y},${psi},${df},${dr},${beta_dot},${r_dot},${speed},${radius}`);
    }
    const csv = rows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = `telemetry_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  } catch (e) {
    console.warn('导出遥测 CSV 失败', e);
    alert('导出失败，请稍后重试');
  }
}

// 导出当前规划轨迹为 CSV 文件
function exportPlanCSV() {
  try {
    if (!Array.isArray(plan) || plan.length === 0) { alert('当前无可导出的规划'); return; }
    const hasTime = typeof plan[0].t === 'number';
    const t0 = hasTime ? plan[0].t : null;
    const rows = [];
    rows.push('t,x,y,psi');
    for (let i = 0; i < plan.length; i++) {
      const p = plan[i] || {};
      const t = (hasTime && typeof p.t === 'number') ? (t0 !== null ? (p.t - t0) : p.t) : '';
      const x = (typeof p.x === 'number') ? p.x : '';
      const y = (typeof p.y === 'number') ? p.y : '';
      const psi = (typeof p.psi === 'number') ? p.psi : '';
      rows.push(`${t},${x},${y},${psi}`);
    }
    const csv = rows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = `plan_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  } catch (e) {
    console.warn('导出规划 CSV 失败', e);
    alert('导出失败，请稍后重试');
  }
}

// 后端交互（节流）
let postTimer = null;
function schedulePostParams() {
  if (postTimer) clearTimeout(postTimer);
  postTimer = setTimeout(async () => {
    try {
      const patch = { m: params.m, Iz: params.Iz, a: params.a, b: params.b, width: params.width, track: params.track, kf: params.kf, kr: params.kr, U: params.U, mu: params.mu, g: params.g, U_min: params.U_min, U_blend: params.U_blend, tire_model: params.tire_model };
      const res = await fetch('/api/params', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) });
      setConnStatus(res.ok);
    } catch (e) { console.warn('POST /api/params 失败', e); setConnStatus(false); }
  }, 200);
}
async function loadParams() {
  try {
    const res = await fetch('/api/params'); setConnStatus(res.ok);
    const data = await res.json();
    // 覆写参数
    for (const k of ['m','Iz','a','b','width','track','kf','kr','U','mu','g','U_min','U_blend','tire_model']) { if (typeof data[k] !== 'undefined') params[k] = data[k]; }
    // 更新输入框
    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = String(v); };
    setVal('speed_input', (ctrl.U ?? params.U).toFixed(2));
    setVal('df_input', deg(ctrl.df ?? 0).toFixed(1));
    setVal('dr_input', deg(ctrl.dr ?? 0).toFixed(1));
    setVal('p_m', params.m); setVal('p_Iz', params.Iz); setVal('p_a', params.a); setVal('p_b', params.b);
    setVal('p_kf', params.kf); setVal('p_kr', params.kr); setVal('p_width', params.width); setVal('p_track', params.track); setVal('p_mu', params.mu); setVal('p_g', params.g); setVal('p_Umin', params.U_min); setVal('p_Ublend', params.U_blend);
    setVal('p_tire_model', params.tire_model);
    // 更新派生量显示（也可从后端 data.L / data.K）
    const pL = document.getElementById('p_L'); const pK = document.getElementById('p_K');
    if (pL && typeof data.L !== 'undefined') pL.textContent = Number(data.L).toFixed(3);
    if (pK && typeof data.K !== 'undefined') pK.textContent = Number(data.K).toFixed(6);
  updateDashboard();
  } catch (e) { console.warn('GET /api/params 失败', e); setConnStatus(false); }
}

// 初始化
window.addEventListener('load', async () => { 
  await loadParams();      // 加载车辆参数
  await fetchMapInfo();    // [新增] 加载地图参数
  bindUI(); 
  updateDerivedStats(); 
});

// 工具：度/弧度转换
function normAngle(a) { let x = a; while (x > Math.PI) x -= 2 * Math.PI; while (x < -Math.PI) x += 2 * Math.PI; return x; }
function angleDiff(a, b) { return normAngle(a - b); }
function deg(rad) { return rad * 180 / Math.PI; }
function rad(deg) { return deg * Math.PI / 180; }
// 每图控制：显示/隐藏、锁定Y范围、导出CSV
const plotCfg = {
  epsi: { show: true, lockY: false, yMin: null, yMax: null },
  ey:   { show: true, lockY: false, yMin: null, yMax: null },
  df:   { show: true, lockY: false, yMin: null, yMax: null },
  beta: { show: true, lockY: false, yMin: null, yMax: null },
  u:    { show: true, lockY: false, yMin: null, yMax: null },
};
const plotPaused = { epsi: true, ey: true, df: true, beta: true, u: true };
let globalPaused = true;

function applyYLock(key) {
  const chart = charts[key]; if (!chart) return;
  const cfg = plotCfg[key];
  if (cfg.lockY && isFinite(cfg.yMin ?? NaN) && isFinite(cfg.yMax ?? NaN)) {
    chart.options.scales.y.min = Number(cfg.yMin);
    chart.options.scales.y.max = Number(cfg.yMax);
  } else {
    chart.options.scales.y.min = undefined;
    chart.options.scales.y.max = undefined;
  }
  chart.update('none');
}

function exportChartCSV(key) {
  const chart = charts[key]; if (!chart) return;
  const labels = chart.data.labels.map((s) => Number(s));
  const cols = chart.data.datasets.map((ds) => ({ name: ds.label || 'value', data: ds.data }));
  let header = 't';
  for (const c of cols) header += `,${c.name}`;
  let csv = header + '\n';
  for (let i = 0; i < labels.length; i++) {
    const row = [labels[i]];
    for (const c of cols) row.push(c.data[i] ?? '');
    csv += row.join(',') + '\n';
  }
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `${key}.csv`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
}

function exportAllChartsCSV() {
  // 合并导出：t、epsi、ey、df、dr、beta
  const epsi = charts.epsi; const ey = charts.ey; const df = charts.df; const beta = charts.beta; const u = charts.u;
  if (!epsi || !ey || !df || !beta || !u) { alert('图表尚未就绪'); return; }
  const N = Math.min(epsi.data.labels.length, ey.data.labels.length, df.data.labels.length, beta.data.labels.length, u.data.labels.length);
  const labels = epsi.data.labels.slice(-N).map((s) => Number(s));
  const epsiVals = epsi.data.datasets[0].data.slice(-N);
  const eyVals = ey.data.datasets[0].data.slice(-N);
  const dfVals = df.data.datasets[0].data.slice(-N);
  const drVals = df.data.datasets[1] ? df.data.datasets[1].data.slice(-N) : [];
  const betaVals = beta.data.datasets[0].data.slice(-N);
  const uVals = u.data.datasets[0].data.slice(-N);
  let csv = 't,epsi,ey,df,dr,beta,U\n';
  for (let i = 0; i < N; i++) {
    const row = [labels[i], epsiVals[i] ?? '', eyVals[i] ?? '', dfVals[i] ?? '', drVals[i] ?? '', betaVals[i] ?? '', uVals[i] ?? ''];
    csv += row.join(',') + '\n';
  }
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  a.download = `charts_${ts}.csv`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
}

function bindPlotControls() {
  const keys = ['epsi','ey','df','beta'];
  for (const key of keys) {
    const exportEl = document.getElementById(`export_${key}`);
    const pauseEl = document.getElementById(`pause_${key}`);
    const clearEl = document.getElementById(`clear_${key}`);
    if (exportEl) exportEl.addEventListener('click', () => exportChartCSV(key));
    if (pauseEl) {
      pauseEl.addEventListener('click', () => {
        plotPaused[key] = !plotPaused[key];
        pauseEl.textContent = plotPaused[key] ? '开始记录' : '暂停记录';
      });
    }
    if (clearEl) clearEl.addEventListener('click', () => clearChart(key));
  }
}

function bindGlobalPlotControls() {
  const pauseBtn = document.getElementById('global_pause');
  const clearBtn = document.getElementById('global_clear');
  const exportBtn = document.getElementById('global_export');
  // 初始按钮文案与全局暂停状态
  if (pauseBtn) { pauseBtn.textContent = globalPaused ? '开始记录' : '暂停记录'; }
  for (const k of Object.keys(plotPaused)) plotPaused[k] = globalPaused;
  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
      globalPaused = !globalPaused;
      // 同步所有图表的暂停状态
      for (const k of Object.keys(plotPaused)) plotPaused[k] = globalPaused;
      pauseBtn.textContent = globalPaused ? '开始记录' : '暂停记录';
    });
  }
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      for (const k of Object.keys(charts)) clearChart(k);
      // 重置时间轴，使清空后重新从0开始
      plotsStartMs = performance.now();
    });
  }
  if (exportBtn) exportBtn.addEventListener('click', () => exportAllChartsCSV());
}

function clearChart(key) {
  const chart = charts[key]; if (!chart) return;
  chart.data.labels = [];
  for (const ds of chart.data.datasets) { ds.data = []; }
  chart.update('none');
}


// [新增] 1. 获取地图元数据
async function fetchMapInfo() {
  try {
    const res = await fetch('/api/map/info');
    if (res.ok) {
      mapInfo = await res.json();
      console.log('地图参数已加载:', mapInfo);
    }
  } catch (e) { console.warn('获取地图参数失败', e); }
}

// [新增] 2. 屏幕坐标转世界坐标 (用于鼠标点击)
function screenToWorld(sx, sy) {
  // 反解 worldToScreenX/Y 公式
  // worldToScreenX: sx = x * scale + offsetX
  // worldToScreenY: sy = -y * scale + offsetY + H
  const dpr = window.devicePixelRatio || 1;
  const H = canvas.height / dpr;
  
  const x = (sx - view.offsetX) / view.scale;
  const y = (view.offsetY + H - sy) / view.scale;
  return { x, y };
}

// [新增] 3. 通用路径绘制 (支持虚线)
function drawPath(points, color, width, dash) {
  if (!Array.isArray(points) || points.length < 2) return;
  ctx.save();
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const sx = worldToScreenX(points[i].x);
    const sy = worldToScreenY(points[i].y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.restore();
}

// [新增] 4. 绘制终点标记 (红色图钉)
function drawTarget(pt) {
  const sx = worldToScreenX(pt.x);
  const sy = worldToScreenY(pt.y);
  ctx.save();
  ctx.translate(sx, sy);
  
  // 画个圈
  ctx.fillStyle = 'rgba(231, 76, 60, 0.8)'; // 红色
  ctx.beginPath();
  ctx.arc(0, 0, 6, 0, Math.PI*2);
  ctx.fill();
  
  // 画个杆子 (模拟插旗)
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0); ctx.lineTo(0, -12);
  ctx.stroke();
  
  // 文字
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('GOAL', 0, -16);
  
  ctx.restore();
}

// [新增] 5. 绘制地图边界 (辅助调试)
function drawMapBounds() {
  if (!mapInfo) return;
  const x0 = worldToScreenX(mapInfo.origin_x);
  const y0 = worldToScreenY(mapInfo.origin_y); // 左下角(世界) -> 屏幕Y较大值
  const x1 = worldToScreenX(mapInfo.origin_x + mapInfo.width * mapInfo.resolution);
  const y1 = worldToScreenY(mapInfo.origin_y + mapInfo.height * mapInfo.resolution); // 右上角(世界) -> 屏幕Y较小值
  
  // 屏幕坐标系中，y1 < y0
  const w = x1 - x0;
  const h = y0 - y1; 
  
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  // canvas rect: x, y, w, h. 这里左上角是 (x0, y1)
  ctx.strokeRect(x0, y1, w, h);
  ctx.restore();
}

// [新增] 绘制障碍物
// [新增/修改] 绘制障碍物 (受 showObstacles 开关控制)
function drawObstacles() {
  // 关键修改：增加 showObstacles 判断
  // 只有当用户点击了"加载测试场景"按钮 (showObstacles=true) 且地图数据存在时才绘制
  if (!showObstacles || !mapInfo || !mapInfo.obstacles) return;
  
  ctx.save();
  ctx.fillStyle = 'rgba(255, 80, 80, 0.6)'; // 半透明红色
  
  for (let obs of mapInfo.obstacles) {
    // map_server.py 传回来的 obstacles 是列表的列表 [[x1, y1], [x2, y2], ...]
    const wx = obs[0];
    const wy = obs[1];
    
    // 转换到屏幕坐标
    const sx = worldToScreenX(wx);
    const sy = worldToScreenY(wy);
    
    // 障碍物尺寸：对应 map_server.py 中的膨胀尺寸 (2m x 1m)
    // 根据当前视图缩放比例 (view.scale) 计算像素大小
    const w = 2.0 * view.scale; 
    const h = 1.0 * view.scale;
    
    // 绘制矩形 (以中心点为基准)
    ctx.fillRect(sx - w/2, sy - h/2, w, h);
  }
  
  ctx.restore();
}