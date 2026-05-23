"""Generates a self-contained HTML 3D skeleton viewer from k3d.pkl."""
import base64
import json
import logging
import os
import pickle

import numpy as np

from src.data_io.path_config import GamePath

# Body25 skeleton connections (zero-indexed)
BODY25_SKELETON = [
  [0, 1],
  [1, 2], [2, 3], [3, 4],
  [1, 5], [5, 6], [6, 7],
  [1, 8],
  [8, 9], [9, 10], [10, 11],
  [8, 12], [12, 13], [13, 14],
  [0, 15], [15, 17],
  [0, 16], [16, 18],
  [14, 19], [19, 20], [14, 21],
  [11, 22], [22, 23], [11, 24],
]

PLAYER_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#9a6324", "#800000",
    "#aaffc3", "#808000", "#000075", "#a9a9a9",
]

# ---------------------------------------------------------------------------
# HTML template  –  __SCENE_DATA__ is replaced at runtime with serialized JSON
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>K3D Skeleton Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1923;color:#dde;font-family:system-ui,sans-serif;overflow:hidden}
#wrap{position:fixed;inset:0}
#ui{
  position:fixed;top:12px;left:12px;z-index:9;
  background:rgba(5,10,20,.88);backdrop-filter:blur(8px);
  border:1px solid #1e3a5a;border-radius:10px;
  padding:14px 16px;width:266px;
  max-height:calc(100vh - 24px);overflow-y:auto;font-size:12px;
  scrollbar-width:thin;scrollbar-color:#1e3a5a transparent
}
h2{font-size:13px;color:#6af;margin-bottom:8px}
h3{font-size:11px;color:#59b;margin:10px 0 5px;letter-spacing:.5px;text-transform:uppercase}
.r{display:flex;align-items:center;gap:6px;margin:5px 0}
.r label{flex-shrink:0;width:84px;color:#99a}
.r .v{flex-shrink:0;width:38px;text-align:right;color:#ccc}
input[type=range]{flex:1;cursor:pointer;accent-color:#4a90e2}
.br{display:flex;gap:5px;margin:8px 0}
button{
  flex:1;padding:5px 0;border:none;border-radius:5px;
  background:#1e4a8e;color:#cdf;font-size:11px;cursor:pointer;
  transition:background .15s
}
button:hover{background:#2a6ace}
button.on{background:#1a7a3a;color:#afc}
hr{border:none;border-top:1px solid #1e3a5a;margin:8px 0}
#fdisp{font-size:11px;color:#88a;text-align:right;margin:2px 0 6px}
.pi{display:flex;align-items:center;gap:7px;padding:3px 4px;border-radius:5px;cursor:pointer}
.pi:hover{background:rgba(255,255,255,.06)}
.dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
input[type=checkbox]{accent-color:#4a90e2;cursor:pointer}
.tip{margin-top:10px;font-size:10px;color:#447;line-height:1.7}
</style>
</head>
<body>
<div id="wrap"></div>
<div id="ui">
  <h2>&#129464; K3D Skeleton Viewer</h2>
  <div id="fdisp">&#8212;</div>
  <div class="r">
    <label>Frame</label>
    <input type="range" id="sf" min="0" value="0" step="1" style="flex:1">
  </div>
  <div class="br">
    <button id="bplay">&#9654; Play</button>
    <button id="bprev" title="Prev frame">&#9664;</button>
    <button id="bnext" title="Next frame">&#9654;</button>
  </div>
  <div class="r">
    <label>Speed</label>
    <input type="range" id="sspd" min="0.1" max="5" step="0.1" value="1">
    <span class="v" id="vspd">1.0&#xD7;</span>
  </div>
  <hr>
  <div class="r">
    <label>Confidence</label>
    <input type="range" id="sconf" min="0" max="1" step="0.05" value="0.3">
    <span class="v" id="vconf">0.30</span>
  </div>
  <div class="r">
    <label>Joint size</label>
    <input type="range" id="sjsz" min="0.01" max="0.3" step="0.005" value="0.05">
  </div>
  <hr>
  <h3>Players</h3>
  <div id="plist"></div>
  <p class="tip">
    &#9000;&nbsp;Space: play/pause &nbsp;|&nbsp; &#8592;/&#8594;: step frame<br>
    &#128432;&nbsp;L-drag: orbit &nbsp; R-drag: pan &nbsp; Scroll: zoom
  </p>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
/* ===== EMBEDDED DATA ===== */
const SD = __SCENE_DATA__;
/* ========================= */

const {T, K, fps: FPS, skeleton: SKEL, players: PLAYERS, court: {width: CW, length: CL}} = SD;

/* Decode base64 string -> Float32Array */
function f32(b64) {
  const s = atob(b64), n = s.length;
  const buf = new ArrayBuffer(n);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < n; i++) u8[i] = s.charCodeAt(i);
  return new Float32Array(buf);
}

/*
 * Coordinate mapping:  world(x, y, z)  ->  THREE(x, z, -y)
 *   world x  = court width  direction  ->  THREE X
 *   world y  = court length direction  ->  THREE Z  (negated so court faces +Z camera)
 *   world z  = height                  ->  THREE Y  (Y-up scene)
 */
const w2t = (x, y, z) => [x, z, -y];

/* ===== RENDERER & CAMERA ===== */
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
document.getElementById('wrap').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1923);
scene.fog = new THREE.FogExp2(0x0f1923, 0.010);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 500);
camera.position.set(CW / 2, 14, CL * 0.65);
camera.lookAt(CW / 2, 0.5, -CL / 2);

const orbit = new THREE.OrbitControls(camera, renderer.domElement);
orbit.target.set(CW / 2, 0, -CL / 2);
orbit.enableDamping = true;
orbit.dampingFactor = 0.07;
orbit.update();

/* ===== LIGHTING ===== */
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dlight = new THREE.DirectionalLight(0xddeeff, 1.0);
dlight.position.set(CW / 2 + 5, 15, 5);
scene.add(dlight);
scene.add(new THREE.HemisphereLight(0x8ab4ff, 0x223322, 0.35));

/* ===== COURT ===== */
(function buildCourt() {
  // Extended ground plane
  const gnd = new THREE.Mesh(
    new THREE.PlaneGeometry(CW + 20, CL + 20),
    new THREE.MeshLambertMaterial({color: 0x0a1a0a})
  );
  gnd.rotation.x = -Math.PI / 2;
  gnd.position.set(CW / 2, -0.01, -CL / 2);
  scene.add(gnd);

  // Court surface
  const sur = new THREE.Mesh(
    new THREE.PlaneGeometry(CW, CL),
    new THREE.MeshLambertMaterial({color: 0x1e4a1e})
  );
  sur.rotation.x = -Math.PI / 2;
  sur.position.set(CW / 2, 0, -CL / 2);
  scene.add(sur);

  // Grid
  const grid = new THREE.GridHelper(Math.max(CW, CL) + 6, 24, 0x1e3e1e, 0x162b16);
  grid.position.set(CW / 2, -0.005, -CL / 2);
  scene.add(grid);

  // Helper: draw a line through world-coord point array
  function line3(pts, color) {
    const pos = new Float32Array(pts.length * 3);
    pts.forEach(([x, y, z], i) => {
      [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]] = w2t(x, y, z);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({color})));
  }

  // Court boundary
  line3([[0,0,0],[CW,0,0],[CW,CL,0],[0,CL,0],[0,0,0]], 0xffffff);
  // Centre line
  line3([[0, CL/2, 0], [CW, CL/2, 0]], 0xffffff);
  // Net posts
  line3([[0,  CL/2, 0], [0,  CL/2, 2.5]], 0xffaa44);
  line3([[CW, CL/2, 0], [CW, CL/2, 2.5]], 0xffaa44);
  // Net top wire
  line3([[0, CL/2, 2.43], [CW, CL/2, 2.43]], 0xffcc66);
  // Coordinate axes (1.5 m)
  scene.add(new THREE.AxesHelper(1.5));
})();

/* ===== PLAYER MESHES ===== */
const PM = {};

for (const [pid, pd] of Object.entries(PLAYERS)) {
  const arr = f32(pd.b64);           // Float32Array, length = T * K * 4
  const col = new THREE.Color(pd.color);
  const grp = new THREE.Group();

  // Joint spheres
  const jGeo = new THREE.SphereGeometry(0.05, 8, 5);
  const joints = [];
  for (let k = 0; k < K; k++) {
    const m = new THREE.Mesh(jGeo, new THREE.MeshLambertMaterial({color: col}));
    m.visible = false;
    grp.add(m);
    joints.push(m);
  }

  // Bone line segments (one segment = 2 vertices = 6 floats)
  const bpos = new Float32Array(SKEL.length * 6);
  const bGeo = new THREE.BufferGeometry();
  const bAttr = new THREE.BufferAttribute(bpos, 3);
  bAttr.setUsage(THREE.DynamicDrawUsage);
  bGeo.setAttribute('position', bAttr);
  const bLines = new THREE.LineSegments(bGeo, new THREE.LineBasicMaterial({color: col, linewidth: 2}));
  grp.add(bLines);

  scene.add(grp);
  PM[pid] = {arr, grp, joints, bpos, bGeo, bAttr, bLines, visible: true};
}

/* ===== RENDER A FRAME ===== */
let confThr = 0.3;

function render(t) {
  for (const [pid, pm] of Object.entries(PM)) {
    pm.grp.visible = pm.visible;
    if (!pm.visible) continue;

    const A = pm.arr;
    const b0 = t * K * 4;
    const wp = new Array(K).fill(null);

    // Update joint positions
    for (let k = 0; k < K; k++) {
      const b = b0 + k * 4;
      const conf = A[b + 3];
      if (conf >= confThr) {
        const p = w2t(A[b], A[b + 1], A[b + 2]);
        pm.joints[k].position.set(p[0], p[1], p[2]);
        pm.joints[k].visible = true;
        wp[k] = p;
      } else {
        pm.joints[k].visible = false;
      }
    }

    // Update bone segments
    const bp = pm.bpos;
    for (let i = 0; i < SKEL.length; i++) {
      const pa = wp[SKEL[i][0]], pb = wp[SKEL[i][1]];
      if (pa && pb) {
        bp[i*6]   = pa[0]; bp[i*6+1] = pa[1]; bp[i*6+2] = pa[2];
        bp[i*6+3] = pb[0]; bp[i*6+4] = pb[1]; bp[i*6+5] = pb[2];
      } else {
        bp.fill(0, i * 6, i * 6 + 6); // degenerate segment (invisible)
      }
    }
    pm.bAttr.needsUpdate = true;
  }
}

/* ===== PLAYBACK STATE ===== */
let cur = 0, playing = false, spd = 1.0, lastTs = null, accum = 0;

function setF(f) {
  cur = Math.max(0, Math.min(T - 1, Math.round(f)));
  document.getElementById('sf').value = cur;
  document.getElementById('fdisp').textContent = 'Frame ' + cur + ' / ' + (T - 1);
  render(cur);
}

/* ===== ANIMATION LOOP ===== */
function loop(ts) {
  requestAnimationFrame(loop);
  if (playing && lastTs !== null) {
    accum += (ts - lastTs) * spd;
    const mpf = 1000 / FPS;
    if (accum >= mpf) {
      const adv = Math.floor(accum / mpf);
      accum -= adv * mpf;
      setF((cur + adv) % T);
    }
  }
  lastTs = ts;
  orbit.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(loop);
setF(0);

/* ===== UI WIRING ===== */
document.getElementById('sf').max = T - 1;
document.getElementById('sf').addEventListener('input', e => setF(+e.target.value));

const bplay = document.getElementById('bplay');
bplay.addEventListener('click', () => {
  playing = !playing;
  accum = 0;
  bplay.textContent = playing ? '\u23F8 Pause' : '\u25B6 Play';
  bplay.classList.toggle('on', playing);
});

document.getElementById('bprev').addEventListener('click', () => setF(cur - 1));
document.getElementById('bnext').addEventListener('click', () => setF(cur + 1));

document.getElementById('sspd').addEventListener('input', e => {
  spd = +e.target.value;
  document.getElementById('vspd').textContent = spd.toFixed(1) + '\u00D7';
});

document.getElementById('sconf').addEventListener('input', e => {
  confThr = +e.target.value;
  document.getElementById('vconf').textContent = confThr.toFixed(2);
  render(cur);
});

document.getElementById('sjsz').addEventListener('input', e => {
  const s = +e.target.value / 0.05;
  for (const pm of Object.values(PM)) pm.joints.forEach(j => j.scale.setScalar(s));
});

/* Player toggles */
const plist = document.getElementById('plist');
for (const [pid, pd] of Object.entries(PLAYERS)) {
  const row = document.createElement('div');
  row.className = 'pi';
  const dot = document.createElement('span');
  dot.className = 'dot';
  dot.style.background = pd.color;
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.addEventListener('change', () => { PM[pid].visible = cb.checked; render(cur); });
  const lbl = document.createElement('span');
  lbl.textContent = 'Player ' + pd.label;
  row.append(dot, cb, lbl);
  row.addEventListener('click', ev => {
    if (ev.target !== cb) { cb.checked = !cb.checked; cb.dispatchEvent(new Event('change')); }
  });
  plist.appendChild(row);
}

/* Keyboard */
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if      (e.code === 'Space')      { bplay.click(); e.preventDefault(); }
  else if (e.code === 'ArrowLeft')  { setF(cur - 1); e.preventDefault(); }
  else if (e.code === 'ArrowRight') { setF(cur + 1); e.preventDefault(); }
});

/* Resize */
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------

def _build_scene_data(
    k3d_result: dict,
    fps: float,
    court_width: float,
    court_length: float,
) -> dict:
    """Convert {player_id: ndarray(T, K, 4)} to JSON-serialisable scene data."""
    pid_sorted = sorted(k3d_result.keys())
    first = k3d_result[pid_sorted[0]]
    T, K = int(first.shape[0]), int(first.shape[1])

    players = {}
    for i, pid in enumerate(pid_sorted):
        arr = np.ascontiguousarray(k3d_result[pid], dtype=np.float32)  # (T, K, 4)
        b64 = base64.b64encode(arr.tobytes()).decode("ascii")
        players[str(pid)] = {
            "b64": b64,
            "color": PLAYER_COLORS[i % len(PLAYER_COLORS)],
            "label": str(pid),
        }

    return {
        "T": T,
        "K": K,
        "fps": float(fps),
        "skeleton": BODY25_SKELETON,
        "players": players,
        "court": {"width": float(court_width), "length": float(court_length)},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_processor(data_path_cfg: GamePath, fps: float = None) -> None:
    """
    Generate a self-contained HTML 3D skeleton viewer from the model-tagged k3d file.

    The HTML file embeds all joint data as base64-encoded Float32Arrays and uses
    Three.js (loaded from CDN) for interactive 3D rendering.  Open the generated
    file in any modern browser – internet access is required for the CDN scripts.

    Args:
        data_path_cfg: GamePath configuration object.
        fps:           Playback frame-rate.  Defaults to video_fps / frame_step.
    """
    logger = logging.getLogger("visualize_processor")

    output_html = data_path_cfg.get_output_artifact_path("k3d_vis", ".html")
    if os.path.exists(output_html):
        logger.info(f"Visualization already exists at {output_html}, skipping.")
        return

    k3d_path = data_path_cfg.find_output_artifact_path("k3d", ".pkl")
    if not os.path.exists(k3d_path):
        logger.error(
            f"k3d.pkl not found at {k3d_path}. Run collect_processor first."
        )
        return

    logger.info(f"Loading k3d data from {k3d_path}...")
    with open(k3d_path, "rb") as f:
        k3d_result: dict = pickle.load(f)

    if not k3d_result:
        logger.warning("k3d.pkl is empty, nothing to visualize.")
        return

    if fps is None:
        fps = data_path_cfg.fps / max(1, data_path_cfg.frame_step)

    from src.utils import constant
    court_width = float(constant.court_width)
    court_length = float(constant.court_height)  # court_height = Y-dimension (length)

    logger.info(
        f"Building scene: {len(k3d_result)} players, "
        f"T={next(iter(k3d_result.values())).shape[0]} frames, fps={fps:.1f}"
    )
    scene_data = _build_scene_data(k3d_result, fps, court_width, court_length)

    scene_json = json.dumps(scene_data, separators=(",", ":"))
    html_content = _HTML.replace("__SCENE_DATA__", scene_json)

    os.makedirs(data_path_cfg.output_dir, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Visualization saved → {output_html}")
    logger.info("Open the file in a browser (Three.js loaded from CDN).")
