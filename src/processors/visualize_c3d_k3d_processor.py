from __future__ import annotations

import base64
import json
import logging
import os
import pickle

import numpy as np

from src.data_io.path_config import GamePath
from src.processors.c3d_alignment_processor import c3d_alignment_processor


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

JOINT_EDGE_NAMES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C3D + K3D Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#091017;color:#dde;font-family:system-ui,sans-serif;overflow:hidden}
#wrap{position:fixed;inset:0}
#ui{
  position:fixed;top:12px;left:12px;z-index:9;
  background:rgba(6,12,18,.9);backdrop-filter:blur(8px);
  border:1px solid #1e3a5a;border-radius:12px;
  padding:14px 16px;width:308px;
  max-height:calc(100vh - 24px);overflow-y:auto;font-size:12px;
}
h2{font-size:14px;color:#72d6ff;margin-bottom:8px}
h3{font-size:11px;color:#5caed6;margin:10px 0 6px;letter-spacing:.5px;text-transform:uppercase}
.r{display:flex;align-items:center;gap:6px;margin:5px 0}
.r label{flex-shrink:0;width:98px;color:#9bb7c7}
.r .v{flex-shrink:0;width:52px;text-align:right;color:#d8e6ee}
input[type=range]{flex:1;cursor:pointer;accent-color:#2ec4ff}
.br{display:flex;gap:5px;margin:8px 0}
button{
  flex:1;padding:6px 0;border:none;border-radius:6px;
  background:#1a4d78;color:#d7f3ff;font-size:11px;cursor:pointer;
}
button:hover{background:#2570ad}
button.on{background:#1f7a44;color:#d4ffd8}
hr{border:none;border-top:1px solid #1e3a5a;margin:9px 0}
#fdisp{font-size:11px;color:#88a8ba;text-align:right;margin:2px 0 6px}
.pi{display:flex;align-items:center;gap:7px;padding:3px 4px;border-radius:5px;cursor:pointer}
.pi:hover{background:rgba(255,255,255,.06)}
.dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
input[type=checkbox]{accent-color:#2ec4ff;cursor:pointer}
.tip{margin-top:10px;font-size:10px;color:#53707f;line-height:1.7}
.meta{font-size:11px;line-height:1.6;color:#c8d6df;background:rgba(255,255,255,.04);padding:8px;border-radius:8px}
.meta b{color:#fff}
</style>
</head>
<body>
<div id="wrap"></div>
<div id="ui">
  <h2>C3D + K3D Joint Viewer</h2>
  <div id="fdisp">—</div>
  <div class="r">
    <label>Frame</label>
    <input type="range" id="sf" min="0" value="0" step="1" style="flex:1">
  </div>
  <div class="br">
    <button id="bplay">▶ Play</button>
    <button id="bprev">◀</button>
    <button id="bnext">▶</button>
  </div>
  <div class="r">
    <label>Speed</label>
    <input type="range" id="sspd" min="0.1" max="5" step="0.1" value="1">
    <span class="v" id="vspd">1.0×</span>
  </div>
  <div class="r">
    <label>K3D conf</label>
    <input type="range" id="sconf" min="0" max="1" step="0.05" value="0.3">
    <span class="v" id="vconf">0.30</span>
  </div>
  <div class="r">
    <label>K3D joint size</label>
    <input type="range" id="sjsz" min="0.01" max="0.3" step="0.005" value="0.05">
  </div>
  <div class="r">
    <label>C3D joint size</label>
    <input type="range" id="sc3djsz" min="0.01" max="0.35" step="0.005" value="0.07">
  </div>
  <hr>
  <h3>Overlays</h3>
  <div class="r"><label>C3D overlay</label><input type="checkbox" id="showc3d" checked></div>
  <div class="r"><label>Selected player</label><input type="checkbox" id="showsel" checked></div>
  <hr>
  <h3>Players</h3>
  <div id="plist"></div>
  <hr>
  <h3>Alignment</h3>
  <div class="meta" id="meta"></div>
  <p class="tip">Space: play/pause | ←/→: frame step<br>鼠标左键旋转，右键平移，滚轮缩放</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const SD = __SCENE_DATA__;
const {T, K, fps: FPS, skeleton: SKEL, players: PLAYERS, court: {width: CW, length: CL}, c3d: C3D} = SD;

function f32(b64) {
  const s = atob(b64), n = s.length;
  const buf = new ArrayBuffer(n);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < n; i++) u8[i] = s.charCodeAt(i);
  return new Float32Array(buf);
}

function u8fromb64(b64) {
  const s = atob(b64), n = s.length;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = s.charCodeAt(i);
  return out;
}

const w2t = (x, y, z) => [x, z, -y];

const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.getElementById('wrap').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x091017);
scene.fog = new THREE.FogExp2(0x091017, 0.011);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 500);
camera.position.set(CW / 2, 14, CL * 0.65);
camera.lookAt(CW / 2, 0.5, -CL / 2);

const orbit = new THREE.OrbitControls(camera, renderer.domElement);
orbit.target.set(CW / 2, 0, -CL / 2);
orbit.enableDamping = true;
orbit.dampingFactor = 0.07;
orbit.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dlight = new THREE.DirectionalLight(0xddeeff, 1.05);
dlight.position.set(CW / 2 + 5, 15, 5);
scene.add(dlight);
scene.add(new THREE.HemisphereLight(0x8ab4ff, 0x233823, 0.35));

(function buildCourt() {
  const gnd = new THREE.Mesh(
    new THREE.PlaneGeometry(CW + 20, CL + 20),
    new THREE.MeshLambertMaterial({color: 0x0a1a0a})
  );
  gnd.rotation.x = -Math.PI / 2;
  gnd.position.set(CW / 2, -0.01, -CL / 2);
  scene.add(gnd);

  const sur = new THREE.Mesh(
    new THREE.PlaneGeometry(CW, CL),
    new THREE.MeshLambertMaterial({color: 0x1c4720})
  );
  sur.rotation.x = -Math.PI / 2;
  sur.position.set(CW / 2, 0, -CL / 2);
  scene.add(sur);

  const grid = new THREE.GridHelper(Math.max(CW, CL) + 6, 24, 0x214a21, 0x162b16);
  grid.position.set(CW / 2, -0.005, -CL / 2);
  scene.add(grid);

  function line3(pts, color) {
    const pos = new Float32Array(pts.length * 3);
    pts.forEach(([x, y, z], i) => {
      [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]] = w2t(x, y, z);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({color})));
  }

  line3([[0,0,0],[CW,0,0],[CW,CL,0],[0,CL,0],[0,0,0]], 0xffffff);
  line3([[0, CL/2, 0], [CW, CL/2, 0]], 0xffffff);
  line3([[0,  CL/2, 0], [0,  CL/2, 2.5]], 0xffaa44);
  line3([[CW, CL/2, 0], [CW, CL/2, 2.5]], 0xffaa44);
  line3([[0, CL/2, 2.43], [CW, CL/2, 2.43]], 0xffcc66);
  scene.add(new THREE.AxesHelper(1.5));
})();

const PM = {};
for (const [pid, pd] of Object.entries(PLAYERS)) {
  const arr = f32(pd.b64);
  const col = new THREE.Color(pd.color);
  const grp = new THREE.Group();
  const jGeo = new THREE.SphereGeometry(0.05, 8, 5);
  const joints = [];
  for (let k = 0; k < K; k++) {
    const m = new THREE.Mesh(jGeo, new THREE.MeshLambertMaterial({color: col}));
    m.visible = false;
    grp.add(m);
    joints.push(m);
  }
  const bpos = new Float32Array(SKEL.length * 6);
  const bGeo = new THREE.BufferGeometry();
  const bAttr = new THREE.BufferAttribute(bpos, 3);
  bAttr.setUsage(THREE.DynamicDrawUsage);
  bGeo.setAttribute('position', bAttr);
  const bLines = new THREE.LineSegments(bGeo, new THREE.LineBasicMaterial({color: col}));
  grp.add(bLines);
  scene.add(grp);
  PM[pid] = {arr, grp, joints, bpos, bAttr, visible: true};
}

const C3DARR = f32(C3D.b64);
const C3DVALID = u8fromb64(C3D.valid_b64);
const SELARR = f32(C3D.selected_b64);
const SELVALID = u8fromb64(C3D.selected_valid_b64);
const C3DK = C3D.num_joints;
const C3DGRP = new THREE.Group();
const C3DSELGRP = new THREE.Group();
scene.add(C3DGRP);
scene.add(C3DSELGRP);

const c3dGeo = new THREE.SphereGeometry(0.07, 10, 6);
const c3dJoints = [];
for (let i = 0; i < C3DK; i++) {
  const mesh = new THREE.Mesh(c3dGeo, new THREE.MeshLambertMaterial({color: new THREE.Color(C3D.color)}));
  mesh.visible = false;
  C3DGRP.add(mesh);
  c3dJoints.push(mesh);
}
const c3dPos = new Float32Array(C3D.skeleton.length * 6);
const c3dGeoLines = new THREE.BufferGeometry();
const c3dAttr = new THREE.BufferAttribute(c3dPos, 3);
c3dAttr.setUsage(THREE.DynamicDrawUsage);
c3dGeoLines.setAttribute('position', c3dAttr);
const c3dLines = new THREE.LineSegments(c3dGeoLines, new THREE.LineBasicMaterial({color: new THREE.Color(C3D.color)}));
C3DGRP.add(c3dLines);

const selJoints = [];
for (let i = 0; i < C3DK; i++) {
  const mesh = new THREE.Mesh(c3dGeo, new THREE.MeshLambertMaterial({color: new THREE.Color(C3D.selected_color)}));
  mesh.visible = false;
  C3DSELGRP.add(mesh);
  selJoints.push(mesh);
}
const selPos = new Float32Array(C3D.skeleton.length * 6);
const selGeoLines = new THREE.BufferGeometry();
const selAttr = new THREE.BufferAttribute(selPos, 3);
selAttr.setUsage(THREE.DynamicDrawUsage);
selGeoLines.setAttribute('position', selAttr);
const selLines = new THREE.LineSegments(selGeoLines, new THREE.LineBasicMaterial({color: new THREE.Color(C3D.selected_color)}));
C3DSELGRP.add(selLines);

let confThr = 0.3;
let showC3D = true;
let showSel = true;

function drawSkeletonFromFrame(arr, t, kCount, joints, positions, skeleton, validBytes) {
  const base = t * kCount * 3;
  const wp = new Array(kCount).fill(null);
  for (let k = 0; k < kCount; k++) {
    const valid = validBytes ? !!validBytes[t * kCount + k] : true;
    if (!valid) {
      joints[k].visible = false;
      continue;
    }
    const p = w2t(arr[base + k * 3], arr[base + k * 3 + 1], arr[base + k * 3 + 2]);
    joints[k].position.set(p[0], p[1], p[2]);
    joints[k].visible = true;
    wp[k] = p;
  }
  for (let i = 0; i < skeleton.length; i++) {
    const pa = wp[skeleton[i][0]], pb = wp[skeleton[i][1]];
    if (pa && pb) {
      positions[i*6] = pa[0]; positions[i*6+1] = pa[1]; positions[i*6+2] = pa[2];
      positions[i*6+3] = pb[0]; positions[i*6+4] = pb[1]; positions[i*6+5] = pb[2];
    } else {
      positions.fill(0, i * 6, i * 6 + 6);
    }
  }
}

function render(t) {
  for (const [pid, pm] of Object.entries(PM)) {
    pm.grp.visible = pm.visible;
    if (!pm.visible) continue;

    const A = pm.arr;
    const b0 = t * K * 4;
    const wp = new Array(K).fill(null);
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
    for (let i = 0; i < SKEL.length; i++) {
      const pa = wp[SKEL[i][0]], pb = wp[SKEL[i][1]];
      if (pa && pb) {
        pm.bpos[i*6] = pa[0]; pm.bpos[i*6+1] = pa[1]; pm.bpos[i*6+2] = pa[2];
        pm.bpos[i*6+3] = pb[0]; pm.bpos[i*6+4] = pb[1]; pm.bpos[i*6+5] = pb[2];
      } else {
        pm.bpos.fill(0, i * 6, i * 6 + 6);
      }
    }
    pm.bAttr.needsUpdate = true;
  }

  C3DGRP.visible = showC3D;
  C3DSELGRP.visible = showSel;
  if (t >= 0 && t < T) {
    drawSkeletonFromFrame(C3DARR, t, C3DK, c3dJoints, c3dPos, C3D.skeleton, C3DVALID);
    c3dAttr.needsUpdate = true;
    drawSkeletonFromFrame(SELARR, t, C3DK, selJoints, selPos, C3D.skeleton, SELVALID);
    selAttr.needsUpdate = true;
  }
}

let cur = 0, playing = false, spd = 1.0, lastTs = null, accum = 0;
function setF(f) {
  cur = Math.max(0, Math.min(T - 1, Math.round(f)));
  document.getElementById('sf').value = cur;
  document.getElementById('fdisp').textContent = 'Frame ' + cur + ' / ' + (T - 1);
  render(cur);
}
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

document.getElementById('sf').max = T - 1;
document.getElementById('sf').addEventListener('input', e => setF(+e.target.value));
const bplay = document.getElementById('bplay');
bplay.addEventListener('click', () => {
  playing = !playing;
  accum = 0;
  bplay.textContent = playing ? '⏸ Pause' : '▶ Play';
  bplay.classList.toggle('on', playing);
});
document.getElementById('bprev').addEventListener('click', () => setF(cur - 1));
document.getElementById('bnext').addEventListener('click', () => setF(cur + 1));
document.getElementById('sspd').addEventListener('input', e => {
  spd = +e.target.value;
  document.getElementById('vspd').textContent = spd.toFixed(1) + '×';
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
document.getElementById('sc3djsz').addEventListener('input', e => {
  const s = +e.target.value / 0.07;
  c3dJoints.forEach(j => j.scale.setScalar(s));
  selJoints.forEach(j => j.scale.setScalar(s));
});
document.getElementById('showc3d').addEventListener('change', e => { showC3D = e.target.checked; render(cur); });
document.getElementById('showsel').addEventListener('change', e => { showSel = e.target.checked; render(cur); });

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

document.getElementById('meta').innerHTML = [
  '<div><b>Selected Player:</b> ' + C3D.selected_player_id + '</div>',
  '<div><b>Joint Count:</b> ' + C3D.num_joints + '</div>',
  '<div><b>Time Offset:</b> ' + C3D.time_offset_frames + ' frames (' + C3D.time_offset_seconds.toFixed(3) + ' s)</div>',
  '<div><b>Scale:</b> ' + C3D.scale.toFixed(6) + '</div>',
  '<div><b>RMSE After:</b> ' + C3D.rmse_after.toFixed(4) + '</div>'
].join('');

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { bplay.click(); e.preventDefault(); }
  else if (e.code === 'ArrowLeft') { setF(cur - 1); e.preventDefault(); }
  else if (e.code === 'ArrowRight') { setF(cur + 1); e.preventDefault(); }
});
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>"""


def _encode_f32(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("ascii")


def _encode_u8(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array.astype(np.uint8))
    return base64.b64encode(array.tobytes()).decode("ascii")


def _joint_subset_skeleton(joint_names):
    index = {name: i for i, name in enumerate(joint_names)}
    edges = []
    for a_name, b_name in JOINT_EDGE_NAMES:
        if a_name in index and b_name in index:
            edges.append([index[a_name], index[b_name]])
    return edges


def _build_aligned_c3d_timeline(alignment_result: dict):
    k3d_joints = np.asarray(alignment_result["k3d_joints"], dtype=np.float32)
    c3d_joints = np.asarray(alignment_result["c3d_joints_resampled"], dtype=np.float32)
    c3d_valid = np.asarray(alignment_result["c3d_valid_resampled"], dtype=bool)
    scale = float(alignment_result["scale"])
    rotation = np.asarray(alignment_result["rotation"], dtype=np.float32)
    translation = np.asarray(alignment_result["translation"], dtype=np.float32)
    offset = int(alignment_result["time_offset_frames"])

    transformed = (scale * (rotation @ c3d_joints.reshape(-1, 3).T)).T + translation
    transformed = transformed.reshape(c3d_joints.shape)

    timeline = np.full_like(k3d_joints, np.nan, dtype=np.float32)
    timeline_valid = np.zeros(k3d_joints.shape[:2], dtype=bool)
    for k3d_frame in range(len(k3d_joints)):
        c3d_frame = k3d_frame + offset
        if c3d_frame < 0 or c3d_frame >= len(transformed):
            continue
        timeline[k3d_frame] = transformed[c3d_frame]
        timeline_valid[k3d_frame] = c3d_valid[c3d_frame]
    return timeline, timeline_valid


def _build_scene_data(k3d_result: dict, alignment_report: dict, alignment_result: dict, fps: float, court_width: float, court_length: float):
    pid_sorted = sorted(k3d_result.keys())
    first = k3d_result[pid_sorted[0]]
    T, K = int(first.shape[0]), int(first.shape[1])

    players = {}
    for i, pid in enumerate(pid_sorted):
        arr = np.ascontiguousarray(k3d_result[pid], dtype=np.float32)
        players[str(pid)] = {
            "b64": _encode_f32(arr),
            "color": PLAYER_COLORS[i % len(PLAYER_COLORS)],
            "label": str(pid),
        }

    c3d_timeline, c3d_valid = _build_aligned_c3d_timeline(alignment_result)
    joint_names = alignment_report["joint_names"]

    return {
        "T": T,
        "K": K,
        "fps": float(fps),
        "skeleton": BODY25_SKELETON,
        "players": players,
        "court": {"width": float(court_width), "length": float(court_length)},
        "c3d": {
            "b64": _encode_f32(c3d_timeline),
            "valid_b64": _encode_u8(c3d_valid),
            "num_joints": len(joint_names),
            "joint_names": joint_names,
            "skeleton": _joint_subset_skeleton(joint_names),
            "color": "#ffd166",
            "selected_color": "#ff6b6b",
            "selected_player_id": alignment_report["selected_player_id"],
            "selected_b64": _encode_f32(np.asarray(alignment_result["k3d_joints"], dtype=np.float32)),
            "selected_valid_b64": _encode_u8(np.asarray(alignment_result["k3d_valid"], dtype=bool)),
            "time_offset_frames": alignment_report["time_alignment"]["c3d_to_k3d_offset_frames"],
            "time_offset_seconds": alignment_report["time_alignment"]["c3d_to_k3d_offset_seconds"],
            "scale": alignment_report["spatial_alignment"]["scale"],
            "rmse_after": alignment_report["spatial_alignment"]["rmse_after"],
        },
    }


def visualize_c3d_k3d_processor(data_path_cfg: GamePath, fps: float = None, rerun_alignment_if_missing: bool = True):
    logger = logging.getLogger("visualize_c3d_k3d_processor")

    output_html = data_path_cfg.get_output_artifact_path("c3d_k3d_vis", ".html")
    k3d_path = data_path_cfg.find_output_artifact_path("k3d", ".pkl")
    alignment_report_path = data_path_cfg.find_output_artifact_path("c3d_alignment_report", ".json")
    alignment_result_path = data_path_cfg.find_output_artifact_path("c3d_alignment", ".pkl")

    if not os.path.exists(k3d_path):
        raise FileNotFoundError(f"k3d.pkl not found: {k3d_path}")

    if (not os.path.exists(alignment_report_path) or not os.path.exists(alignment_result_path)) and rerun_alignment_if_missing:
        logger.info("Alignment artifacts missing; running c3d_alignment_processor first.")
        c3d_alignment_processor(data_path_cfg=data_path_cfg)

    if not os.path.exists(alignment_report_path) or not os.path.exists(alignment_result_path):
        raise FileNotFoundError(
            "c3d alignment artifacts not found. Please run c3d_alignment_processor first."
        )

    with open(k3d_path, "rb") as f:
        k3d_result = pickle.load(f)
    with open(alignment_report_path, "r", encoding="utf-8") as f:
        alignment_report = json.load(f)
    with open(alignment_result_path, "rb") as f:
        alignment_result = pickle.load(f)

    if not k3d_result:
        raise ValueError("k3d.pkl is empty, nothing to visualize")

    if fps is None:
        fps = data_path_cfg.fps / max(1, data_path_cfg.frame_step)

    from src.utils import constant

    scene_data = _build_scene_data(
        k3d_result=k3d_result,
        alignment_report=alignment_report,
        alignment_result=alignment_result,
        fps=fps,
        court_width=float(constant.court_width),
        court_length=float(constant.court_height),
    )

    html_content = _HTML.replace("__SCENE_DATA__", json.dumps(scene_data, separators=(",", ":")))
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Combined C3D+K3D visualization saved to %s", output_html)
