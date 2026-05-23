# Sport-Cap 使用文档

## 项目简介

Sport-Cap 是一套多视角体育比赛球员追踪与分析系统，适用于排球、羽毛球等运动场景（球场尺寸 28.1m × 15.1m）。系统通过多台同步摄像机拍摄的视频，完成球员检测、跟踪、重识别、三维定位、轨迹优化和运动分析，最终生成球员运动表现统计报告。

## 系统架构

整条流水线分为 6 个步骤，按顺序依次执行：

```
视频同步 → 球员检测 → 球员重识别(ReID) → 三维三角化 → 轨迹优化 → 运动分析
```

| 步骤 | 说明 | 关键模型/方法 |
|------|------|--------------|
| Step 0: 视频同步 | 基于音频互相关对齐多路视频时间线 | FFT 互相关 (44.1kHz) |
| Step 1: 球员检测 | 目标检测 + 2D 姿态估计 + 帧间跟踪 | RTMDet-M, RTMPose-M, OC-SORT |
| Step 2: 重识别(ReID) | 通过外观特征匹配为每个球员分配统一 ID | SOLIDER-ReID (Swin Transformer) |
| Step 3: 三维三角化 | 多视角 2D→3D 转换 + 3D 跟踪匹配 | EasyMocap 射线亲和, K3D-SORT |
| Step 4: 轨迹优化 | 缺失帧插值、异常检测、轨迹平滑 | 插值 + 异常点剔除 |
| Step 5: 运动分析 | 计算球员运动指标并生成统计报告 | 速度/距离/加速度/冲刺分析 |

## 环境依赖

### 核心依赖

- Python 3.8+
- PyTorch (GPU 推荐，支持 TensorRT 加速)
- OpenCV (`cv2`)
- NumPy / SciPy
- MMDetection / MMPose（RTMDet、RTMPose 模型推理）
- EasyMocap（三维三角化）
- FilterPy（卡尔曼滤波）
- LAP（匈牙利算法/线性分配）
- torchaudio / ffmpeg-python（音频同步与视频处理）
- Pillow

### 安装建议

```bash
# 新建环境（默认环境名为 sport-cap）
conda env create -f environment.yml

# 如果要同步更新当前 basket 环境，也可以直接指向现有前缀
conda env update -p /data/wangwei/conda-envs/basket -f environment.yml

conda activate sport-cap
```

仓库根目录的 `environment.yml` 基于当前使用的 `/data/wangwei/conda-envs/basket` 环境整理，移除了本机绝对路径和底层构建锁定项，保留了当前代码路径直接依赖的核心运行时包，包括 PyTorch、OpenMMLab、OpenCV、FilterPy、C3D、ImageIO、MediaPipe、`ffmpeg-python` 与 `torchaudio`。

如果你需要启用 ReID 的 reranking 路径，还需要额外安装 `faiss`；当前仓库默认流程中该分支不是必需依赖。

## 数据准备

### 目录结构

每场比赛的数据按照以下格式组织，放在 `games/` 目录下：

```
games/<workdir>/
├── videos/<game_id>/
│   ├── 00.mp4              # 视角 0 视频
│   ├── 01.mp4              # 视角 1 视频
│   ├── 02.mp4              # 视角 2
│   └── ...                 # 更多视角
├── prepare/
│   ├── camera/<game_id>/
│   │   ├── intri.yml        # 相机内参
│   │   └── extri.yml        # 相机外参
│   └── profiles/<game_id>/
│       ├── 00.json           # 视角 0 的标注文件 (VGG Image Annotator 格式)
│       └── profiles.pkl      # 球员外观特征库
```

### 相机标定文件

#### intri.yml （内参）

```yaml
names:
  - "00"
  - "01"
K_00: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_00: !!opencv-matrix
  rows: 1
  cols: 5
  dt: d
  data: [k1, k2, p1, p2, k3]
# ... 其他视角同理
```

#### extri.yml （外参）

```yaml
R_00: !!opencv-matrix
  rows: 3
  cols: 1
  dt: d
  data: [rx, ry, rz]           # Rodrigues 旋转向量
T_00: !!opencv-matrix
  rows: 3
  cols: 1
  dt: d
  data: [tx, ty, tz]           # 平移向量 (米)
# ... 其他视角同理
```

### 球员特征库 (profiles.pkl)

`profiles.pkl` 是一个 Python pickle 文件，包含每个球员的外观特征用于重识别匹配。键名格式为 `{team}#{number}`，例如 `china#1`。

可以使用工具脚本从标注视频中提取：

```bash
python tools/extract_annotated_profiles.py <game_name> --workdir games/<workdir>
```

标注文件为 VGG Image Annotator (VIA) 导出的 JSON 格式。

## 运行

### 基本命令

```bash
python scripts/run.py \
  --workdir games/sus-e07 \
  --game_id 0719-chn-jpn \
  --view_list 00 01 02 03 \
  --frame_step 1 \
  --time_align \
  --vis_video
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--workdir` | str | 必填 | 比赛数据工作目录 |
| `--game_id` | str | 必填 | 比赛标识符 |
| `--view_list` | str[] | 必填 | 视角 ID 列表（空格分隔） |
| `--video_type` | str | `.mp4` | 视频文件扩展名 |
| `--frame_step` | int | `1` | 帧采样步长（1=每帧, 5=每5帧取一帧） |
| `--time_align` | flag | `False` | 启用基于音频的多路视频时间同步 |
| `--max_frame_num` | int | `-1` | 最大处理帧数（-1=处理全部） |
| `--vis_video` | flag | `False` | 生成可视化视频 |

### 运行示例

**快速测试**（前 500 帧，每 5 帧取一帧）：

```bash
python scripts/run.py \
  --workdir games/sus-e07 \
  --game_id 0719-chn-jpn \
  --view_list 00 01 02 03 \
  --frame_step 5 \
  --max_frame_num 500
```

**完整处理**（所有帧 + 时间对齐 + 可视化）：

```bash
python scripts/run.py \
  --workdir games/sus-e07 \
  --game_id 0719-chn-jpn \
  --view_list 00 01 02 03 \
  --time_align \
  --vis_video
```

## 输出说明

### 输出目录结构

```
games/<workdir>/
├── prediction/<game_id>/
│   ├── video_sync.json                          # 视频同步偏移量
│   ├── {view}_detection_s{step}_{frames}.pkl    # 各视角检测结果
│   ├── {view}_reid_s{step}_{frames}.pkl         # 各视角重识别结果
│   ├── 3d_{frames}_v{views}-raw.pkl             # 原始三维三角化结果
│   ├── 3d_{frames}_v{views}.pkl                 # 匹配后的三维轨迹
│   └── 3d_{frames}_v{views}-opt.pkl             # 优化后的三维轨迹
├── vis/<game_id>/
│   ├── detection_{view}_s{step}_{frames}.mp4    # 检测可视化视频
│   └── 3d_{frames}_v{views}-opt.mp4             # 三维轨迹可视化视频
└── output/<game_id>/
    └── report.json                               # 运动分析报告
```

### report.json 格式

最终输出的统计报告包含每个球员的运动数据：

```json
{
  "player_data": {
    "china#1": {
      "locations": [[x, y], [x, y], ...],
      "total_distance": 125.4,
      "sprint_times": 18,
      "distance_speed_level": [45.2, 60.1, 20.1],
      "total_load": 156.3,
      "load_level": 10.2,
      "speed": [0.5, 1.2, 2.1, ...],
      "acceleration": [0.1, 0.3, ...]
    }
  }
}
```

**指标说明：**

| 字段 | 说明 |
|------|------|
| `locations` | 球员在球场平面上的位置序列 (单位: 米) |
| `total_distance` | 总跑动距离 (米) |
| `sprint_times` | 冲刺次数 (速度 > 4 m/s 的次数) |
| `distance_speed_level` | 不同速度区间的累计距离 |
| `total_load` | 基于加速度的总体力负荷 |
| `load_level` | 体力负荷等级 |
| `speed` | 逐帧速度序列 (m/s) |
| `acceleration` | 逐帧加速度序列 |

**速度区间划分：**

| 区间 | 速度范围 |
|------|---------|
| 站立 | 0 - 0.3 m/s |
| 走路 | 0.3 - 2 m/s |
| 慢跑 | 2 - 4 m/s |
| 冲刺 | > 4 m/s |

### video_sync.json 格式

```json
{
  "00": {"frame": 0, "time": 0.0},
  "01": {"frame": 5, "time": 0.2},
  "02": {"frame": -3, "time": -0.12}
}
```

每个视角的 `frame` 表示相对于参考视角的帧偏移量，`time` 为对应的时间偏移（秒）。

## 核心模块说明

```
src/
├── data_io/             # 数据输入/输出
│   ├── path_config.py   # 路径配置 (GamePath)
│   ├── loader.py        # 视频/pickle 数据加载
│   └── saver.py         # 数据保存
├── modules/             # 功能模块
│   ├── player_detector/ # 球员检测 (RTMDet + RTMPose + OC-SORT)
│   ├── player_matcher/  # 球员重识别 (SOLIDER-ReID)
│   ├── player_trackers/ # 跟踪器 (K3D-SORT, OC-SORT)
│   ├── player_triangulation/  # 三维三角化 (EasyMocap)
│   ├── player_optimizer/      # 轨迹优化 (插值 + 平滑)
│   └── video_syncer/    # 视频时间同步 (音频互相关)
├── processors/          # 流水线处理器 (串联各模块)
├── structures/          # 数据结构 (Frame, Player, MvFrame)
└── utils/               # 工具函数 (相机变换, 可视化, 关键点等)
```

## 关键参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 检测置信度阈值 | 0.2 | 人体检测的最低置信度 |
| ReID 距离阈值 | 500 | 特征距离上限，超过则不匹配 |
| 跟踪重叠阈值 | 0.5 | Bbox 重叠达到此阈值时触发 ID 重置 |
| 3D 跟踪距离 | 0.5m | 相邻帧 3D 位置最大跳变 |
| 检测 Batch 大小 | 12 | 每批同时处理的帧数 |
| 最小出现帧数 | 200 | 球员需出现超过此帧数才被确认 |
| 分析时间分辨率 | 0.2s | 运动指标的采样间隔 |

## 注意事项

1. **GPU 要求**：检测和重识别模块依赖 GPU，建议使用支持 TensorRT 加速的 NVIDIA GPU。
2. **相机标定**：三维三角化的精度高度依赖于相机标定质量，请确保内外参准确。
3. **球员特征库**：ReID 质量取决于 `profiles.pkl` 中每个球员的参考图像数量和质量，建议每人提供 10-20 张不同姿态的图像。
4. **视频同步**：使用 `--time_align` 时需要各视角的音频流，确保视频文件包含音轨。
5. **处理时间**：完整处理一场比赛可能耗时较长，可通过 `--frame_step` 和 `--max_frame_num` 进行快速测试。
