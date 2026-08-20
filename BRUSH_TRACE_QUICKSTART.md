# Brush Trace Recognition Quick Start

本分支提供基于 OpenCV 的空心轨迹、左侧尖端、毛笔接触点和像素距离识别。

## 1. 安装

需要 Python 3.10+：

```bash
python3 -m venv .venv-brush-trace
source .venv-brush-trace/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-brush-trace.txt
```

## 2. 准备数据

默认目录结构：

```text
data_collect/2026-08-13/
└── 10-10-10/
    └── camera_paper_aruco/
        ├── frame_001.jpg
        └── ...
```

数据也可以放在任意目录，运行时设置 `BRUSH_DATA_ROOT`。

## 3. 运行

处理一个实验组：

```bash
bash tools/run_brush_contact_conversion.sh 10-10-10
```

处理数据根目录中的全部实验组：

```bash
bash tools/run_brush_contact_conversion.sh
```

使用外部数据目录：

```bash
BRUSH_DATA_ROOT=/path/to/data_collect/2026-08-13 \
bash tools/run_brush_contact_conversion.sh 10-10-10
```

输出默认位于：

```text
outputs/brush_contact_trials/2026-08-13/<实验组>/
```

其中包括逐帧标注图、`contact_sheet.jpg` 和 `report.json`。

更完整的参数、标注含义和调参说明见
[`tools/README_brush_contact_distance.md`](tools/README_brush_contact_distance.md)。
