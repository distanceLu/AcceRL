#!/usr/bin/env bash
set -uo pipefail

BRUSH_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BRUSH_PROJECT_ROOT=$(cd -- "$BRUSH_SCRIPT_DIR/.." && pwd)

# ============================ 数据集与输出 ================================
# BRUSH_DATA_DATE
#   数据集的日期标签，用于组成默认输入/输出路径，并选择下方已知的逐实验组
#   参考帧例外。修改该值不会读取或改动图片时间戳。
#   默认值：2026-08-13
BRUSH_DATA_DATE=${BRUSH_DATA_DATE:-2026-08-13}

# BRUSH_DATA_ROOT
#   包含 10-10-10 等实验组目录的输入根目录。每个选中的实验组必须包含
#   camera_paper_aruco/*.jpg。建议使用绝对路径。
#   默认值：/mnt/data/lcx2/yanjieworkspace/data_collect/$BRUSH_DATA_DATE
BRUSH_DATA_ROOT=${BRUSH_DATA_ROOT:-/mnt/data/lcx2/yanjieworkspace/data_collect/$BRUSH_DATA_DATE}

# BRUSH_OUTPUT_ROOT
#   标注帧、contact_sheet.jpg 和 report.json 的输出根目录。每个实验组建立
#   一个子目录。重新处理某组之前，会删除该组旧的 frame_*.jpg、
#   contact_sheet.jpg 和 report.json，避免新旧结果混合；其他文件保留。
#   默认值：<项目目录>/outputs/brush_contact_trials/$BRUSH_DATA_DATE
BRUSH_OUTPUT_ROOT=${BRUSH_OUTPUT_ROOT:-$BRUSH_PROJECT_ROOT/outputs/brush_contact_trials/$BRUSH_DATA_DATE}

# BRUSH_PYTHON_BIN
#   Python 可执行程序，所在环境必须包含 cv2 和 numpy。可以填写 PATH 中的
#   命令（python/python3），也可以填写虚拟环境 Python 的绝对路径。
#   默认值：当前 Shell 环境中的 python
BRUSH_PYTHON_BIN=${BRUSH_PYTHON_BIN:-python}

# ============================== 目标 ROI ==================================
# ROI 使用整张图片的像素坐标，左上角为原点 (0,0)，x 向右增大，y 向下增大。
# 整个矩形必须位于每一帧内部。这里配置的是首选 ROI；如果开启自动搜索且
# 首选区域识别失败，程序会继续搜索纵向移动后的纸带位置。
#
# BRUSH_ROI_X / BRUSH_ROI_Y
#   裁剪区域左上角的像素坐标。增大 X 表示右移，增大 Y 表示下移。
# BRUSH_ROI_WIDTH / BRUSH_ROI_HEIGHT
#   裁剪区域的像素宽度和高度。区域应包含完整目标轨迹及白色笔尖/接触区，
#   同时尽量排除其他相似图案。
# 默认值：x=250，y=180，宽度=1400，高度=300
BRUSH_ROI_X=${BRUSH_ROI_X:-250}
BRUSH_ROI_Y=${BRUSH_ROI_Y:-180}
BRUSH_ROI_WIDTH=${BRUSH_ROI_WIDTH:-1400}
BRUSH_ROI_HEIGHT=${BRUSH_ROI_HEIGHT:-300}

# ============================ 时间与尺度参数 ===============================
# BRUSH_MEDIAN_WINDOW
#   对最近若干个“已检测到接触的帧”的 (x,y) 坐标进行因果中值滤波，必须为
#   正整数。1 表示关闭平滑。建议使用奇数：3 响应较快，5 是默认折中，
#   7/9 抑制抖动更强，但毛笔移动时滞后更明显。它不是普通视频帧平均窗口。
BRUSH_MEDIAN_WINDOW=${BRUSH_MEDIAN_WINDOW:-5}

# BRUSH_MM_PER_PIXEL
#   可选的纸面尺度，单位为毫米/像素，必须为正数。设置后：
#   distance_mm = distance_px * BRUSH_MM_PER_PIXEL。只有完成相机/纸面标定后
#   才应填写；猜测的比例会产生误导性物理距离。
#   默认值：空，仅报告像素距离。
BRUSH_MM_PER_PIXEL=${BRUSH_MM_PER_PIXEL:-}

# BRUSH_REFERENCE_FRAME
#   可选的参考帧号，从 1 开始。设置后，所有选中的实验组都强制使用该帧。
#   参考帧应在本次接触/出墨前显示完整目标轨迹，并尽量减少毛笔遮挡。
#   留空时使用下方逐实验组规则：已知例外使用指定帧，其他组使用第 1 帧。
#   默认值：空
BRUSH_REFERENCE_FRAME=${BRUSH_REFERENCE_FRAME:-}

# ============================ 稳健参考选择 ================================
# BRUSH_AUTO_SEARCH
#   1：抽查指定帧及其他帧，并搜索纵向移动后的 ROI；按多帧重复出现的
#      轨迹几何形状进行一致性投票。这样既适应纸张/相机位置变化，也避免
#      单帧墨迹、纸板边缘或黑色物体获得高分后被误当成目标轨迹。
#   0：只使用 BRUSH_REFERENCE_FRAME 和配置的 ROI。
#   默认值：1
BRUSH_AUTO_SEARCH=${BRUSH_AUTO_SEARCH:-1}

# BRUSH_SEARCH_SAMPLES
#   自动参考搜索时，在整段序列中近似均匀抽查的帧数。数值越大，越容易找到
#   毛笔没有遮挡轨迹的时刻，但失败/参考搜索耗时也越长。必须为 >=2 的整数。
#   配置的参考帧始终会加入抽查候选，但开启自动搜索后不会绕过一致性投票。
#   默认值：12
BRUSH_SEARCH_SAMPLES=${BRUSH_SEARCH_SAMPLES:-12}

# BRUSH_TEMPLATE_FALLBACK
#   1：如果抽查的所有帧都看不到完整轨迹，复用最近一个成功实验组的轨迹
#      几何信息；当前组仍使用自己的参考帧作为新增墨迹差分背景。仅适用于
#      相机和纸张几何位置没有变化的连续实验组。
#   0：不复用以前的轨迹，直接把当前组报告为失败。
#   默认值：1
BRUSH_TEMPLATE_FALLBACK=${BRUSH_TEMPLATE_FALLBACK:-1}

# BRUSH_TRACE_TEMPLATE_REPORT
#   可选的成功 report.json 路径，用作第一个回退模板。通常保持为空：完整
#   批处理时，Bash 会自动传递前一成功组的报告。单独处理完全遮挡的实验组时
#   可以显式设置该参数。
#   默认值：空
BRUSH_TRACE_TEMPLATE_REPORT=${BRUSH_TRACE_TEMPLATE_REPORT:-}

print_help() {
    cat <<'EOF'
用法：
  bash tools/run_brush_contact_conversion.sh [RUN_NAME ...]

不提供 RUN_NAME 时，处理 BRUSH_DATA_ROOT 下所有 */camera_paper_aruco 目录；
提供实验组名称时，只处理指定实验组。

环境超参数：
  BRUSH_DATA_DATE        默认路径/例外规则使用的日期标签（默认 2026-08-13）
  BRUSH_DATA_ROOT        包含 <实验组>/camera_paper_aruco 的输入根目录
  BRUSH_OUTPUT_ROOT      生成结果的输出根目录
  BRUSH_PYTHON_BIN       Python 命令或解释器绝对路径
  BRUSH_ROI_X            ROI 左边界的整图像素坐标（默认 250）
  BRUSH_ROI_Y            ROI 上边界的整图像素坐标（默认 180）
  BRUSH_ROI_WIDTH        ROI 像素宽度（默认 1400）
  BRUSH_ROI_HEIGHT       ROI 像素高度（默认 300）
  BRUSH_MEDIAN_WINDOW    参与中值滤波的最近接触帧数，>=1（默认 5）
  BRUSH_MM_PER_PIXEL     标定后的毫米/像素比例；留空时只输出像素距离
  BRUSH_REFERENCE_FRAME  强制所有实验组使用的参考帧号，从 1 开始
  BRUSH_AUTO_SEARCH      1 表示失败后搜索其他帧/纵向 ROI；0 表示关闭
  BRUSH_SEARCH_SAMPLES   自动搜索抽查的帧数，>=2（默认 12）
  BRUSH_TEMPLATE_FALLBACK
                         1 表示完全遮挡时复用前一成功组的轨迹
  BRUSH_TRACE_TEMPLATE_REPORT
                         可选的成功 report.json，用作初始回退模板

示例：
  bash tools/run_brush_contact_conversion.sh
  bash tools/run_brush_contact_conversion.sh 10-10-10
  BRUSH_MEDIAN_WINDOW=7 bash tools/run_brush_contact_conversion.sh 10-10-10
  BRUSH_MM_PER_PIXEL=0.12 bash tools/run_brush_contact_conversion.sh
  BRUSH_TRACE_TEMPLATE_REPORT=outputs/brush_contact_trials/2026-08-13/10-33-27/report.json \
    bash tools/run_brush_contact_conversion.sh 10-34-45

调参影响和 OpenCV 阈值详见 tools/README_brush_contact_distance.md。
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

reference_frame_for_run() {
    local run_name=$1

    if [[ -n "$BRUSH_REFERENCE_FRAME" ]]; then
        echo "$BRUSH_REFERENCE_FRAME"
        return
    fi

    # 参考帧号从 1 开始，与 ROI 像素坐标不同。以下例外组已经人工验证：
    # 它们的第 1 帧无法显示有效的完整轨迹。
    if [[ "$BRUSH_DATA_DATE" == "2026-08-13" ]]; then
        case "$run_name" in
            10-52-01|10-53-03)
                echo 2
                return
                ;;
            10-54-19)
                echo 3
                return
                ;;
        esac
    fi
    echo 1
}

if ! command -v "$BRUSH_PYTHON_BIN" >/dev/null 2>&1; then
    echo "[错误] 找不到 Python 命令：$BRUSH_PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$BRUSH_SCRIPT_DIR/batch_brush_contact_distance.py" ]]; then
    echo "[错误] 检测脚本不存在：$BRUSH_SCRIPT_DIR" >&2
    exit 2
fi
if [[ ! -d "$BRUSH_DATA_ROOT" ]]; then
    echo "[错误] 输入数据根目录不存在：$BRUSH_DATA_ROOT" >&2
    exit 2
fi
if [[ "$BRUSH_AUTO_SEARCH" != "0" && "$BRUSH_AUTO_SEARCH" != "1" ]]; then
    echo "[错误] BRUSH_AUTO_SEARCH 必须为 0 或 1" >&2
    exit 2
fi
if [[ ! "$BRUSH_SEARCH_SAMPLES" =~ ^[0-9]+$ ]] || (( BRUSH_SEARCH_SAMPLES < 2 )); then
    echo "[错误] BRUSH_SEARCH_SAMPLES 必须为 >=2 的整数" >&2
    exit 2
fi
if [[ "$BRUSH_TEMPLATE_FALLBACK" != "0" && "$BRUSH_TEMPLATE_FALLBACK" != "1" ]]; then
    echo "[错误] BRUSH_TEMPLATE_FALLBACK 必须为 0 或 1" >&2
    exit 2
fi
if [[ -n "$BRUSH_TRACE_TEMPLATE_REPORT" && ! -f "$BRUSH_TRACE_TEMPLATE_REPORT" ]]; then
    echo "[错误] 模板报告不存在：$BRUSH_TRACE_TEMPLATE_REPORT" >&2
    exit 2
fi

mkdir -p "$BRUSH_OUTPUT_ROOT"
shopt -s nullglob

BRUSH_CAMERA_DIRS=()
if (( $# > 0 )); then
    for run_name in "$@"; do
        BRUSH_CAMERA_DIRS+=("$BRUSH_DATA_ROOT/$run_name/camera_paper_aruco")
    done
else
    BRUSH_CAMERA_DIRS=("$BRUSH_DATA_ROOT"/*/camera_paper_aruco)
fi

if (( ${#BRUSH_CAMERA_DIRS[@]} == 0 )); then
    echo "[错误] 在 $BRUSH_DATA_ROOT 下未找到 camera_paper_aruco 目录" >&2
    exit 2
fi

BRUSH_SUCCESS_COUNT=0
BRUSH_FAILURE_COUNT=0
BRUSH_LAST_TEMPLATE_REPORT=$BRUSH_TRACE_TEMPLATE_REPORT

echo "[信息] 输入目录=$BRUSH_DATA_ROOT"
echo "[信息] 输出目录=$BRUSH_OUTPUT_ROOT"
echo "[信息] ROI=$BRUSH_ROI_X,$BRUSH_ROI_Y,$BRUSH_ROI_WIDTH,$BRUSH_ROI_HEIGHT"
echo "[信息] 自动搜索=$BRUSH_AUTO_SEARCH 抽查帧数=$BRUSH_SEARCH_SAMPLES 模板回退=$BRUSH_TEMPLATE_FALLBACK"

for camera_dir in "${BRUSH_CAMERA_DIRS[@]}"; do
    if [[ ! -d "$camera_dir" ]]; then
        echo "[失败] 相机图片目录不存在：$camera_dir" >&2
        ((BRUSH_FAILURE_COUNT += 1))
        continue
    fi

    run_name=$(basename -- "$(dirname -- "$camera_dir")")
    reference_frame=$(reference_frame_for_run "$run_name")
    output_dir="$BRUSH_OUTPUT_ROOT/$run_name"

    BRUSH_COMMAND=(
        "$BRUSH_PYTHON_BIN"
        "$BRUSH_SCRIPT_DIR/batch_brush_contact_distance.py"
        "$camera_dir"
        --reference-frame "$reference_frame"
        --roi "$BRUSH_ROI_X" "$BRUSH_ROI_Y" "$BRUSH_ROI_WIDTH" "$BRUSH_ROI_HEIGHT"
        --median-window "$BRUSH_MEDIAN_WINDOW"
        --output-dir "$output_dir"
    )
    if [[ -n "$BRUSH_MM_PER_PIXEL" ]]; then
        BRUSH_COMMAND+=(--mm-per-pixel "$BRUSH_MM_PER_PIXEL")
    fi
    if [[ "$BRUSH_AUTO_SEARCH" == "1" ]]; then
        BRUSH_COMMAND+=(--auto-search --search-samples "$BRUSH_SEARCH_SAMPLES")
    fi
    if [[ "$BRUSH_TEMPLATE_FALLBACK" == "1" && -n "$BRUSH_LAST_TEMPLATE_REPORT" ]]; then
        BRUSH_COMMAND+=(--trace-template-report "$BRUSH_LAST_TEMPLATE_REPORT")
    fi

    echo "[运行] $run_name（参考帧 $reference_frame）"
    if "${BRUSH_COMMAND[@]}"; then
        echo "[成功] $output_dir/report.json"
        ((BRUSH_SUCCESS_COUNT += 1))
        BRUSH_LAST_TEMPLATE_REPORT=$output_dir/report.json
    else
        echo "[失败] $run_name" >&2
        ((BRUSH_FAILURE_COUNT += 1))
    fi
done

echo "[完成] 成功=$BRUSH_SUCCESS_COUNT 失败=$BRUSH_FAILURE_COUNT"
echo "[完成] 结果目录=$BRUSH_OUTPUT_ROOT"

if (( BRUSH_FAILURE_COUNT > 0 )); then
    exit 1
fi
