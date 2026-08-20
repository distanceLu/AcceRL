# 毛笔接触点与轨迹顶点距离检测

这套工具会从 `camera_paper_aruco` 图片序列中完成以下处理：

1. 在参考帧中识别目标空心轨迹，并标出左侧尖端。
2. 白色笔尖可见时，直接识别笔尖与纸面的接触位置。
3. 笔尖被墨水遮挡后，自动使用相对参考帧新增的黑色墨迹作为接触位置。
4. 输出接触点到轨迹尖端的直线距离、沿轨迹距离和横向偏差。
5. 保存逐帧标注图片、抽样总览图和 JSON 报告。

## 在新电脑上安装

要求 Python 3.10 或更高版本。轨迹识别只依赖 NumPy 和 OpenCV，不需要安装
项目中用于强化学习训练的完整依赖。

```bash
git clone --branch feature/brush-trace-detection \
  https://github.com/distanceLu/AcceRL.git
cd AcceRL

python3 -m venv .venv-brush-trace
source .venv-brush-trace/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-brush-trace.txt
```

把数据放在默认目录：

```text
AcceRL/data_collect/2026-08-13/<实验组>/camera_paper_aruco/*.jpg
```

然后运行：

```bash
bash tools/run_brush_contact_conversion.sh 10-10-10
```

如果数据位于仓库外部，通过环境变量指定绝对路径：

```bash
BRUSH_DATA_ROOT=/path/to/data_collect/2026-08-13 \
bash tools/run_brush_contact_conversion.sh 10-10-10
```

安装后的快速检查：

```bash
python tools/detect_hollow_trace.py --help
python tools/batch_brush_contact_distance.py --help
bash tools/run_brush_contact_conversion.sh --help
```

## 一键运行

在项目目录执行：

```bash
cd /path/to/AcceRL
bash tools/run_brush_contact_conversion.sh
```

默认处理：

- 输入：`data_collect/2026-08-13`
- 输出：`outputs/brush_contact_trials/2026-08-13`
- 首选 ROI：`250 180 1400 300`（失败时默认自动搜索垂直位置）
- 时间滤波窗口：`5`
- 自动参考搜索：开启，均匀抽查 `12` 帧
- 完全遮挡回退：开启，复用上一组成功识别的轨迹几何

只处理指定实验组：

```bash
bash tools/run_brush_contact_conversion.sh 10-10-10
```

也可以一次指定多组：

```bash
bash tools/run_brush_contact_conversion.sh 10-10-10 10-19-27 10-25-34
```

脚本已经包含 2026-08-13 数据中三个参考帧例外：

- `10-52-01`：第 2 帧
- `10-53-03`：第 2 帧
- `10-54-19`：第 3 帧
- 其他组：第 1 帧

## 输出文件

每组结果目录包含：

```text
outputs/brush_contact_trials/2026-08-13/<实验组>/
├── annotated/          # 所有识别到接触的标注帧
├── contact_sheet.jpg   # 六张抽样结果总览
└── report.json         # 逐帧坐标、距离和检测方法
```

标注颜色：

- 红色轮廓：目标空心轨迹
- 紫色点：轨迹最左侧尖端
- 黄色点：毛笔或新增墨迹的接触位置
- 黄色连线：接触点到轨迹尖端的直线距离

`report.json` 中的主要字段：

- `trace_tip_global`：轨迹尖端的全图像素坐标
- `contact_global`：经过时间中值滤波的接触点坐标
- `raw_contact_global`：当前帧未经滤波的接触点坐标
- `distance_px`：两点的欧氏距离
- `axial_distance_px`：沿轨迹长轴方向的距离
- `lateral_offset_px`：垂直轨迹长轴的偏差
- `method`：接触点检测方法，可能为 `white_nib`（独立白色笔尖）、
  `white_nib_extrapolated`（从粘连区域中提取竖直笔尖轴并外推到纸面）、
  `white_nib_guided_ink`（用外推笔尖约束附近新增墨迹）或 `new_ink`
  （没有笔尖引导时的新增墨迹）
- `distance_mm`：提供像素比例后计算的毫米距离

## Bash 超参数总表

一键脚本的参数通过环境变量覆盖，不需要修改代码。环境变量只对当前命令生效，不会永久修改脚本。

| 参数 | 默认值 | 单位/类型 | 主要作用 |
|---|---|---|---|
| `BRUSH_DATA_DATE` | `2026-08-13` | 日期字符串 | 组成默认输入/输出路径，并选择该日期已知的参考帧例外 |
| `BRUSH_DATA_ROOT` | `<项目目录>/data_collect/<日期>` | 目录 | 包含各实验组和 `camera_paper_aruco` 的输入根目录 |
| `BRUSH_OUTPUT_ROOT` | `outputs/brush_contact_trials/<日期>` | 目录 | 标注图、总览图和 JSON 报告的输出根目录 |
| `BRUSH_PYTHON_BIN` | `python` | 命令或路径 | 指定包含 OpenCV、NumPy 的 Python 解释器 |
| `BRUSH_ROI_X` | `250` | 像素，整数 | 首选 ROI 左上角横坐标，增大表示右移 |
| `BRUSH_ROI_Y` | `180` | 像素，整数 | 首选 ROI 左上角纵坐标，增大表示下移 |
| `BRUSH_ROI_WIDTH` | `1400` | 像素，正整数 | ROI 宽度 |
| `BRUSH_ROI_HEIGHT` | `300` | 像素，正整数 | ROI 高度 |
| `BRUSH_MEDIAN_WINDOW` | `5` | 接触帧数，正整数 | 接触坐标的时间中值滤波窗口 |
| `BRUSH_MM_PER_PIXEL` | 空 | mm/px，正浮点数 | 将像素距离换算为毫米；没有标定时必须留空 |
| `BRUSH_REFERENCE_FRAME` | 空 | 帧号，正整数 | 强制所有实验组使用同一个参考帧；帧号从 1 开始 |
| `BRUSH_AUTO_SEARCH` | `1` | `0` 或 `1` | 是否抽查多帧、搜索垂直 ROI 并进行轨迹几何一致性投票 |
| `BRUSH_SEARCH_SAMPLES` | `12` | 帧数，整数 `>=2` | 自动搜索时均匀抽查的帧数；越大越稳但越慢 |
| `BRUSH_TEMPLATE_FALLBACK` | `1` | `0` 或 `1` | 当前组全程遮挡时，是否复用上一成功组的轨迹几何 |
| `BRUSH_TRACE_TEMPLATE_REPORT` | 空 | JSON 文件路径 | 为单独处理遮挡组时手动提供初始成功模板 |

查看 Bash 内置说明：

```bash
bash tools/run_brush_contact_conversion.sh --help
```

## Bash 超参数详细说明

### 输入日期和目录

```bash
BRUSH_DATA_DATE=2026-08-13 \
BRUSH_DATA_ROOT=/path/to/data_collect/2026-08-13 \
BRUSH_OUTPUT_ROOT=/path/to/AcceRL/outputs/my_brush_result \
bash tools/run_brush_contact_conversion.sh
```

#### `BRUSH_DATA_DATE`

- 作用：为默认路径提供日期部分，并决定是否启用脚本中 `2026-08-13` 的参考帧例外。
- 它只是目录标签，不会读取图片 EXIF，也不会修改图片时间戳。
- 如果同时显式提供 `BRUSH_DATA_ROOT`，真正读取的位置以 `BRUSH_DATA_ROOT` 为准；但参考帧例外仍由 `BRUSH_DATA_DATE` 判断。

#### `BRUSH_DATA_ROOT`

目录必须满足以下结构：

```text
BRUSH_DATA_ROOT/
└── 10-10-10/
    └── camera_paper_aruco/
        ├── xxx.000001.jpg
        └── ...
```

- 路径不存在时脚本会立即报错。
- 不传实验组名称时，会处理该目录下所有包含 `camera_paper_aruco` 的实验组。
- 传入实验组名称时，例如 `10-10-10`，只处理指定组。

#### `BRUSH_OUTPUT_ROOT`

- 每个实验组会建立独立子目录，不会把不同组的报告混在一起。
- 再次运行某一组时，会先删除该组旧的 `annotated/frame_*.jpg`、`contact_sheet.jpg` 和 `report.json`，避免调参前后的标注帧混在一起；其他自定义文件不会删除。
- 调参对比时建议使用不同输出根目录，避免覆盖基准结果。

### ROI

2026-08-13 的画面同时包含上下两张轨迹纸，因此先使用 ROI 锁定上方轨迹：

```bash
BRUSH_ROI_X=250 \
BRUSH_ROI_Y=180 \
BRUSH_ROI_WIDTH=1400 \
BRUSH_ROI_HEIGHT=300 \
bash tools/run_brush_contact_conversion.sh
```

参数顺序为 `左上角 x、左上角 y、宽度、高度`。坐标原点是整张图左上角 `(0,0)`，x 向右增加，y 向下增加。ROI 必须完全位于图片内部。

四个参数分别表示：

- `BRUSH_ROI_X`：ROI 左边界。增大时整体向右移动；减小时向左移动。
- `BRUSH_ROI_Y`：ROI 上边界。增大时整体向下移动；减小时向上移动。
- `BRUSH_ROI_WIDTH`：从 X 起点向右保留的像素数。
- `BRUSH_ROI_HEIGHT`：从 Y 起点向下保留的像素数。

ROI 同时影响轨迹检测、白色笔尖检测和新增墨迹检测。它至少要包含完整目标轨迹及毛笔接触区域，但不应包含另一条形状相似的轨迹。默认开启自动搜索后，这四个值是“首选 ROI”：程序会保留相同横向范围，并检查首选位置及若干纵向窗口，再通过多帧一致性选择最终区域。最终实际采用的区域记录在 `report.json` 的 `roi_xywh` 中。

调整原则：

- 红框选到下方轨迹：减小 `BRUSH_ROI_HEIGHT` 或上移 `BRUSH_ROI_Y`。
- 轨迹被截断：增大宽度或高度，并相应调整起点。
- 毛笔尖在 ROI 上方被截断：减小 `BRUSH_ROI_Y`，同时适当增大高度。
- 同时框出上下两条轨迹：缩小高度，使 ROI 中只剩目标纸带。
- 相机位置改变：先用一组数据测试 ROI，再批量运行。

ROI 不是越大越好。范围过大会引入下方轨迹、黑色定位块或其他深色物体，增加误识别概率；范围过小则会截断轨迹，导致完全检测不到。

### 参考帧

`BRUSH_REFERENCE_FRAME` 是从 `1` 开始的帧号，而不是从 `0` 开始。参考帧用于两个任务：识别轨迹轮廓；作为后续新增墨迹的颜色基准。

理想参考帧应满足：

- 目标轨迹完整、清晰且没有被毛笔筒体遮挡。
- 毛笔尚未产生本次需要测量的新墨迹。
- 光照与后续帧接近，没有明显曝光跳变。
- 图片分辨率、相机位置和纸张位置与后续帧一致。

强制所有组使用同一帧：

```bash
BRUSH_REFERENCE_FRAME=1 bash tools/run_brush_contact_conversion.sh
```

如果不同实验组需要不同参考帧，请修改
`tools/run_brush_contact_conversion.sh` 中的 `reference_frame_for_run` 函数。

选择错误时的典型现象：

- 参考帧中毛笔遮住轨迹：红色轮廓缺失或框到错误对象。
- 参考帧已经含有本次墨点：`new_ink` 方法看不到新增变化，导致漏检。
- 参考帧过早且相机移动：整张纸产生差分，可能把阴影当成墨迹。

留空时，脚本会使用每组的预设规则：2026-08-13 的三个例外组使用第 2/3 帧，其余组使用第 1 帧。

### 自动参考搜索与完全遮挡回退

新数据并非所有实验组都保持相同构图。后半段纸带会向下移动，而且某些短序列从第一帧开始就被毛笔筒体或已有墨迹遮住。固定使用“第 1 帧 + `250 180 1400 300`”时，前面的组可以成功，后面的组就可能出现 `No elongated hollow trace was found`。

默认设置为：

```bash
BRUSH_AUTO_SEARCH=1 \
BRUSH_SEARCH_SAMPLES=12 \
BRUSH_TEMPLATE_FALLBACK=1 \
bash tools/run_brush_contact_conversion.sh
```

#### `BRUSH_AUTO_SEARCH`

- `1`：均匀抽查指定参考帧和其他帧，并搜索纸带可能出现的不同纵向位置；即使首选帧识别成功，也必须通过多帧一致性选择。
- `0`：严格只用指定参考帧和 ROI。适合相机完全固定且希望失败立即暴露的调试场景。
- 自动搜索先按尖端、中心、长度和角度将候选分组，再选择获得最多不同抽查帧支持的几何簇，最后取最接近该簇中位几何的轮廓。单帧旧墨迹把空腔截断、纸板边缘或黑色结构偶然获得高分时，不再能够覆盖多帧稳定结果。
- 报告中的 `reference_selection_method` 为 `configured`（关闭自动搜索）、`auto_search_consensus`（多帧一致性搜索）或以 `template:` 开头（模板回退）。

#### `BRUSH_SEARCH_SAMPLES`

- 表示自动搜索时，从整段序列近似均匀抽查多少帧，不是连续读取前 N 帧。
- 增大到 `18` 或 `24`：更有机会找到毛笔暂时移开的画面，但搜索耗时增加。
- 减小到 `6` 或 `8`：速度更快，但短暂露出的完整轨迹可能被跳过。
- 必须为 `>=2` 的整数；首选参考帧始终加入候选，不计是否恰好落在均匀采样点上。

#### `BRUSH_TEMPLATE_FALLBACK`

- `1`：如果当前组抽查后仍没有一帧显示完整轨迹，使用按时间排序的上一成功组的轨迹轮廓和尖端坐标。
- 当前组自己的参考帧仍作为新增墨迹的背景，不会把上一组的墨迹图直接当成当前背景。
- 适用于相邻实验组之间相机、纸带和轨迹位置基本不变，而当前组全程被遮挡的情况。
- 如果相机或纸带在两组之间明显移动，模板坐标也会偏移；此时应关闭模板回退，或提供同一构图下更合适的模板。
- 报告中的 `reference_selection_method` 以 `template:` 开头时，说明使用了回退模板；`trace_reference_source` 给出真正提供轨迹轮廓的图片。

#### `BRUSH_TRACE_TEMPLATE_REPORT`

批量处理时通常留空，脚本会自动记住上一成功组。若只运行一个从头到尾被遮挡的实验组，则需要显式指定同一构图下的成功报告：

```bash
BRUSH_TRACE_TEMPLATE_REPORT=outputs/brush_contact_trials/2026-08-13/10-33-27/report.json \
bash tools/run_brush_contact_conversion.sh 10-34-45
```

模板必须是新版本脚本产生且 `status` 为 `ok` 的报告，并包含 `roi_xywh` 和 `trace_reference_source`。建议优先选择时间相邻、相机未移动的一组。

如果自动搜索和模板回退都失败，脚本不会再输出 Python traceback，而会显示简洁的 `[ERROR]`/`[FAIL]`，并在该组 `report.json` 中写入 `status: "failed"` 和原因；其余实验组仍会继续处理。

### 时间中值窗口

```bash
BRUSH_MEDIAN_WINDOW=7 bash tools/run_brush_contact_conversion.sh
```

该参数必须是 `>=1` 的整数，推荐使用奇数。滤波对象是最近若干个“已经检测到接触的帧”，而不是所有原始视频帧。计算方式是分别对 x、y 坐标取中位数，然后重新计算距离。

- `1`：关闭时间平滑，输出当前帧原始坐标；响应最快、抖动最大。
- `3`：轻度平滑，适合接触点仍在移动的情况。
- `5`：默认值，当前测试集上稳定性与响应速度的折中。
- `7` 或 `9`：强平滑，适合毛笔接触后基本静止的情况；移动时会出现数帧滞后。
- 不建议使用很大的值，例如 `>15`，因为新的接触位置可能长时间受旧位置影响。

报告中的 `raw_contact_global` 是滤波前坐标，`contact_global` 是滤波后坐标。调参时可以比较两者判断是否需要增大窗口。

### 像素换算毫米

完成纸面标定并得到每像素对应的毫米数后：

```bash
BRUSH_MM_PER_PIXEL=0.12 bash tools/run_brush_contact_conversion.sh
```

换算公式为：

```text
distance_mm = distance_px × BRUSH_MM_PER_PIXEL
```

参数必须是正浮点数。此时报告会同时填写 `distance_mm`；留空时该字段为 `null`。

注意：一个固定的 `mm/px` 只适合纸面近似平行于成像平面、透视变化较小的情况。如果纸面倾斜明显，图像左侧和右侧的比例不同，应先做相机标定和纸面单应性矫正，不能仅使用单个比例。没有可靠标定值时应保持为空，避免把像素距离误当成物理距离。

### Python 环境

默认调用当前环境中的 `python`。如需指定解释器：

```bash
BRUSH_PYTHON_BIN=/path/to/venv/bin/python \
bash tools/run_brush_contact_conversion.sh
```

- 指定解释器必须能导入 `cv2` 和 `numpy`。
- 使用已经激活的虚拟环境时保持默认 `python` 即可。
- 找不到解释器时 Bash 脚本会在处理数据前退出，不会生成部分结果。

## OpenCV 检测超参数

相机曝光、纸张颜色或墨水颜色发生明显变化时，可以修改
`tools/detect_brush_contact_distance.py` 文件顶部的常量：

这些常量不能通过 Bash 环境变量覆盖，需要编辑 Python 文件。程序先寻找独立白色笔尖；如果白色笔尖与纸面在差分图中粘连，则通过纵向形态学提取笔尖轴并外推接触点，再只在该点附近匹配新增墨迹。只有笔尖引导也不可用时，才使用无引导的新增墨迹方法。这一约束用于避免把轨迹内较远处的旧墨迹当成当前笔尖。

| 常量 | 默认值 | 有效尺度 | 判定方式 |
|---|---:|---|---|
| `WHITE_MAX_SATURATION` | `40` | HSV S，`0–255` | 当前像素饱和度必须小于该值 |
| `WHITE_MIN_VALUE` | `75` | HSV V，`0–255` | 当前像素亮度必须大于该值 |
| `WHITE_MIN_LAB_DIFFERENCE` | `18.0` | OpenCV Lab 欧氏距离 | 当前帧与参考帧颜色差必须大于该值 |
| `WHITE_MAX_ALONG_SPAN_RATIO` | `3.0` | 轨迹宽度倍数 | 白色候选沿轨迹方向的最大跨度；排除笔尖与大块白纸粘连形成的假候选 |
| `INK_MIN_DARKENING` | `35` | 灰度差，`0–255` | `参考帧灰度−当前灰度` 必须大于该值 |
| `INK_MAX_GRAY` | `55` | 当前帧灰度，`0–255` | 新墨迹像素灰度必须小于该值 |

### `WHITE_MAX_SATURATION`

白色、灰色物体的 HSV 饱和度较低，黄色夹具和绿色物体的饱和度较高。

- 增大：允许偏黄、偏色的笔尖进入候选；召回率提高，但更容易把彩色高光、纸张边缘或夹具误认为笔尖。
- 减小：对白色要求更严格；误报减少，但阴影下或沾墨后的笔尖可能漏检。
- 建议调节步长：`5`；常见试验范围约 `25–70`。

### `WHITE_MIN_VALUE`

控制白色笔尖的最低亮度。

- 增大：只保留更亮的区域，能排除灰色轨迹纸，但阴影中的白色笔尖容易漏检。
- 减小：允许较暗笔尖，但可能引入灰纸、金属反光边缘或阴影。
- 建议调节步长：`10`；常见试验范围约 `50–120`。

### `WHITE_MIN_LAB_DIFFERENCE`

用于确认候选物体相对参考帧确实发生了变化，从而排除静态白纸。

- 增大：抑制相机噪声、轻微曝光变化和纸面纹理，误报减少；移动幅度小或颜色接近背景的笔尖可能漏检。
- 减小：更容易发现低对比度笔尖，但整张纸的亮度波动可能进入候选。
- 建议调节步长：`2–3`；常见试验范围约 `10–35`。

### `WHITE_MAX_ALONG_SPAN_RATIO`

正常竖直笔尖沿轨迹方向较窄；曝光变化可能把笔尖和长条白纸连接成一个横向很宽的组件，使接触点被拉到旧位置。

- 增大：允许更宽的笔尖候选，但更容易重新引入“白笔尖＋白纸”粘连误检。
- 减小：过滤更严格，可能在笔尖倾斜较大时漏检。
- 默认 `3.0` 已覆盖当前正常笔尖，同时排除了 `10-33-27` 中跨度约为轨迹宽度 `7.5` 倍的错误组件。

### `INK_MIN_DARKENING`

定义相对参考帧至少变暗多少才算“新增”。因此旧墨迹如果在参考帧中已经存在，不会被重复当作新墨迹。

- 增大：只有明显变黑的区域通过，可排除浅阴影；浅墨、水渍边缘可能漏检。
- 减小：可以识别更浅的墨迹，但毛笔阴影、曝光变化更容易误报。
- 建议调节步长：`5`；常见试验范围约 `20–70`。

### `INK_MAX_GRAY`

定义当前像素本身必须有多黑。它与 `INK_MIN_DARKENING` 同时满足时才成为墨迹候选。

- 增大：允许灰色、较浅墨迹；召回率提高，同时也更容易包含轨迹打印线和阴影。
- 减小：只接受很黑的墨迹；误报减少，但淡墨可能消失。
- 建议调节步长：`5–10`；常见试验范围约 `35–100`。

例如，要识别偏浅的墨点，可以先把 `INK_MAX_GRAY` 从 `55` 增大到 `70`；若仍漏检，再把 `INK_MIN_DARKENING` 从 `35` 减小到 `30`。不要同时大幅修改两个参数，否则难以判断是哪一个参数产生了影响。

轨迹本身的灰度候选阈值位于 `tools/detect_hollow_trace.py` 的 `detect_hollow_trace(..., thresholds=range(90, 171, 10))`：

- `90`：最严格、只包含较暗像素的起始阈值。
- `170`：较宽松的结束阈值，Python `range` 不包含 `171`，因此最后一次实际测试为 `170`。
- `10`：每次尝试的阈值步长。减小步长会增加候选次数和运行时间，但可能改善临界曝光下的检测。

通常不要先改轨迹阈值。应优先确认 ROI 和参考帧正确，因为选错轨迹或轨迹被遮挡无法靠放宽灰度阈值可靠解决。

同一文件顶部还有三个轨迹形状/背景约束：

| 常量 | 默认值 | 含义 |
|---|---:|---|
| `TRACE_MAX_WIDTH_RATIO` | `0.40` | 轨迹外轮廓宽度最多占 ROI 高度的比例；允许识别带 U 形刻度的较宽纸带 |
| `CAVITY_MIN_GRAY` | `100` | 判断轨迹空腔是否位于浅色纸带时使用的最低灰度 |
| `CAVITY_MIN_BRIGHT_FRACTION` | `0.55` | 空腔中灰度高于前一阈值的像素至少应占 55%，用于排除黑色桌面或机器人结构 |

这三个值是配套约束：只放宽轨迹宽度会使带 U 形刻度的下方轨迹可见，但也会增加黑色背景误检；浅色空腔比例检查负责排除这些背景候选。除非更换了纸张颜色或背景，建议保持默认值。

每次只调整一个阈值，并先运行一组数据：

```bash
bash tools/run_brush_contact_conversion.sh 10-10-10
```

确认 `contact_sheet.jpg` 中的紫色点和黄色点都正确后，再运行全部数据。

## 推荐调参顺序

出现问题时建议按以下顺序处理，每次只改一项：

1. **ROI**：确认红色轮廓选中当前目标轨迹且没有截断。
2. **参考帧**：确认参考帧轨迹完整、没有本次新增墨迹。
3. **中值窗口**：只有检测点正确但抖动时才修改。
4. **白色笔尖阈值**：只有 `white_nib`、`white_nib_extrapolated` 漏检或误检时修改。
5. **新增墨迹阈值**：只有 `white_nib_guided_ink`、`new_ink` 漏检或把阴影当墨迹时修改。
6. **毫米比例**：视觉坐标稳定后再做物理尺度标定。

症状与优先参数：

| 症状 | 优先检查/修改 |
|---|---|
| 红框选中下方图案 | ROI |
| 完全找不到轨迹 | ROI、参考帧，然后才考虑轨迹灰度阈值 |
| 接触前就出现黄色点 | 参考帧、`WHITE_MIN_LAB_DIFFERENCE`、`INK_MIN_DARKENING` |
| 白色笔尖清晰但无黄色点 | `WHITE_MAX_SATURATION`、`WHITE_MIN_VALUE` |
| 出墨后黄色点消失 | `INK_MAX_GRAY`、`INK_MIN_DARKENING` |
| 黄色点在正确位置附近抖动 | `BRUSH_MEDIAN_WINDOW` |
| 像素距离正确但毫米不正确 | `BRUSH_MM_PER_PIXEL` 或纸面透视标定 |

## 直接调用 Python

不使用 Bash 包装脚本时，单组命令为：

```bash
python tools/batch_brush_contact_distance.py \
  /path/to/data_collect/2026-08-13/10-10-10/camera_paper_aruco \
  --reference-frame 1 \
  --roi 250 180 1400 300 \
  --auto-search \
  --search-samples 12 \
  --median-window 5 \
  --output-dir outputs/brush_contact_trials/2026-08-13/10-10-10
```

查看所有命令行参数：

```bash
python tools/batch_brush_contact_distance.py --help
```
