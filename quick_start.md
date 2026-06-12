
## 第一步：配环境

`yiqinworkspace` 是当前的工作目录。

在工作目录创建 clone 的目标文件夹：
```bash
mkdir -p /mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test
cd /mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test
```

克隆环境：
```bash
conda create \
  --prefix /mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env \  ##需要克隆的目标目录
  --clone /mnt/data/lcx2/conda/envs/rlinf_env  ##克隆的环境所在目录
```

## 第二步：克隆代码并配置 LIBERO

从 `https://github.com/distanceLu/AcceRL/tree/lcx-26-6-9` git clone 代码到 yiqinworkspace：
```bash
git clone -b lcx-26-6-9 https://github.com/distanceLu/AcceRL.git
```

配置 LIBERO：
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
export WORKSPACE=/mnt/data/lcx1/yiqinworkspace
export CONDA_ENV=$WORKSPACE/clone_env_smoke_test/rlinf_env
export PROJECT_DIR=$WORKSPACE/AcceRL
export LIBERO_DIR=$PROJECT_DIR/LIBERO
conda activate clone_env_smoke_test/rlinf_env/
pip list  # 看 LIBERO 是否指向自己的工作目录
```

验证导入链路：
```bash
python -c "from libero.libero import benchmark; import ds_com; import rl.ds_libero_ppo_discrete as m; print('OK')"
```
成功输出 `OK` 就说明 `libero`、`ds_com`、主训练脚本导入链路都通了。

先跑通 `rl/libero_env.py` 测试 LIBERO 环境：
```bash
conda activate $CONDA_ENV
cd $PROJECT_DIR/rl
python libero_env.py
```

## 第三步：跑通actor_model_discrete.py脚本

执行 `yiqinworkspace/AcceRL/rl/actor_critic_model_discrete.py`：
```bash
cd $PROJECT_DIR/rl
python actor_critic_model_discrete.py
```

> **注意**：脚本内默认 checkpoint 指向 lcx2 路径时，需改为本机路径（见 `actor_critic_model_discrete.py` 中 `object_checkpoint`）。

## 第四步：跑通ds_libero_ppo_discrete.py脚本
执行`yiqinworkspace/AcceRL/rl/ds_libero_ppo_discrete.py`：
```bash
cd $PROJECT_DIR/rl
python ds_libero_ppo_discrete.py
```


## 问题排查记录

---

### 1. `flash_attn` / `CXXABI_1.3.15` 报错

**现象：**
```text
ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
  (required by .../flash_attn_2_cuda....so)
```

**原因：** 系统 `libstdc++` 过旧，与 `flash_attn` 编译时使用的 C++ ABI 不匹配。

**修复：**

### 安装 libstdc++（可选）
```bash
conda activate $CONDA_ENV
conda install -y -c conda-forge libstdcxx-ng
conda deactivate && conda activate $CONDA_ENV   # 加载 activate.d 里的 LD_LIBRARY_PATH
```

直接使用这个方法更简单一点：
若仍报错，可卸载 `flash-attn`（推理/评估可不依赖它，`transformers` 会回退普通 attention）：

```bash
pip uninstall flash-attn -y
```
---


### 2. `libcudart.so.13` 找不到

**现象：**

```text
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

**原因：** 已安装的 `flash_attn` 针对 CUDA 13 编译，而当前 PyTorch 为 `2.2.0+cu121`（CUDA 12.1），版本不一致。

- 推理场景：直接 `pip uninstall flash-attn -y`
- 训练需要 Flash Attention：在当前环境下重新编译安装，命令如下：

```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

---

### 3. `actor_critic_model_discrete.py` — `Floating point exception (core dumped)`

**现象：** 模型加载、10 个 LIBERO 环境初始化均成功，打印「开始第 1 轮并行执行...」后进程崩溃：

```text
Floating point exception (core dumped)
```

**说明：** 这是 Linux `SIGFPE`（浮点异常），不是 Python 异常，常见与 MuJoCo 仿真、多环境并行、或 GPU 框架冲突有关。

**带教建议「仿真禁用 GPU」：** 指 MuJoCo **渲染**不走 GPU，应使用 CPU 软件渲染：

在AcceRL/rl/actor_critic_model_discrete.py中添加以下代码可解决该问题：
```python
# 必须在 import mujoco / 创建 LIBERO 环境之前设置（文件最顶部）
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
```
仅加这两行若仍崩溃，通常不是因为 `osmesa` 没生效——若渲染失败，会在**创建环境**阶段就报错，而不是等到并行执行。

**另一个常见原因：TensorFlow 占用 GPU**

`experiments/robot/sole_utils.py` 中有 `import tensorflow`。日志里若出现 **8 条** `compute capability 9.0` 警告，说明 TensorFlow 在探测 8 张 H20，与 PyTorch（如 `cuda:2`）抢卡，进入推理循环时可能崩溃。

在导入 `sole_utils` **之前**禁用 TensorFlow GPU：

```python
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
```

---