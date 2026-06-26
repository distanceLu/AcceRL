## 第一步：克隆代码并配置 LIBERO

```bash
git clone https://github.com/distanceLu/AcceRL.git
```

配置 LIBERO（与Accerl并列）：
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```
## 第二步：配环境

克隆环境：
```bash
conda create -n accerl_env --clone /mnt/data/lcx2/conda/envs/rlinf_env  ##克隆的环境所在目录
conda activate accerl_env
cd /LIBERO
pip install -e LIBERO
pip list  # 看 LIBERO 是否指向自己的工作目录
cd /AcceRL
pip install -e . --no-deps
```

验证导入链路：
```bash
python -c "from libero.libero import benchmark; import rl.ds_com; import rl.ds_libero_ppo_discrete as m; print('OK')"
```
成功输出 `OK` 就说明 `libero`、`ds_com`、主训练脚本导入链路都通了。

先跑通 `rl/libero_env.py` 测试 LIBERO 环境：
```bash
conda activate accerl_env
cd /AcceRL/rl
python libero_env.py
```

## 第三步：跑通actor_model_discrete.py脚本

执行 `/AcceRL/rl/actor_critic_model_discrete.py`：
```bash
cd /AcceRL/rl
python rl/actor_critic_model_discrete.py
```

> **注意**：脚本内默认 checkpoint 指向 lcx2 路径时，需改为本机路径（见 `actor_critic_model_discrete.py` 中 `object_checkpoint`）。

## 第四步：跑通ds_libero_ppo_discrete.py脚本
执行`/AcceRL/rl/ds_libero_ppo_discrete.py`：
```bash
cd /AcceRL/rl
python ds_libero_ppo_discrete.py
```


## 问题排查记录

---
### 1. `torch.load` — `Weights only load failed`

**现象：**

```text
Weights only load failed. Unsupported global: numpy.core.multiarray._reconstruct was not an allowed global by default.

**原因：** PyTorch 2.6+ 将 torch.load 的 weights_only 默认值由 False 改为 True，LIBERO 旧 checkpoint 含 numpy 对象，无法以默认安全模式加载。

- 在调用 torch.load 处显式传入 weights_only=False：torch.load(path, weights_only=False)
