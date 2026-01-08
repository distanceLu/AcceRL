# 裁剪指标完整指南

> **最后更新**: 2026-01-08  
> **版本**: v2.0 - 包含所有新增指标（Explained Variance, Gradient Norm, 完整分桶统计）

---

## 📋 目录

1. [指标分类体系](#指标分类体系)
2. [核心指标详解](#核心指标详解)
3. [Soft Clip vs Hard Clip 对比分析](#soft-clip-vs-hard-clip-对比分析)
4. [论文图表推荐](#论文图表推荐)
5. [指标计算细节](#指标计算细节)

---

## 指标分类体系

所有指标按照功能分为 7 大类，每类有对应的 TensorBoard/SwanLab 路径前缀：

| 类别 | 前缀 | 主要用途 |
|------|------|----------|
| **Staleness** | `Staleness/` | 数据陈旧度分析 |
| **Ratio** | `Ratio/` | 重要性采样权重分布 |
| **Hard** | `Hard/` | Hard Clip (PPO) 专用指标 |
| **Soft** | `Soft/` | Soft Clip 专用指标 |
| **Contribution** | `Contribution/` | 数据贡献度分析 |
| **ESS** | `ESS/` | 有效样本量统计 |
| **Metrics** | `Metrics/` | 训练质量与稳定性指标 |

---

## 核心指标详解

### 1. Staleness（数据陈旧度）

#### 1.1 版本差（Policy Version Gap）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `staleness_ver_mean` | `Staleness/Version_Mean` | `mean(current_version - sample_version)` | 平均策略版本差，越大样本越旧 |
| `staleness_ver_p95` | `Staleness/Version_P95` | `P95(current_version - sample_version)` | 95 分位版本差，反映最旧样本 |

#### 1.2 时间差（Age in Steps）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `age_steps_mean` | `Staleness/Age_Steps_Mean` | `mean(max_insert_step - sample_insert_step)` | 平均存储时长（步数） |
| `age_steps_p95` | `Staleness/Age_Steps_P95` | `P95(max_insert_step - sample_insert_step)` | 95 分位存储时长 |
| `age_steps_max` | `Staleness/Age_Steps_Max` | `max(max_insert_step - sample_insert_step)` | 最大存储时长 |

#### 1.3 分桶统计（绝对阈值）

**阈值定义**：`NEW_THRESHOLD = 2`, `OLD_THRESHOLD = 10`

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `staleness_new_frac_abs` | `Staleness/NewFrac_Abs` | `mean(Δv ≤ 2)` | 新数据占比（版本差 ≤ 2） |
| `staleness_old_frac_abs` | `Staleness/OldFrac_Abs` | `mean(Δv ≥ 10)` | 旧数据占比（版本差 ≥ 10） |
| `staleness_old_gap_mean_abs` | `Staleness/OldGapMean_Abs` | `mean(Δv)` for Δv ≥ 10 | 旧桶内平均版本差 |
| `staleness_old_gap_p95_abs` | `Staleness/OldGapP95_Abs` | `P95(Δv)` for Δv ≥ 10 | 旧桶内 95 分位版本差 |

#### 1.4 分桶统计（相对阈值）

**阈值定义**：新数据 `Δv / current_version ≤ 5%`，旧数据 `≥ 50%`

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `staleness_ratio_mean` | `Staleness/RatioMean` | `mean(Δv / current_version)` | 平均相对落后度 |
| `staleness_ratio_p95` | `Staleness/RatioP95` | `P95(Δv / current_version)` | 95 分位相对落后度 |
| `staleness_new_frac_ratio` | `Staleness/NewFrac_Ratio` | `mean(Δv/current_version ≤ 5%)` | 新数据占比（相对） |
| `staleness_old_frac_ratio` | `Staleness/OldFrac_Ratio` | `mean(Δv/current_version ≥ 50%)` | 旧数据占比（相对） |

---

### 2. Ratio（重要性采样权重）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `rho_mean` | `Ratio/Rho_Mean` | `mean(ρ)` | 平均 IS 权重（joint） |
| `rho_p50` | `Ratio/Rho_P50` | `median(ρ)` | 中位数 IS 权重 |
| `rho_p90` | `Ratio/Rho_P90` | `P90(ρ)` | 90 分位 IS 权重 |
| `rho_p99` | `Ratio/Rho_P99` | `P99(ρ)` | 99 分位 IS 权重 |
| `rho_max` | `Ratio/Rho_Max` | `max(ρ)` | 最大 IS 权重，反映分布尾部 |
| `logrho_mean` | `Ratio/LogRho_Mean` | `mean(log ρ)` | 对数偏移均值 |
| `abs_logrho_p95` | `Ratio/AbsLogRho_P95` | `P95(\|log ρ\|)` | **稳定性核心指标**，越小越稳定 |

**关键指标说明**：
- **`AbsLogRho_P95`**：用于论文图表，衡量训练稳定性
  - 建议值：< 0.5（非常稳定），< 1.0（稳定），> 2.0（不稳定）
  - 用途：图 D（稳定性-利用率 Pareto）的 X 轴

---

### 3. Hard Clip（PPO 专用指标）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `pg_active_frac` | `Hard/PG_Active_Frac` | `1 - dead_frac` | 有效梯度占比 |
| `pg_dead_frac` | `Hard/PG_Dead_Frac` | `mean(dead_mask)` | **死梯度占比（核心）** |
| `pg_active_frac_new` | `Hard/PG_Active_Frac_New` | 新桶有效梯度占比 | 新数据有效性 |
| `pg_active_frac_old` | `Hard/PG_Active_Frac_Old` | 旧桶有效梯度占比 | 旧数据有效性 |
| `pg_dead_frac_new` | `Hard/PG_Dead_Frac_New` | 新桶死梯度占比 | 新数据失效率 |
| `pg_dead_frac_old` | `Hard/PG_Dead_Frac_Old` | 旧桶死梯度占比 | **旧数据失效率（关键）** |
| `pg_active_frac_new_ratio` | `Hard/PG_Active_Frac_New_Ratio` | 相对阈值新桶有效占比 | - |
| `pg_active_frac_old_ratio` | `Hard/PG_Active_Frac_Old_Ratio` | 相对阈值旧桶有效占比 | - |
| `pg_dead_frac_new_ratio` | `Hard/PG_Dead_Frac_New_Ratio` | 相对阈值新桶失效率 | - |
| `pg_dead_frac_old_ratio` | `Hard/PG_Dead_Frac_Old_Ratio` | 相对阈值旧桶失效率 | - |

**Dead Mask 定义**：
```python
dead = ((ratio > 1 + ε) & (advantage > 0)) | ((ratio < 1 - ε) & (advantage < 0))
```

**期望**：
- Hard Clip：`pg_dead_frac_old` 通常 **> 0.5**（旧数据大部分失效）
- Soft Clip：无此指标，对比 `suppressed_frac_old`

---

### 4. Soft Clip（Soft Clip 专用指标）

#### 4.1 按 Ratio 定义（与 PPO 可比）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `outside_clip_frac` | `Soft/Outside_Clip_Frac` | `mean((ρ < 1-ε) \| (ρ > 1+ε))` | 落在硬阈外的比例 |
| `outside_clip_frac_new` | `Soft/Outside_Clip_Frac_New` | 新桶落在硬阈外比例 | - |
| `outside_clip_frac_old` | `Soft/Outside_Clip_Frac_Old` | 旧桶落在硬阈外比例 | - |

#### 4.2 按权重阈值定义（更贴近 Soft 机制）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `suppressed_frac` | `Soft/Suppressed_Frac` | `mean(w < 1e-3)` | **强抑制占比（核心）** |
| `suppressed_frac_new` | `Soft/Suppressed_Frac_New` | 新桶强抑制占比 | - |
| `suppressed_frac_old` | `Soft/Suppressed_Frac_Old` | 旧桶强抑制占比 | **旧数据抑制率（关键）** |

**期望**：
- Soft Clip：`suppressed_frac_old` 应 **< 0.3**（大部分旧数据仍有贡献）
- 对比 Hard Clip 的 `pg_dead_frac_old`（通常 > 0.5），Soft Clip 更温和

---

### 5. Contribution（数据贡献度）

#### 5.1 贡献权重 U（Contribution Weight）

**U 的定义**：
- **Hard Clip**: `u = ρ * (1 - dead)`
- **Soft Clip**: `u = w(ρ) * ρ`（如 `(1/max(ρ,1/ρ))^α * ρ` 或 `gate(ρ) * ρ`）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `u_mean` | `Contribution/U_Mean` | `mean(u)` | 平均有效贡献 |
| `u_p50` | `Contribution/U_P50` | `median(u)` | 中位数有效贡献 |
| `u_p90` | `Contribution/U_P90` | `P90(u)` | 90 分位有效贡献 |
| `u_p99` | `Contribution/U_P99` | `P99(u)` | 99 分位有效贡献 |
| `u_max` | `Contribution/U_Max` | `max(u)` | 最大有效贡献 |
| `u_mean_new` | `Contribution/U_Mean_New` | 新桶平均贡献 | - |
| `u_p90_new` | `Contribution/U_P90_New` | 新桶 90 分位贡献 | - |
| `u_mean_old` | `Contribution/U_Mean_Old` | 旧桶平均贡献 | **旧数据贡献强度** |
| `u_p90_old` | `Contribution/U_P90_Old` | 旧桶 90 分位贡献 | **旧数据高端贡献** |

#### 5.2 近零贡献比例（NearZero U Fraction）

**阈值**：`u < 1e-3`

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `nearzero_u_frac` | `Contribution/NearZero_U_Frac` | `mean(u < 1e-3)` | 无贡献样本占比 |
| `nearzero_u_frac_new` | `Contribution/NearZero_U_Frac_New` | 新桶无贡献占比 | - |
| `nearzero_u_frac_old` | `Contribution/NearZero_U_Frac_Old` | 旧桶无贡献占比 | **旧数据"被掐死"率（关键）** |
| `nearzero_u_frac_new_ratio` | `Contribution/NearZero_U_Frac_New_Ratio` | 相对阈值新桶无贡献占比 | - |
| `nearzero_u_frac_old_ratio` | `Contribution/NearZero_U_Frac_Old_Ratio` | 相对阈值旧桶无贡献占比 | - |

**期望**：
- Soft Clip：`nearzero_u_frac_old` 应 **< 0.2**（旧数据少被完全压死）
- Hard Clip：通常 **> 0.4**（旧数据大部分无贡献）

#### 5.3 数据贡献占比（Weight Share）

##### 5.3.1 基于 U 的占比（绝对阈值）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `contribution_old_u_share` | `Contribution/OldUShare` | `sum(u_old) / sum(u_all)` | 旧桶贡献占比（绝对） |
| `contribution_new_u_share` | `Contribution/NewUShare` | `sum(u_new) / sum(u_all)` | 新桶贡献占比（绝对） |

##### 5.3.2 基于 U 的占比（相对阈值）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `contribution_old_u_share_ratio` | `Contribution/OldUShare_Ratio` | `sum(u_old_ratio) / sum(u_all)` | 旧桶贡献占比（相对） |
| `contribution_new_u_share_ratio` | `Contribution/NewUShare_Ratio` | `sum(u_new_ratio) / sum(u_all)` | 新桶贡献占比（相对） |

##### 5.3.3 基于 |U*A| 的占比（梯度代理，绝对阈值）

**最贴近实际梯度贡献**，考虑了 Advantage 的影响。

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `contribution_old_u_share_abs_grad_proxy` | `Contribution/OldUShare_AbsGradProxy` | `sum(\|u*A\|_old) / sum(\|u*A\|_all)` | **旧桶实际梯度贡献占比** |
| `contribution_new_u_share_abs_grad_proxy` | `Contribution/NewUShare_AbsGradProxy` | `sum(\|u*A\|_new) / sum(\|u*A\|_all)` | **新桶实际梯度贡献占比** |

##### 5.3.4 基于 |U*A| 的占比（梯度代理，相对阈值）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `contribution_old_u_share_abs_grad_proxy_ratio` | `Contribution/OldUShare_AbsGradProxy_Ratio` | `sum(\|u*A\|_old_ratio) / sum(\|u*A\|_all)` | 旧桶梯度贡献占比（相对） |
| `contribution_new_u_share_abs_grad_proxy_ratio` | `Contribution/NewUShare_AbsGradProxy_Ratio` | `sum(\|u*A\|_new_ratio) / sum(\|u*A\|_all)` | 新桶梯度贡献占比（相对） |

**期望**：
- Soft Clip：`OldUShare_AbsGradProxy` 应明显 **> Hard Clip**
- 证明旧数据对策略更新有实质贡献

---

### 6. ESS（有效样本量）

#### 6.1 传统 ESS（基于 ρ）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `ess` | - | `(sum(ρ)²) / (sum(ρ²) + ε)` | 原始有效样本量 |
| `ess_norm` | - | `ess / N` | 归一化有效样本量 |

#### 6.2 有效贡献 ESS（基于 U）

**更能反映真实贡献，考虑了裁剪抑制。**

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `ess_eff` | `ESS/ESS_Eff` | `(sum(u)²) / (sum(u²) + ε)` | 有效贡献样本量 |
| `ess_eff_norm` | `ESS/ESS_Eff_Norm` | `ess_eff / N` | **归一化有效贡献量（核心）** |

#### 6.3 分桶 ESS（相对阈值）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `ess_eff_norm_new` | `ESS/ESS_Eff_Norm_New` | 新桶归一化有效贡献量 | - |
| `ess_eff_norm_old` | `ESS/ESS_Eff_Norm_Old` | 旧桶归一化有效贡献量 | **旧数据有效利用率（关键）** |

#### 6.4 分桶 ESS（绝对阈值）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `ess_eff_norm_new_abs` | `ESS/ESS_Eff_Norm_New_Abs` | 绝对阈值新桶 ESS | - |
| `ess_eff_norm_old_abs` | `ESS/ESS_Eff_Norm_Old_Abs` | 绝对阈值旧桶 ESS | - |

#### 6.5 分桶 ESS（相对阈值，另一种命名）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `ess_eff_norm_new_ratio` | `ESS/ESS_Eff_Norm_New_Ratio` | 相对阈值新桶 ESS | - |
| `ess_eff_norm_old_ratio` | `ESS/ESS_Eff_Norm_Old_Ratio` | 相对阈值旧桶 ESS | - |

**期望**：
- Soft Clip：`ESS_Eff_Norm_Old` 应明显 **> Hard Clip**
- 用于论文图表（稳定性-利用率 Pareto）的 Y 轴

---

### 7. Metrics（训练质量与稳定性）

#### 7.1 基础训练指标

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `entropy` | `Metrics/Entropy` | `mean(H(π))` | 策略熵，衡量探索度 |
| `kl_divergence` | `Metrics/KL_Divergence` | `mean(KL(π_old \|\| π_new))` | 策略变化幅度 |
| `ineffective_data_ratio` | `Metrics/Ineffective_Data_Ratio` | Clip ratio（向后兼容） | 无效数据比例 |
| `training_speed` | `Metrics/Training_Speed_Steps_per_Sec` | 全局步数/秒 | 训练吞吐量 |

#### 7.2 ✅ **新增：Explained Variance（解释方差）**

**重要性**：⭐⭐⭐⭐⭐（论文必备）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `explained_variance` | `Metrics/ExplainedVariance` | `1 - Var(target - pred) / Var(target)` | **Critic 质量评估（核心）** |

**公式**：
```python
EV = 1 - Var(V_target - V_pred) / Var(V_target)
```

**解读**：
- **EV ≈ 1**：Critic 完美拟合目标，可以信任 Value 估计
- **EV > 0.8**：Critic 拟合良好（健康状态）
- **EV ≈ 0**：Critic 基本没学到信息（警告）
- **EV < 0**：Critic 比常数基线还差（危险，可能崩坏）

**用途**：
- 排除"Critic 崩坏"导致的假信号
- 确保 Return 提升不是因为 Value 网络失效
- 论文中用于证明训练稳定性

#### 7.3 ✅ **新增：Gradient Norm（梯度范数）**

**重要性**：⭐⭐⭐⭐⭐（论文必备）

| 指标名 | TensorBoard 路径 | 计算方式 | 解读 |
|--------|-----------------|----------|------|
| `grad_norm` | `Metrics/Grad_Norm` | `\|\|∇_θ L\|\|_2` | **Policy 更新强度（核心）** |

**公式**：
```python
grad_norm = model.get_global_grad_norm()  # DeepSpeed 全局梯度范数
```

**解读**：
- **Grad_Norm < 1.0**：温和更新（通常很稳定）
- **Grad_Norm ∈ [1.0, 10.0]**：正常更新强度
- **Grad_Norm > 10.0**：激进更新（可能不稳定）
- **Grad_Norm > 100.0**：危险，可能发散

**用途**：
- **Policy Update Proxy**：衡量策略更新强度
- 证明性能提升不是靠"更猛的更新"换来的
- 配合 `AbsLogRho_P95` 使用，证明稳定性

**论文使用场景**：
1. **图表 X 轴**：使用 `AbsLogRho_P95`（稳定性）或 `Grad_Norm`（更新强度）
2. **辅助验证**：在相近 `Grad_Norm` 下，对比不同方法的性能
3. **稳定性证明**：展示 Soft Clip 在保持低 `Grad_Norm` 的同时提升性能

---

## Soft Clip vs Hard Clip 对比分析

### 核心对比指标（按重要性排序）

| # | 指标类别 | Hard Clip 指标 | Soft Clip 指标 | 期望差异 |
|---|----------|---------------|---------------|----------|
| 1 | **旧数据失效率** | `Hard/PG_Dead_Frac_Old` | `Soft/Suppressed_Frac_Old` | Hard > 0.5, Soft < 0.3 |
| 2 | **旧数据贡献** | `Contribution/U_Mean_Old` | `Contribution/U_Mean_Old` | Soft 明显 > Hard |
| 3 | **旧数据梯度贡献** | `Contribution/OldUShare_AbsGradProxy` | `Contribution/OldUShare_AbsGradProxy` | Soft 明显 > Hard |
| 4 | **旧数据有效性** | `ESS/ESS_Eff_Norm_Old` | `ESS/ESS_Eff_Norm_Old` | Soft 明显 > Hard |
| 5 | **旧数据"被掐死"率** | `Contribution/NearZero_U_Frac_Old` | `Contribution/NearZero_U_Frac_Old` | Hard > 0.4, Soft < 0.2 |
| 6 | **稳定性** | `Ratio/AbsLogRho_P95` | `Ratio/AbsLogRho_P95` | Soft ≤ Hard（无劣化） |
| 7 | **Critic 质量** | `Metrics/ExplainedVariance` | `Metrics/ExplainedVariance` | Soft ≥ Hard（无崩坏） |
| 8 | **更新强度** | `Metrics/Grad_Norm` | `Metrics/Grad_Norm` | Soft ≤ Hard（非暴力更新） |

### 论证逻辑链

#### 证明 1：Soft Clip 更好利用旧数据

**核心证据链**：
```
图 A（机制证明）: Soft/Outside_Clip_Frac_Old (X) vs Contribution/NearZero_U_Frac_Old (Y)
  → Hard Clip：X 高 + Y 高 = 旧数据大多在 clip 外且被压成零贡献
  → Soft Clip：X 可能也高，但 Y 低 = 旧数据虽在 clip 外但仍有贡献

图 B（贡献度证明）: Staleness/Version_Mean (X) vs Contribution/OldUShare_AbsGradProxy (Y)
  → 在相近陈旧度下，Soft 的 Y 更高 = 旧数据实际贡献更多

图 C（有效利用率）: ESS/ESS_Eff_Norm_Old (X) vs (1 - Contribution/NearZero_U_Frac_Old) (Y)
  → Hard：ESS 虽高但有效利用率低
  → Soft：兼顾 ESS 和有效利用率
```

#### 证明 2：Log-Gauss > SAPO（稳定性-利用率 Pareto）

**最推荐的图表**：
```
图 D（Pareto 前沿）: Ratio/AbsLogRho_P95 (X, 稳定性) vs ESS/ESS_Eff_Norm_Old (Y, 利用率)
  → Log-Gauss（尤其 σ=0.5）：更小 X + 更大 Y = 支配 SAPO
  → 结论：在 Pareto 前沿上占优
```

#### 证明 3：排除"暴力更新"假说

**辅助验证**：
```
图 E（同等稳定性对比）: Ratio/AbsLogRho_P95 (X) vs Contribution/OldUShare_AbsGradProxy (Y)
  → 同等 X 下，Soft 的 Y 更高 = 利用更多旧数据，非更激进

辅助指标: Metrics/Grad_Norm
  → Soft ≤ Hard = 不是靠更猛的梯度更新取胜
  
辅助指标: Metrics/ExplainedVariance
  → Soft ≥ Hard = 排除 Critic 崩坏导致的假信号
```

#### 证明 4：最终效果

```
图 F（最终性能）: Env_Steps (X) vs Eval/Average_Return or Average_Success_Rate (Y)
  → Soft Clip 的帕累托优势最终转化为更高 return/更快上升
```

---

## 论文图表推荐

### 主证据图表（必须包含）

| 图表 | X 轴 | Y 轴 | 用途 | 优先级 |
|------|------|------|------|--------|
| **图 D** | `Ratio/AbsLogRho_P95` | `ESS/ESS_Eff_Norm_Old` | **Pareto 前沿（最强证据）** | ⭐⭐⭐⭐⭐ |
| **图 A** | `Soft/Outside_Clip_Frac_Old` | `Contribution/NearZero_U_Frac_Old` | 机制一锤定音 | ⭐⭐⭐⭐⭐ |
| **图 B** | `Staleness/Version_Mean` | `Contribution/OldUShare_AbsGradProxy` | 旧数据真实贡献 | ⭐⭐⭐⭐ |
| **图 F** | `Env_Steps` | `Eval/Average_Return` | 最终效果 | ⭐⭐⭐⭐ |

### 辅助验证图表（推荐包含）

| 图表 | X 轴 | Y 轴 | 用途 | 优先级 |
|------|------|------|------|--------|
| **图 C** | `ESS/ESS_Eff_Norm_Old` | `1 - Contribution/NearZero_U_Frac_Old` | 有效利用率 | ⭐⭐⭐ |
| **图 E** | `Ratio/AbsLogRho_P95` | `Contribution/OldUShare_AbsGradProxy` | 同等稳定性对比 | ⭐⭐⭐ |
| **稳定性验证** | `Env_Steps` | `Metrics/Grad_Norm` | 排除暴力更新 | ⭐⭐⭐ |
| **Critic 验证** | `Env_Steps` | `Metrics/ExplainedVariance` | 排除 Critic 崩坏 | ⭐⭐⭐ |

### 图表制作建议

1. **多种子平均**：每个方法至少 3 个随机种子，显示均值 ± 标准差
2. **窗口平滑**：使用滑动窗口平滑（如 100 步），避免抖动
3. **颜色编码**：
   - Hard Clip (PPO)：红色
   - Soft Clip (α=0)：蓝色
   - SAPO：绿色
   - Log-Gauss (σ=0.5/1/2)：紫色系（深→浅）
4. **标注关键点**：在 Pareto 图上标注支配关系

---

## 指标计算细节

### 分桶阈值说明

#### 绝对阈值（Absolute Thresholds）

```python
NEW_THRESHOLD = 2    # 版本差 ≤ 2 为"新数据"
OLD_THRESHOLD = 10   # 版本差 ≥ 10 为"旧数据"
```

**优点**：
- 跨实验可比（不受 buffer 大小影响）
- 便于论文报告固定阈值

**适用场景**：
- 主要分析和论文图表
- 需要跨任务对比时

#### 相对阈值（Relative Thresholds）

```python
NEW_RATIO = 0.05     # 版本差 / 当前版本 ≤ 5% 为"新数据"
OLD_RATIO = 0.50     # 版本差 / 当前版本 ≥ 50% 为"旧数据"
```

**优点**：
- 自适应训练进度
- 捕捉相对陈旧度

**适用场景**：
- 长训练过程分析
- 需要归一化的对比

### 贡献权重 U 的计算

#### Hard Clip (PPO)

```python
# 1. 计算 dead mask
dead = ((ratio > 1 + ε) & (advantage > 0)) | 
       ((ratio < 1 - ε) & (advantage < 0))

# 2. 计算 U
u = ratio * (1 - dead.float())  # 死梯度区域 u = 0
```

#### Soft Clip (多种实现)

**Soft Clip (α=0)**:
```python
w = 1 / torch.max(ratio, 1/ratio)  # 对称抑制
u = w * ratio
```

**SAPO Soft Clip**:
```python
gate = smooth_gate(ratio, ε)  # 平滑门控
u = gate * ratio
```

**Log-Gauss Clip**:
```python
log_rho = torch.log(ratio + eps)
coeff = torch.exp(-0.5 * (log_rho / sigma) ** 2)
u = coeff * ratio
```

### Explained Variance 计算

```python
# 每个 mini-batch 计算
with torch.no_grad():
    value_pred = value.squeeze(-1) if value.dim() > 1 else value
    target = value_target  # V-trace 目标或 GAE 目标
    
    var_target = torch.var(target, unbiased=False)
    if var_target < 1e-12:
        ev = 0.0
    else:
        ev = 1.0 - torch.var(target - value_pred, unbiased=False) / (var_target + 1e-12)
    
    epoch_ev_list.append(float(ev))

# 每个 epoch 平均
explained_variance = np.mean(epoch_ev_list)
```

### Gradient Norm 计算

```python
# 在 backward 后、step 前计算
self.model.backward(loss)

# 获取全局梯度范数（DeepSpeed）
try:
    grad_norm = self.model.get_global_grad_norm()
    epoch_grad_norms.append(float(grad_norm))
except Exception:
    pass

self.model.step()

# 每个 epoch 平均
grad_norm_mean = np.mean(epoch_grad_norms)
```

### ESS 计算

#### 传统 ESS（基于 ρ）

```python
w = ratio_flat  # [B*D]
w_sum = w.sum()
w_sq_sum = (w * w).sum()
ess = (w_sum * w_sum) / (w_sq_sum + 1e-8)
ess_norm = ess / w.numel()
```

#### 有效贡献 ESS（基于 U）

```python
u_sum = u.sum()
u_sq_sum = (u * u).sum()
ess_eff = (u_sum * u_sum) / (u_sq_sum + 1e-12)
ess_eff_norm = ess_eff / u.numel()

# 分桶计算
if old_mask.any():
    u_old = u[old_mask]
    u_old_sum = u_old.sum()
    u_old_sq_sum = (u_old * u_old).sum()
    ess_eff_old = (u_old_sum * u_old_sum) / (u_old_sq_sum + 1e-12)
    ess_eff_norm_old = ess_eff_old / u_old.numel()
```

---

## 常见问题（FAQ）

### Q1: 为什么有两套分桶（绝对/相对）？

**A**: 
- **绝对阈值**：跨实验可比，适合论文报告
- **相对阈值**：自适应训练进度，适合长训练过程

建议主要使用**绝对阈值**指标（无 `_ratio` 后缀）。

### Q2: `Outside_Clip_Frac` vs `Suppressed_Frac` 的区别？

**A**:
- **Outside_Clip_Frac**: 按 ratio 定义（`ρ < 1-ε` 或 `ρ > 1+ε`），与 PPO 可比
- **Suppressed_Frac**: 按权重阈值定义（`w < 1e-3`），更贴近 Soft Clip 机制

对于 Soft Clip，**优先看 `Suppressed_Frac`**。

### Q3: 如何选择核心指标画图？

**A**: 按优先级：
1. **必须**: `AbsLogRho_P95` (X) vs `ESS_Eff_Norm_Old` (Y) - Pareto 图
2. **必须**: `PG_Dead_Frac_Old` (Hard) vs `Suppressed_Frac_Old` (Soft)
3. **必须**: `OldUShare_AbsGradProxy` - 实际梯度贡献
4. **强烈推荐**: `Grad_Norm` 和 `ExplainedVariance` - 排除假说
5. **推荐**: `NearZero_U_Frac_Old` - 补充证据

### Q4: `Explained Variance` 为负是什么情况？

**A**: 
- **EV < 0**: Critic 比常数基线还差，可能原因：
  - 强 off-policy（旧数据太多）
  - 目标噪声很大（V-trace 不稳定）
  - 训练不稳定/崩坏
- **处理方式**:
  1. 检查 `AbsLogRho_P95` 是否过大（> 2.0）
  2. 减少 replay buffer 容量或增加采样新数据比例
  3. 降低学习率或增加 value 网络容量

### Q5: `Grad_Norm` 多大算正常？

**A**:
- **< 1.0**: 温和更新，非常稳定
- **1.0 - 10.0**: 正常范围，大部分训练在此区间
- **10.0 - 100.0**: 较激进，需要监控稳定性
- **> 100.0**: 危险，可能发散

如果 `Grad_Norm` 过大且性能不稳定：
1. 降低学习率
2. 增加 Gradient Clipping 阈值
3. 减少 batch size

---

## 版本历史

- **v2.0** (2026-01-08): 
  - ✅ 新增 `Explained Variance` 和 `Gradient Norm`
  - ✅ 完善所有分桶指标说明（_New, _Old, _Ratio）
  - ✅ 新增论文图表推荐和制作建议
  - ✅ 新增常见问题解答
  
- **v1.0** (2025-12-XX): 初始版本，基础指标说明

---

## 参考文献

1. **Importance Sampling**: Precup et al. (2000) - "Eligibility Traces for Off-Policy Policy Evaluation"
2. **V-trace**: Espeholt et al. (2018) - "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
3. **Soft Clipping**: 本项目原创实现
4. **Explained Variance**: Greensmith et al. (2004) - "(In)sensitivity of Policy Gradient Methods with Respect to TD Error"

---

**文档维护**: 请在添加新指标时及时更新本文档 🚀
