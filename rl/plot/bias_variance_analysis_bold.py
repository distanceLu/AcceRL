import numpy as np
from typing import Dict, Tuple
import argparse
import os
import math
import matplotlib.pyplot as plt

class GridWorld2x2:
    """2x2 GridWorld环境"""
    def __init__(self, gamma=0.9):
        self.states = ['S0', 'S1', 'S2', 'SG']
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = gamma
        self.transitions = self._build_transitions()
        
    def _build_transitions(self):
        """构建确定性转移函数"""
        transitions = {}
        
        # S0: (0,0) 左上
        transitions[('S0', 'up')] = 'S0'    # 上移出界，不动
        transitions[('S0', 'down')] = 'S2'   # 下移
        transitions[('S0', 'left')] = 'S0'   # 左移出界，不动
        transitions[('S0', 'right')] = 'S1'  # 右移
        
        # S1: (0,1) 右上
        transitions[('S1', 'up')] = 'S1'    # 上移出界，不动
        transitions[('S1', 'down')] = 'SG'   # 下移
        transitions[('S1', 'left')] = 'S0'   # 左移
        transitions[('S1', 'right')] = 'S1'  # 右移出界，不动
        
        # S2: (1,0) 左下
        transitions[('S2', 'up')] = 'S0'    # 上移
        transitions[('S2', 'down')] = 'S2'   # 下移出界，不动
        transitions[('S2', 'left')] = 'S2'   # 左移出界，不动
        transitions[('S2', 'right')] = 'SG'  # 右移
        
        # SG: (1,1) 右下，吸收状态
        for a in self.actions:
            transitions[('SG', a)] = 'SG'
            
        return transitions
    
    def get_next_state(self, state, action):
        return self.transitions[(state, action)]
    
    def get_reward(self, state, action, next_state):
        if state == 'SG':
            return 0  # 终止状态奖励为0
        return -1  # 每步-1


class SoftmaxPolicy:
    """Softmax策略"""
    def __init__(self, env):
        self.env = env
        self.theta = {}  # theta[state][action]
        
    def set_theta(self, state, action_weights):
        """为状态设置参数"""
        # action_weights: 动作权重列表，按['up', 'down', 'left', 'right']顺序
        self.theta[state] = dict(zip(self.env.actions, action_weights))
        
    def get_prob(self, state, action):
        """获取策略概率π(a|s)"""
        if state == 'SG':  # 终止状态
            return 0.0 if action != 'up' else 1.0  # 任意值，不会用到
            
        weights = [self.theta.get(state, {}).get(a, 0) for a in self.env.actions]
        exp_weights = np.exp(weights)
        probs = exp_weights / np.sum(exp_weights)
        return dict(zip(self.env.actions, probs))[action]
    
    def get_all_probs(self, state):
        """获取所有动作的概率"""
        if state == 'SG':
            return {a: 0.0 for a in self.env.actions}
            
        weights = [self.theta.get(state, {}).get(a, 0) for a in self.env.actions]
        exp_weights = np.exp(weights)
        probs = exp_weights / np.sum(exp_weights)
        return dict(zip(self.env.actions, probs))
    
    def grad_log_prob(self, state, action, target_action):
        """计算∇_θ log π(a|s)，其中θ对应target_action的参数"""
        if state == 'SG':
            return 0.0
            
        probs = self.get_all_probs(state)
        if action == target_action:
            return 1 - probs[action]
        else:
            return -probs[target_action]


class UniformPolicy:
    """均匀随机策略"""
    def __init__(self, env):
        self.env = env
        
    def get_prob(self, state, action):
        if state == 'SG':
            return 0.0
        return 0.25  # 4个动作，均匀分布


def solve_value_function(policy, env):
    """求解策略的值函数V(s)"""
    states = ['S0', 'S1', 'S2', 'SG']
    
    # 构建线性方程组 A * V = b
    A = np.zeros((4, 4))
    b = np.zeros(4)
    
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    for i, s in enumerate(states):
        if s == 'SG':
            A[i, i] = 1.0
            b[i] = 0.0
            continue
            
        A[i, i] = 1.0
        for a in env.actions:
            s_next = env.get_next_state(s, a)
            j = state_to_idx[s_next]
            r = env.get_reward(s, a, s_next)
            prob = policy.get_prob(s, a)
            
            A[i, j] -= env.gamma * prob
            b[i] += prob * r
            
    # 解线性方程组
    V = np.linalg.solve(A, b)
    
    V_dict = {s: V[i] for i, s in enumerate(states)}
    return V_dict


def compute_q_values(policy, env, V_dict):
    """计算Q值"""
    Q = {}
    for s in env.states:
        Q[s] = {}
        for a in env.actions:
            s_next = env.get_next_state(s, a)
            r = env.get_reward(s, a, s_next)
            Q[s][a] = r + env.gamma * V_dict[s_next]
    return Q


def compute_advantages(Q_dict, V_dict):
    """计算优势函数"""
    A = {}
    for s in Q_dict:
        A[s] = {}
        for a in Q_dict[s]:
            A[s][a] = Q_dict[s][a] - V_dict[s]
    return A


def compute_gradient_stats(policy, behavior_policy, env, state='S0', target_action='right', sigmas=[0.1, 0.3, 0.5, 1.0, 2.0], clip_eps=0.2, tau_pos=1.0, tau_neg=2.0):
    """
    计算梯度估计量的期望和方差
    
    Args:
        policy: 目标策略
        behavior_policy: 行为策略
        env: 环境
        state: 计算的状态
        target_action: 目标参数对应的动作
        sigmas: sigma参数列表，默认[0.1, 0.3, 0.5, 1.0, 2.0]
        clip_eps: PPO裁剪参数，默认0.2
        tau_pos: SAPO正优势的tau参数，默认1.0
        tau_neg: SAPO负优势的tau参数，默认2.0
    """
    # 1. 计算目标策略的价值函数和优势函数
    V_pi = solve_value_function(policy, env)
    Q_pi = compute_q_values(policy, env, V_pi)
    A_pi = compute_advantages(Q_pi, V_pi)
    
    # 2. 计算每个动作的重要性采样比和梯度项
    actions = env.actions
    g_is_list = []  # IS估计量
    g_wis_dict = {sigma: [] for sigma in sigmas}  # WIS估计量，按sigma分组
    g_ppo_list = []  # PPO估计量
    g_sapo_list = []  # SAPO估计量
    mu_probs = []  # 行为策略概率
    
    for a in actions:
        mu_prob = behavior_policy.get_prob(state, a)
        pi_prob = policy.get_prob(state, a)
        
        if mu_prob > 0:
            rho = pi_prob / mu_prob  # 重要性采样比
            grad_log = policy.grad_log_prob(state, a, target_action)
            advantage = A_pi[state][a]
            
            # IS估计量: rho * grad_log * advantage
            g_is = rho * grad_log * advantage
            
            # WIS估计量 (多个sigma值)
            log_rho = np.log(rho + 1e-9)
            for sigma in sigmas:
                weight = np.exp(-0.5 * (log_rho / sigma) ** 2)
                g_wis = weight * g_is
                g_wis_dict[sigma].append(g_wis)
            
            # PPO估计量: min(rho * A, clip(rho, 1-eps, 1+eps) * A) * grad_log
            # 注意：在PPO中，surr1 = ratio * A, surr2 = clip(ratio, 1-eps, 1+eps) * A
            # 然后取 min(surr1, surr2)，但这里我们需要考虑grad_log
            if advantage >= 0:
                if rho <= 1.0 + clip_eps:
                    g_ppo = (rho * advantage) * grad_log
                else:
                    g_ppo = 0.0
            else:
                if rho >= 1.0 - clip_eps:
                    g_ppo = (rho * advantage) * grad_log
                else:
                    g_ppo = 0.0
            
            ratio_min = 1e-6
            ratio_max = 1e6
            r = np.clip(rho, ratio_min, ratio_max)
            tau = tau_pos if advantage > 0 else tau_neg
            
            x = tau * (r - 1.0)
            p = (1.0 / (1.0 + np.exp(-x)))
            gate = 4 * p * (1-p)
            surr_sapo = gate * advantage
            g_sapo = surr_sapo * grad_log * r
            
            g_is_list.append(g_is)
            g_ppo_list.append(g_ppo)
            g_sapo_list.append(g_sapo)
            mu_probs.append(mu_prob)
    
    mu_probs = np.array(mu_probs)
    g_is_list = np.array(g_is_list)
    g_ppo_list = np.array(g_ppo_list)
    g_sapo_list = np.array(g_sapo_list)
    
    # 3. 计算期望和方差
    # IS估计量的期望和方差
    E_g_is = np.sum(mu_probs * g_is_list)
    Var_g_is = np.sum(mu_probs * (g_is_list - E_g_is) ** 2)
    
    # PPO估计量的期望和方差
    E_g_ppo = np.sum(mu_probs * g_ppo_list)
    Var_g_ppo = np.sum(mu_probs * (g_ppo_list - E_g_ppo) ** 2)
    
    # SAPO估计量的期望和方差
    E_g_sapo = np.sum(mu_probs * g_sapo_list)
    Var_g_sapo = np.sum(mu_probs * (g_sapo_list - E_g_sapo) ** 2)
    
    # WIS估计量的期望和方差（每个sigma值）
    E_g_wis_dict = {}
    Var_g_wis_dict = {}
    g_wis_values_dict = {}
    
    for sigma in sigmas:
        g_wis_list = np.array(g_wis_dict[sigma])
        E_g_wis = np.sum(mu_probs * g_wis_list)
        Var_g_wis = np.sum(mu_probs * (g_wis_list - E_g_wis) ** 2)
        E_g_wis_dict[sigma] = E_g_wis
        Var_g_wis_dict[sigma] = Var_g_wis
        g_wis_values_dict[sigma] = dict(zip(actions, g_wis_list))
    
    result = {
        'V': V_pi,
        'A': A_pi,
        'E_g_is': E_g_is,
        'Var_g_is': Var_g_is,
        'E_g_ppo': E_g_ppo,
        'Var_g_ppo': Var_g_ppo,
        'E_g_sapo': E_g_sapo,
        'Var_g_sapo': Var_g_sapo,
        'g_is_values': dict(zip(actions, g_is_list)),
        'g_ppo_values': dict(zip(actions, g_ppo_list)),
        'g_sapo_values': dict(zip(actions, g_sapo_list)),
        'rho_values': dict(zip(actions, [policy.get_prob(state, a) / behavior_policy.get_prob(state, a) for a in actions]))
    }
    
    # 添加每个sigma值的统计信息
    for sigma in sigmas:
        result[f'E_g_wis_sigma_{sigma}'] = E_g_wis_dict[sigma]
        result[f'Var_g_wis_sigma_{sigma}'] = Var_g_wis_dict[sigma]
        result[f'g_wis_values_sigma_{sigma}'] = g_wis_values_dict[sigma]
    
    # 为了向后兼容，保留默认的E_g_wis和Var_g_wis（使用sigma=1.0）
    if 1.0 in sigmas:
        result['E_g_wis'] = E_g_wis_dict[1.0]
        result['Var_g_wis'] = Var_g_wis_dict[1.0]
        result['g_wis_values'] = g_wis_values_dict[1.0]
    
    return result


def summarize_algorithm_metrics(stats):
    """汇总各算法的偏差、方差和MSE，并返回MSE最优算法。"""
    algorithms = {
        'No-Clip': {
            'bias': 0.0,  # No-Clip是基准，偏差为0
            'variance': stats['Var_g_is'],
            'gradient_values': stats['g_is_values'],
        },
        'GIPO(σ=0.1)': {
            'bias': abs(stats['E_g_wis_sigma_0.1'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_0.1'],
            'gradient_values': stats['g_wis_values_sigma_0.1'],
        },
        'GIPO(σ=0.3)': {
            'bias': abs(stats['E_g_wis_sigma_0.3'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_0.3'],
            'gradient_values': stats['g_wis_values_sigma_0.3'],
        },
        'GIPO(σ=0.5)': {
            'bias': abs(stats['E_g_wis_sigma_0.5'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_0.5'],
            'gradient_values': stats['g_wis_values_sigma_0.5'],
        },
        'GIPO(σ=1.0)': {
            'bias': abs(stats['E_g_wis_sigma_1.0'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_1.0'],
            'gradient_values': stats['g_wis_values_sigma_1.0'],
        },
        'GIPO(σ=2.0)': {
            'bias': abs(stats['E_g_wis_sigma_2.0'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_2.0'],
            'gradient_values': stats['g_wis_values_sigma_2.0'],
        },
        'PPO': {
            'bias': abs(stats['E_g_ppo'] - stats['E_g_is']),
            'variance': stats['Var_g_ppo'],
            'gradient_values': stats['g_ppo_values'],
        },
        'SAPO': {
            'bias': abs(stats['E_g_sapo'] - stats['E_g_is']),
            'variance': stats['Var_g_sapo'],
            'gradient_values': stats['g_sapo_values'],
        }
    }

    for alg_metrics in algorithms.values():
        alg_metrics['mse'] = alg_metrics['bias'] ** 2 + alg_metrics['variance']
        gradient_values = np.array(list(alg_metrics['gradient_values'].values()), dtype=float)
        alg_metrics['excluded_from_ranking'] = np.allclose(gradient_values, 0.0, atol=1e-12)
        alg_metrics['exclusion_reason'] = (
            'zero policy gradient (fully clipped)'
            if alg_metrics['excluded_from_ranking']
            else None
        )

    eligible_algorithms = [
        name for name, metrics in algorithms.items()
        if not metrics['excluded_from_ranking']
    ]
    best_algorithm = None
    if eligible_algorithms:
        best_algorithm = min(
            eligible_algorithms,
            key=lambda name: algorithms[name]['mse']
        )
    return best_algorithm, algorithms


def collect_ratio_scatter_points(policy, behavior_policy, env):
    """收集所有非终止状态上的 (behavior_prob, policy_prob) 散点数据。"""
    points = []
    for state in ['S0', 'S1', 'S2']:
        for action in env.actions:
            behavior_prob = behavior_policy.get_prob(state, action)
            policy_prob = policy.get_prob(state, action)
            ratio = policy_prob / behavior_prob if behavior_prob > 0 else np.inf
            points.append({
                'state': state,
                'action': action,
                'behavior_prob': behavior_prob,
                'policy_prob': policy_prob,
                'ratio': ratio,
            })
    return points


def snapshot_policy_probs(policy, env):
    """导出策略在每个非终止状态上的动作概率。"""
    probs = {}
    for state in ['S0', 'S1', 'S2']:
        probs[state] = {action: float(policy.get_prob(state, action)) for action in env.actions}
    return probs


def write_mse_ranking_markdown(all_cases, output_dir):
    """将MSE排名和策略信息写入Markdown文件。"""
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "mse_ranking_summary.md")

    lines = []
    lines.append("# MSE Ranking Summary")
    lines.append("")
    lines.append("Ranking rule: `MSE = Bias^2 + Variance` (smaller is better).")
    lines.append("Algorithms with zero policy gradient (fully clipped) are excluded from ranking.")
    lines.append("")

    lines.append("## Rankings")
    lines.append("")
    skipped_cases = []
    ranked_case_count = 0

    for case_item in all_cases:
        case_name = case_item['display_name']
        best_algorithm, algorithms = summarize_algorithm_metrics(case_item['stats'])
        mse_values = np.array([algorithms[name]['mse'] for name in sorted(algorithms.keys())], dtype=float)
        if np.allclose(mse_values, mse_values[0], atol=1e-12, rtol=1e-9):
            skipped_cases.append(case_name)
            continue

        ranked_case_count += 1
        lines.append(f"### {case_name}")
        if best_algorithm is None:
            lines.append("- Best MSE: `None`")
        else:
            lines.append(f"- Best MSE: `{best_algorithm}`")

        ranking_items = [
            item for item in algorithms.items()
            if not item[1]['excluded_from_ranking']
        ]
        lines.append("")
        lines.append("| Algorithm | Bias | Variance | MSE |")
        lines.append("|---|---:|---:|---:|")
        for alg_name, metrics in sorted(ranking_items, key=lambda item: item[1]['mse']):
            lines.append(
                f"| `{alg_name}` | {metrics['bias']:.6f} | {metrics['variance']:.6f} | {metrics['mse']:.6f} |"
            )

        excluded_algorithms = [
            (name, metrics) for name, metrics in algorithms.items()
            if metrics['excluded_from_ranking']
        ]
        if excluded_algorithms:
            lines.append("")
            lines.append("Excluded from ranking:")
            for alg_name, metrics in excluded_algorithms:
                lines.append(f"- `{alg_name}`: {metrics['exclusion_reason']} (MSE={metrics['mse']:.6f})")
        lines.append("")

    if ranked_case_count == 0:
        lines.append("No case is ranked because all cases have identical MSE values across algorithms.")
        lines.append("")

    if skipped_cases:
        lines.append("Skipped cases with identical MSE for all algorithms:")
        for case_name in skipped_cases:
            lines.append(f"- `{case_name}`")
        lines.append("")

    lines.append("## Strategy Details")
    lines.append("")
    lines.append("Policy probabilities are listed as `up/down/left/right` for states `S0/S1/S2`.")
    lines.append("")

    for case_item in all_cases:
        lines.append(f"### {case_item['display_name']}")
        lines.append(f"- Target policy: `{case_item['target_name']}`")
        lines.append(f"- Target description: {case_item['target_description']}")
        lines.append(f"- Behavior case: `{case_item['case_name']}`")
        lines.append(f"- Behavior description: {case_item['case_description']}")
        if case_item['behavior_components'] is None:
            lines.append("- Behavior composition: direct policy (`mu = pi`)")
        else:
            component_str = ", ".join(
                [f"{name}:{weight:.2f}" for name, weight in case_item['behavior_components']]
            )
            lines.append(f"- Behavior composition: {component_str}")

        lines.append("")
        lines.append("Target policy probabilities:")
        lines.append("")
        lines.append("| State | Up | Down | Left | Right |")
        lines.append("|---|---:|---:|---:|---:|")
        for state in ['S0', 'S1', 'S2']:
            p = case_item['target_policy_probs'][state]
            lines.append(f"| `{state}` | {p['up']:.4f} | {p['down']:.4f} | {p['left']:.4f} | {p['right']:.4f} |")

        lines.append("")
        lines.append("Behavior policy probabilities:")
        lines.append("")
        lines.append("| State | Up | Down | Left | Right |")
        lines.append("|---|---:|---:|---:|---:|")
        for state in ['S0', 'S1', 'S2']:
            p = case_item['behavior_policy_probs'][state]
            lines.append(f"| `{state}` | {p['up']:.4f} | {p['down']:.4f} | {p['left']:.4f} | {p['right']:.4f} |")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"MSE ranking markdown saved to: {md_path}")


def create_mixed_policy(env, policy_weights):
    """
    创建混合策略
    
    Args:
        env: 环境
        policy_weights: 字典，格式为 {策略名称: (权重, 策略对象)}
    
    Returns:
        SoftmaxPolicy: 状态相关的混合softmax策略
        dict: S0状态下混合策略的概率分布（用于打印）
    """
    eps = 1e-12
    mu = SoftmaxPolicy(env)

    # 按状态混合，避免把所有状态都近似成S0上的同一分布。
    mixed_probs_by_state = {}
    for state in ['S0', 'S1', 'S2']:
        mixed_probs = {}
        for a in env.actions:
            mixed_probs[a] = sum(
                weight * policy.get_prob(state, a)
                for _, (weight, policy) in policy_weights.items()
            )
        mixed_probs_by_state[state] = mixed_probs

        ref_prob = max(mixed_probs['up'], eps)
        theta = [
            0.0,
            float(np.log(max(mixed_probs['down'], eps) / ref_prob)),
            float(np.log(max(mixed_probs['left'], eps) / ref_prob)),
            float(np.log(max(mixed_probs['right'], eps) / ref_prob)),
        ]
        mu.set_theta(state, theta)

    mixed_probs_s0 = mixed_probs_by_state['S0']
    theta_s0 = [
        mu.theta['S0']['up'],
        mu.theta['S0']['down'],
        mu.theta['S0']['left'],
        mu.theta['S0']['right'],
    ]

    return mu, mixed_probs_s0, theta_s0


def build_state_invariant_policy(env, theta):
    """在所有非终止状态上使用同一组 theta 构建策略。"""
    policy = SoftmaxPolicy(env)
    for state in ['S0', 'S1', 'S2']:
        policy.set_theta(state, theta)
    return policy


def build_state_dependent_policy(env, probs_by_state):
    """按状态指定动作概率构建策略（每个状态概率和需为1）。"""
    policy = SoftmaxPolicy(env)
    eps = 1e-12
    for state in ['S0', 'S1', 'S2']:
        state_probs = probs_by_state[state]
        prob_sum = sum(float(state_probs[a]) for a in env.actions)
        if not np.isclose(prob_sum, 1.0, atol=1e-8):
            raise ValueError(f"{state} 的动作概率和应为1，当前为 {prob_sum}")

        # 通过对数概率比将离散分布精确映射到 softmax 参数。
        ref_prob = max(float(state_probs['up']), eps)
        theta = [
            0.0,
            float(np.log(max(float(state_probs['down']), eps) / ref_prob)),
            float(np.log(max(float(state_probs['left']), eps) / ref_prob)),
            float(np.log(max(float(state_probs['right']), eps) / ref_prob)),
        ]
        policy.set_theta(state, theta)
    return policy


def print_case_report(case_name, description, mu, stats, env, policy_weights=None, mixed_probs=None, theta=None):
    """打印单个 case 的详细分析结果。"""
    print("\n" + "=" * 80)
    print(f"{case_name}: {description}")
    print("=" * 80)

    if policy_weights is None:
        print("策略说明: 使用直接指定策略 μ")
        print("策略的概率分布:")
        for a in env.actions:
            print(f"  {a}: {mu.get_prob('S0', a):.4f}")
    else:
        print("混合策略组成:")
        for name, (weight, _) in policy_weights.items():
            print(f"  {weight*100:.0f}% {name}")
        print("混合策略的概率分布:")
        for a in env.actions:
            print(f"  {a}: {mixed_probs[a]:.4f}")
        print(f"对应的theta值: {[round(x, 4) for x in theta]}")

    print(f"\nμ(上|S0): {mu.get_prob('S0', 'up'):.4f}")
    print(f"μ(下|S0): {mu.get_prob('S0', 'down'):.4f}")
    print(f"μ(左|S0): {mu.get_prob('S0', 'left'):.4f}")
    print(f"μ(右|S0): {mu.get_prob('S0', 'right'):.4f}")
    print(f"\nV(S0): {stats['V']['S0']:.4f}")
    print(f"A(S0, 右): {stats['A']['S0']['right']:.4f}")
    print(f"\n重要性采样比:")
    for a, rho in stats['rho_values'].items():
        print(f"  {a}: {rho:.4f}")

    print(f"\nIS估计量期望: {stats['E_g_is']:.6f}")
    print(f"IS估计量方差: {stats['Var_g_is']:.6f}")
    print(f"WIS估计量期望: {stats['E_g_wis']:.6f}")
    print(f"WIS估计量方差: {stats['Var_g_wis']:.6f}")
    print(f"PPO估计量期望: {stats['E_g_ppo']:.6f}")
    print(f"PPO估计量方差: {stats['Var_g_ppo']:.6f}")
    print(f"SAPO估计量期望: {stats['E_g_sapo']:.6f}")
    print(f"SAPO估计量方差: {stats['Var_g_sapo']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bias-Variance analysis plot generator."
    )
    parser.add_argument(
        "--output-dir",
        default="rollouts/grid",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--pareto-border",
        dest="pareto_border",
        action="store_true",
        help="Enable black border for highlighted best-MSE markers.",
    )
    parser.add_argument(
        "--no-pareto-border",
        dest="pareto_border",
        action="store_false",
        help="Disable black border for highlighted best-MSE markers.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save output figures as PNG and PDF.",
    )
    parser.set_defaults(pareto_border=True)
    return parser.parse_args()


def main(output_dir="rollouts/grid", pareto_border=True, save_images=False):
    env = GridWorld2x2(gamma=0.9)
    
    # 定义基础行为子策略
    random_policy = UniformPolicy(env)  # 随机策略：每个动作25%
    
    prefer_right_policy = build_state_invariant_policy(env, [0, 0, 0, 1])  # 偏好右
    prefer_down_policy = build_state_invariant_policy(env, [0, 1, 0, 0])   # 偏好下

    # 更温和、更“正常”的状态相关策略：每个状态都保留一定概率走正确方向。
    normal_policy_1 = build_state_dependent_policy(
        env,
        {
            'S0': {'up': 0.30, 'down': 0.20, 'left': 0.30, 'right': 0.20},
            'S1': {'up': 0.40, 'down': 0.15, 'left': 0.10, 'right': 0.35},
            'S2': {'up': 0.20, 'down': 0.35, 'left': 0.30, 'right': 0.15},
        },
    )
    normal_policy_2 = build_state_dependent_policy(
        env,
        {
            'S0': {'up': 0.25, 'down': 0.30, 'left': 0.20, 'right': 0.25},
            'S1': {'up': 0.20, 'down': 0.45, 'left': 0.15, 'right': 0.20},
            'S2': {'up': 0.15, 'down': 0.25, 'left': 0.15, 'right': 0.45},
        },
    )
    normal_policy_3 = build_state_dependent_policy(
        env,
        {
            'S0': {'up': 0.20, 'down': 0.35, 'left': 0.20, 'right': 0.25},
            'S1': {'up': 0.15, 'down': 0.40, 'left': 0.20, 'right': 0.25},
            'S2': {'up': 0.25, 'down': 0.15, 'left': 0.15, 'right': 0.45},
        },
    )
    normal_policy_4 = build_state_dependent_policy(
        env,
        {
            'S0': {'up': 0.32, 'down': 0.18, 'left': 0.28, 'right': 0.22},
            'S1': {'up': 0.28, 'down': 0.24, 'left': 0.28, 'right': 0.20},
            'S2': {'up': 0.24, 'down': 0.20, 'left': 0.30, 'right': 0.26},
        },
    )
    normal_policy_5 = build_state_dependent_policy(
        env,
        {
            'S0': {'up': 0.12, 'down': 0.38, 'left': 0.12, 'right': 0.38},
            'S1': {'up': 0.12, 'down': 0.52, 'left': 0.21, 'right': 0.15},
            'S2': {'up': 0.10, 'down': 0.16, 'left': 0.22, 'right': 0.52},
        },
    )

    # 多个“温和且靠近最优”的目标策略，模拟训练早/中/后阶段。
    target_policy_specs = [
        {
            'name': 'PI-1 (Mild)',
            'description': '温和偏向最优方向，仍保留较多探索',
            'policy': build_state_dependent_policy(
                env,
                {
                    'S0': {'up': 0.20, 'down': 0.30, 'left': 0.20, 'right': 0.30},
                    'S1': {'up': 0.15, 'down': 0.45, 'left': 0.25, 'right': 0.15},
                    'S2': {'up': 0.15, 'down': 0.15, 'left': 0.25, 'right': 0.45},
                },
            ),
        },
        {
            'name': 'PI-2 (Better)',
            'description': '进一步偏向最优方向（S0向右/下，S1向下，S2向右）',
            'policy': build_state_dependent_policy(
                env,
                {
                    'S0': {'up': 0.10, 'down': 0.40, 'left': 0.10, 'right': 0.40},
                    'S1': {'up': 0.10, 'down': 0.60, 'left': 0.20, 'right': 0.10},
                    'S2': {'up': 0.10, 'down': 0.10, 'left': 0.20, 'right': 0.60},
                },
            ),
        },
        {
            'name': 'PI-3 (Near-Optimal)',
            'description': '接近最优但非确定性，仍保留小概率探索',
            'policy': build_state_dependent_policy(
                env,
                {
                    'S0': {'up': 0.05, 'down': 0.45, 'left': 0.05, 'right': 0.45},
                    'S1': {'up': 0.05, 'down': 0.75, 'left': 0.10, 'right': 0.10},
                    'S2': {'up': 0.05, 'down': 0.10, 'left': 0.10, 'right': 0.75},
                },
            ),
        },
    ]

    all_cases = []
    for target_spec in target_policy_specs:
        pi = target_spec['policy']
        print("\n" + "=" * 80)
        print(f"目标策略 {target_spec['name']}: {target_spec['description']}")
        print("=" * 80)
        print(f"π(上|S0): {pi.get_prob('S0', 'up'):.4f}")
        print(f"π(下|S0): {pi.get_prob('S0', 'down'):.4f}")
        print(f"π(左|S0): {pi.get_prob('S0', 'left'):.4f}")
        print(f"π(右|S0): {pi.get_prob('S0', 'right'):.4f}")

        case_specs = [
            {
                'name': 'A (Random-100%)',
                'description': 'μ = 100%随机策略',
                'policy_weights': {
                    '随机': (1.0, random_policy),
                },
            },
            {
                'name': 'B (40%Rand+30%Right+30%Down)',
                'description': 'μ = 40%随机策略 + 30%偏好右策略 + 30%偏好下策略',
                'policy_weights': {
                    '随机': (0.4, random_policy),
                    '偏好右': (0.3, prefer_right_policy),
                    '偏好下': (0.3, prefer_down_policy),
                },
            },
            {
                'name': 'C (20%Rand+40%Right+40%Down)',
                'description': 'μ = 20%随机策略 + 40%偏好右策略 + 40%偏好下策略',
                'policy_weights': {
                    '随机': (0.2, random_policy),
                    '偏好右': (0.4, prefer_right_policy),
                    '偏好下': (0.4, prefer_down_policy),
                },
            },
            {
                'name': 'D (μ=π)',
                'description': 'μ = π（策略mu与pi一致）',
                'direct_policy': pi,
            },
            {
                'name': 'E (Mild-Mix-1)',
                'description': 'μ 为温和行为混合，结合偏保守与探索分量以形成更宽的ratio散点',
                'policy_weights': {
                    '温和1': (0.45, normal_policy_1),
                    '保守温和': (0.35, normal_policy_4),
                    '随机': (0.2, random_policy),
                },
            },
            {
                'name': 'F (Mild-Mix-2)',
                'description': 'μ 为温和行为混合，结合目标导向与探索分量以形成叶状ratio散点',
                'policy_weights': {
                    '温和2': (0.45, normal_policy_2),
                    '目标导向温和': (0.35, normal_policy_5),
                    '随机': (0.2, random_policy),
                },
            },
            {
                'name': 'G (Mild-Mix-3)',
                'description': 'μ 为多温和策略混合，同时保留绕行与朝终点动作，构造更明显的ratio张角',
                'policy_weights': {
                    '温和3': (0.4, normal_policy_3),
                    '保守温和': (0.3, normal_policy_4),
                    '目标导向温和': (0.3, normal_policy_5),
                },
            },
        ]

        for case_spec in case_specs:
            policy_weights = case_spec.get('policy_weights')
            if policy_weights is None:
                mu = case_spec['direct_policy']
                mixed_probs = None
                theta = None
            else:
                mu, mixed_probs, theta = create_mixed_policy(env, policy_weights)

            stats = compute_gradient_stats(pi, mu, env, state='S0', target_action='right')
            print_case_report(
                case_name=f"{target_spec['name']} | {case_spec['name']}",
                description=case_spec['description'],
                mu=mu,
                stats=stats,
                env=env,
                policy_weights=policy_weights,
                mixed_probs=mixed_probs,
                theta=theta,
            )
            all_cases.append({
                'target_name': target_spec['name'],
                'target_description': target_spec['description'],
                'case_name': case_spec['name'],
                'case_description': case_spec['description'],
                'display_name': f"{target_spec['name']} | {case_spec['name']}",
                'plot_name': f"{target_spec['name'].split()[0]} | {case_spec['name'].split()[0]}",
                'stats': stats,
                'ratio_points': collect_ratio_scatter_points(pi, mu, env),
                'target_policy_probs': snapshot_policy_probs(pi, env),
                'behavior_policy_probs': snapshot_policy_probs(mu, env),
                'behavior_components': (
                    None if policy_weights is None
                    else [(name, float(weight)) for name, (weight, _) in policy_weights.items()]
                ),
            })
    
    # 总结对比表格
    print("\n" + "=" * 80)
    print("总结对比 - 期望")
    print("=" * 80)
    
    # 打印期望相关的表头
    name_col_width = 56
    header_exp = f"{'情况':<{name_col_width}} {'E_g_is':<15}"
    header_exp += f" {'E_g_wis(σ=0.1)':<18} {'E_g_wis(σ=0.3)':<18} {'E_g_wis(σ=0.5)':<18} {'E_g_wis(σ=1.0)':<18} {'E_g_wis(σ=2.0)':<18}"
    header_exp += f" {'E_g_ppo':<15} {'E_g_sapo':<15}"
    print(header_exp)
    print("-" * 120)
    
    # 打印期望相关的数据
    for case_item in all_cases:
        case_name = case_item['display_name']
        stats = case_item['stats']
        row = f"{case_name:<{name_col_width}} {stats['E_g_is']:>14.6f}"
        row += f" {stats['E_g_wis_sigma_0.1']:>17.6f}"
        row += f" {stats['E_g_wis_sigma_0.3']:>17.6f}"
        row += f" {stats['E_g_wis_sigma_0.5']:>17.6f}"
        row += f" {stats['E_g_wis_sigma_1.0']:>17.6f}"
        row += f" {stats['E_g_wis_sigma_2.0']:>17.6f}"
        row += f" {stats['E_g_ppo']:>14.6f}"
        row += f" {stats['E_g_sapo']:>14.6f}"
        print(row)
    
    print("\n" + "=" * 80)
    print("总结对比 - 方差")
    print("=" * 80)
    
    # 打印方差相关的表头
    header_var = f"{'情况':<{name_col_width}} {'Var_g_is':<15}"
    header_var += f" {'Var_g_wis(σ=0.1)':<18} {'Var_g_wis(σ=0.3)':<18} {'Var_g_wis(σ=0.5)':<18} {'Var_g_wis(σ=1.0)':<18} {'Var_g_wis(σ=2.0)':<18}"
    header_var += f" {'Var_g_ppo':<15} {'Var_g_sapo':<15}"
    print(header_var)
    print("-" * 120)
    
    # 打印方差相关的数据
    for case_item in all_cases:
        case_name = case_item['display_name']
        stats = case_item['stats']
        row = f"{case_name:<{name_col_width}} {stats['Var_g_is']:>14.6f}"
        row += f" {stats['Var_g_wis_sigma_0.1']:>17.6f}"
        row += f" {stats['Var_g_wis_sigma_0.3']:>17.6f}"
        row += f" {stats['Var_g_wis_sigma_0.5']:>17.6f}"
        row += f" {stats['Var_g_wis_sigma_1.0']:>17.6f}"
        row += f" {stats['Var_g_wis_sigma_2.0']:>17.6f}"
        row += f" {stats['Var_g_ppo']:>14.6f}"
        row += f" {stats['Var_g_sapo']:>14.6f}"
        print(row)
    
    print("\n" + "=" * 80)
    print("总结对比 - 偏差绝对值（相对于IS）")
    print("=" * 80)
    
    # 打印偏差相关的表头
    header_bias = f"{'情况':<{name_col_width}} {'|E_g_wis(σ=0.1)-E_g_is|':<25} {'|E_g_wis(σ=0.3)-E_g_is|':<25} {'|E_g_wis(σ=0.5)-E_g_is|':<25} {'|E_g_wis(σ=1.0)-E_g_is|':<25} {'|E_g_wis(σ=2.0)-E_g_is|':<25}"
    header_bias += f" {'|E_g_ppo-E_g_is|':<20} {'|E_g_sapo-E_g_is|':<22}"
    print(header_bias)
    print("-" * 150)
    
    # 打印偏差相关的数据（使用绝对值）
    for case_item in all_cases:
        case_name = case_item['display_name']
        stats = case_item['stats']
        row = f"{case_name:<{name_col_width}}"
        row += f" {abs(stats['E_g_wis_sigma_0.1'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_0.3'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_0.5'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_1.0'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_2.0'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_ppo'] - stats['E_g_is']):>19.6f}"
        row += f" {abs(stats['E_g_sapo'] - stats['E_g_is']):>21.6f}"
        print(row)
    
    # MSE总结
    print("\n" + "=" * 80)
    print("总结对比 - MSE")
    print("=" * 80)

    header_mse = f"{'情况':<{name_col_width}} {'MSE(No-Clip)':<15} {'MSE(GIPO 0.1)':<15} {'MSE(GIPO 0.3)':<15} {'MSE(GIPO 0.5)':<15} {'MSE(GIPO 1.0)':<15} {'MSE(GIPO 2.0)':<15} {'MSE(PPO)':<15} {'MSE(SAPO)':<15}"
    print(header_mse)
    print("-" * 180)

    for case_item in all_cases:
        case_name = case_item['display_name']
        stats = case_item['stats']
        _, algorithms = summarize_algorithm_metrics(stats)
        row = f"{case_name:<{name_col_width}}"
        row += f" {algorithms['No-Clip']['mse']:>14.6f}"
        row += f" {algorithms['GIPO(σ=0.1)']['mse']:>14.6f}"
        row += f" {algorithms['GIPO(σ=0.3)']['mse']:>14.6f}"
        row += f" {algorithms['GIPO(σ=0.5)']['mse']:>14.6f}"
        row += f" {algorithms['GIPO(σ=1.0)']['mse']:>14.6f}"
        row += f" {algorithms['GIPO(σ=2.0)']['mse']:>14.6f}"
        row += f" {algorithms['PPO']['mse']:>14.6f}"
        row += f" {algorithms['SAPO']['mse']:>14.6f}"
        print(row)

    print("\n" + "=" * 80)
    print("MSE最优算法分析")
    print("=" * 80)
    print("说明：使用 MSE = Bias^2 + Variance 综合比较算法，越小越好。")
    print("若某算法在当前case下所有动作的策略梯度都为0（完全clip），则不参与排名，并在下方说明。\n")

    for case_item in all_cases:
        case_name = case_item['display_name']
        stats = case_item['stats']
        best_algorithm, algorithms = summarize_algorithm_metrics(stats)
        excluded_algorithms = [
            name for name, metrics in algorithms.items()
            if metrics['excluded_from_ranking']
        ]
        print(f"{case_name}:")
        if best_algorithm is None:
            print("  MSE最优算法: None")
        else:
            print(f"  MSE最优算法: {best_algorithm}")
        print(f"  算法详细指标:")
        ranking_items = [
            item for item in algorithms.items()
            if not item[1]['excluded_from_ranking']
        ]
        for alg_name, alg_metrics in sorted(ranking_items, key=lambda item: item[1]['mse']):
            is_best = "*" if best_algorithm is not None and alg_name == best_algorithm else " "
            print(
                f"    [{is_best}] {alg_name:<15} "
                f"偏差: {alg_metrics['bias']:>10.6f}, "
                f"方差: {alg_metrics['variance']:>10.6f}, "
                f"MSE: {alg_metrics['mse']:>10.6f}"
            )
        if excluded_algorithms:
            print("  不参与排名的算法:")
            for alg_name in excluded_algorithms:
                reason = algorithms[alg_name]['exclusion_reason']
                print(
                    f"    - {alg_name}: {reason}, "
                    f"MSE={algorithms[alg_name]['mse']:.6f}"
                )
        print()

    write_mse_ranking_markdown(all_cases, output_dir)

    # 绘制成对的偏差-方差图与ratio散点图
    print("\n" + "=" * 80)
    print("绘制 Bias-Variance / Ratio Scatter 配对图...")
    print("=" * 80)

    plot_cases = all_cases
    pairs_per_row = 2
    ncols = pairs_per_row * 2
    nrows = math.ceil(len(plot_cases) / pairs_per_row)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows))
    axes = np.atleast_2d(axes)

    algorithm_styles = {
        'No-Clip':      {'color': '#d62728', 'marker': 'X', 'size': 100},
        'GIPO(σ=0.1)':  {'color': '#ff7f0e', 'marker': 'o', 'size': 80},
        'GIPO(σ=0.3)':  {'color': '#ff7f0e', 'marker': 'P', 'size': 80},
        'GIPO(σ=0.5)':  {'color': '#ff7f0e', 'marker': 's', 'size': 80},
        'GIPO(σ=1.0)':  {'color': '#ff7f0e', 'marker': '^', 'size': 80},
        'GIPO(σ=2.0)':  {'color': '#ff7f0e', 'marker': 'v', 'size': 80},
        'PPO':          {'color': '#1f77b4', 'marker': 'D', 'size': 80},
        'SAPO':         {'color': '#2ca02c', 'marker': 'p', 'size': 80}
    }
    ratio_state_styles = {
        'S0': '#1f77b4',
        'S1': '#ff7f0e',
        'S2': '#2ca02c',
    }

    plot_data = []
    global_biases = []
    global_variances = []
    global_prob_values = []

    for case_item in plot_cases:
        plot_name = case_item['plot_name']
        stats = case_item['stats']
        ratio_points = case_item['ratio_points']
        best_algorithm, algorithms = summarize_algorithm_metrics(stats)
        plot_data.append((plot_name, best_algorithm, algorithms, ratio_points))
        global_biases.extend([alg['bias'] for alg in algorithms.values()])
        global_variances.extend([alg['variance'] for alg in algorithms.values()])
        for point in ratio_points:
            global_prob_values.extend([point['behavior_prob'], point['policy_prob']])

    global_bias_min = min(global_biases)
    global_bias_max = max(global_biases)
    global_var_min = min(global_variances)
    global_var_max = max(global_variances)
    global_bias_range = global_bias_max - global_bias_min
    global_var_range = global_var_max - global_var_min
    bias_margin = global_bias_range * 0.1 if global_bias_range > 0 else 1e-3
    var_margin = global_var_range * 0.1 if global_var_range > 0 else 1e-3
    global_xlim = (global_bias_min - bias_margin, global_bias_max + bias_margin)
    global_ylim = (global_var_min - var_margin, global_var_max + var_margin)

    prob_max = max(global_prob_values) if global_prob_values else 1.0
    prob_margin = max(0.02, prob_max * 0.05)
    prob_limit = (0.0, min(1.0, prob_max + prob_margin))

    for idx, (plot_name, best_algorithm, algorithms, ratio_points) in enumerate(plot_data):
        row = idx // pairs_per_row
        pair_col = (idx % pairs_per_row) * 2
        ax_bv = axes[row, pair_col]
        ax_ratio = axes[row, pair_col + 1]

        sorted_alg_names = sorted(
            algorithms.keys(),
            key=lambda name: (
                0 if best_algorithm is not None and name == best_algorithm else 1,
                -algorithm_styles[name]['size'],
                name,
            )
        )
        for alg_name in sorted_alg_names:
            alg_metrics = algorithms[alg_name]
            style = algorithm_styles[alg_name]
            is_best = best_algorithm is not None and alg_name == best_algorithm
            is_excluded = alg_metrics['excluded_from_ranking']

            scatter_kwargs = {
                'c': style['color'],
                'marker': style['marker'],
                's': style['size'],
                'alpha': 0.35 if is_excluded else 1.0,
                'label': alg_name,
                'zorder': 3 if is_best else 2,
            }
            if is_best and pareto_border:
                scatter_kwargs['edgecolors'] = 'black'
                scatter_kwargs['linewidths'] = 2
            ax_bv.scatter(alg_metrics['bias'], alg_metrics['variance'], **scatter_kwargs)

        summary_lines = []
        if best_algorithm is None:
            summary_lines.append("Best MSE: None")
        else:
            summary_lines.append(f"Best MSE: {best_algorithm}")
        excluded_names = [name for name, metrics in algorithms.items() if metrics['excluded_from_ranking']]
        if excluded_names:
            summary_lines.append(f"Excluded: {', '.join(excluded_names)}")
        ax_bv.text(
            0.03, 0.97,
            "\n".join(summary_lines),
            transform=ax_bv.transAxes,
            va='top',
            fontsize=9,
            bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
        )
        ax_bv.set_xlabel('Bias', fontsize=12, fontweight='bold')
        ax_bv.set_ylabel('Variance', fontsize=12, fontweight='bold')
        ax_bv.set_title(f"{plot_name} | Bias-Variance", fontsize=11, fontweight='bold')
        ax_bv.tick_params(axis='both', which='major', labelsize=10)
        ax_bv.grid(True, alpha=0.3, linestyle='--')
        ax_bv.set_xlim(*global_xlim)
        ax_bv.set_ylim(*global_ylim)

        for point in ratio_points:
            ax_ratio.scatter(
                point['behavior_prob'],
                point['policy_prob'],
                color=ratio_state_styles[point['state']],
                s=55,
                alpha=0.9,
                edgecolors='white',
                linewidths=0.5,
            )
        ax_ratio.plot(
            [prob_limit[0], prob_limit[1]],
            [prob_limit[0], prob_limit[1]],
            linestyle='--',
            color='black',
            alpha=0.6,
            linewidth=1.8,
        )
        ax_ratio.text(0.04, 0.93, 'y = x (ratio = 1)', transform=ax_ratio.transAxes, fontsize=9)
        ax_ratio.set_xlabel('Behavior Probability', fontsize=12, fontweight='bold')
        ax_ratio.set_ylabel('Current Policy Probability', fontsize=12, fontweight='bold')
        ax_ratio.set_title(f"{plot_name} | Ratio Scatter", fontsize=11, fontweight='bold')
        ax_ratio.tick_params(axis='both', which='major', labelsize=10)
        ax_ratio.grid(True, alpha=0.3, linestyle='--')
        ax_ratio.set_xlim(*prob_limit)
        ax_ratio.set_ylim(*prob_limit)
        ax_ratio.set_aspect('equal', adjustable='box')

    total_axes = nrows * ncols
    used_axes = len(plot_cases) * 2
    for flat_idx in range(used_axes, total_axes):
        row = flat_idx // ncols
        col = flat_idx % ncols
        axes[row, col].axis('off')

    import matplotlib.lines as mlines
    legend_handles = []
    legend_labels = []

    algorithm_order = [
        'GIPO(σ=0.1)',
        'GIPO(σ=0.3)',
        'GIPO(σ=0.5)',
        'GIPO(σ=1.0)',
        'GIPO(σ=2.0)',
        'No-Clip',
        'PPO',
        'SAPO',
    ]
    for alg_name in algorithm_order:
        style = algorithm_styles[alg_name]
        legend_handles.append(
            mlines.Line2D(
                [0], [0],
                marker=style['marker'],
                color='none',
                markerfacecolor=style['color'],
                markeredgecolor='none',
                markersize=math.sqrt(style['size']),
                alpha=1.0,
            )
        )
        legend_labels.append(alg_name)

    for state_name, color in ratio_state_styles.items():
        legend_handles.append(
            mlines.Line2D(
                [0], [0],
                marker='o',
                color='none',
                markerfacecolor=color,
                markeredgecolor='white',
                markersize=8,
            )
        )
        legend_labels.append(f'Ratio Scatter {state_name}')

    legend_handles.append(
        mlines.Line2D([0], [0], linestyle='--', color='black', alpha=0.6, linewidth=2)
    )
    legend_labels.append('y = x (ratio = 1)')

    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        ncol=6,
        fontsize=11,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.965], pad=2.0, w_pad=1.6, h_pad=1.8)
    
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
        output_pdf = os.path.join(output_dir, 'bias_variance_analysis.pdf')
        output_png = os.path.join(output_dir, 'bias_variance_analysis.png')
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
        print(f"图片已保存为: {output_pdf}")
        print(f"图片已保存为: {output_png}")
    else:
        print("未保存图片（默认关闭，可使用 --save-images 开启 PNG/PDF 导出）。")
    
    # 显示图片
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        pareto_border=args.pareto_border,
        save_images=args.save_images,
    )