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


def find_pareto_optimal(stats):
    """
    找出帕累托最优的算法
    
    Args:
        stats: 统计结果字典
    
    Returns:
        list: 帕累托最优算法的名称列表
    """
    # 定义所有算法及其对应的偏差和方差
    algorithms = {
        'No-Clip': {
            'bias': 0.0,  # No-Clip是基准，偏差为0
            'variance': stats['Var_g_is']
        },
        'GIPO(σ=0.1)': {
            'bias': abs(stats['E_g_wis_sigma_0.1'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_0.1']
        },
        'GIPO(σ=0.3)': {
            'bias': abs(stats['E_g_wis_sigma_0.3'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_0.3']
        },
        'GIPO(σ=0.5)': {
            'bias': abs(stats['E_g_wis_sigma_0.5'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_0.5']
        },
        'GIPO(σ=1.0)': {
            'bias': abs(stats['E_g_wis_sigma_1.0'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_1.0']
        },
        'GIPO(σ=2.0)': {
            'bias': abs(stats['E_g_wis_sigma_2.0'] - stats['E_g_is']),
            'variance': stats['Var_g_wis_sigma_2.0']
        },
        'PPO': {
            'bias': abs(stats['E_g_ppo'] - stats['E_g_is']),
            'variance': stats['Var_g_ppo']
        },
        'SAPO': {
            'bias': abs(stats['E_g_sapo'] - stats['E_g_is']),
            'variance': stats['Var_g_sapo']
        }
    }
    
    # 找出帕累托最优的算法
    pareto_optimal = []
    
    for alg_name, alg_metrics in algorithms.items():
        is_pareto = True
        for other_name, other_metrics in algorithms.items():
            if alg_name == other_name:
                continue
            # 检查是否存在另一个算法在偏差和方差上都优于当前算法
            if (other_metrics['bias'] <= alg_metrics['bias'] and 
                other_metrics['variance'] <= alg_metrics['variance'] and
                (other_metrics['bias'] < alg_metrics['bias'] or 
                 other_metrics['variance'] < alg_metrics['variance'])):
                is_pareto = False
                break
        if is_pareto:
            pareto_optimal.append(alg_name)
    
    return pareto_optimal, algorithms


def create_mixed_policy(env, policy_weights):
    """
    创建混合策略
    
    Args:
        env: 环境
        policy_weights: 字典，格式为 {策略名称: (权重, 策略对象)}
    
    Returns:
        SoftmaxPolicy: 近似混合分布的softmax策略
        dict: 混合策略的概率分布
    """
    # 计算混合策略的概率分布
    mixed_probs = {}
    for a in env.actions:
        mixed_probs[a] = sum(weight * policy.get_prob('S0', a) 
                            for _, (weight, policy) in policy_weights.items())
    
    # 计算对应的softmax theta值
    # 情况1: up=left, down=right，使用[0, a, 0, a]形式
    if abs(mixed_probs['up'] - mixed_probs['left']) < 0.01 and abs(mixed_probs['down'] - mixed_probs['right']) < 0.01:
        ratio = mixed_probs['down'] / mixed_probs['up']
        theta = [0, np.log(ratio), 0, np.log(ratio)]
    # 情况2: up=down, left=right，使用[0, 0, a, a]形式
    elif abs(mixed_probs['up'] - mixed_probs['down']) < 0.01 and abs(mixed_probs['left'] - mixed_probs['right']) < 0.01:
        ratio = mixed_probs['left'] / mixed_probs['up']
        theta = [0, 0, np.log(ratio), np.log(ratio)]
    # 情况3: 其他情况，使用[0, a, b, c]形式
    else:
        ratio_down = mixed_probs['down'] / mixed_probs['up']
        ratio_left = mixed_probs['left'] / mixed_probs['up']
        ratio_right = mixed_probs['right'] / mixed_probs['up']
        theta = [0, np.log(ratio_down), np.log(ratio_left), np.log(ratio_right)]
    
    # 创建softmax策略
    mu = SoftmaxPolicy(env)
    mu.set_theta('S0', theta)
    mu.set_theta('S1', theta)
    mu.set_theta('S2', theta)
    
    return mu, mixed_probs, theta


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
        help="Enable black border for Pareto-optimal markers.",
    )
    parser.add_argument(
        "--no-pareto-border",
        dest="pareto_border",
        action="store_false",
        help="Disable black border for Pareto-optimal markers.",
    )
    parser.set_defaults(pareto_border=True)
    return parser.parse_args()


def main(output_dir="rollouts/grid", pareto_border=True):
    env = GridWorld2x2(gamma=0.9)
    
    # 目标策略π：偏好'下'和'右' (θ=[0,1,0,1])
    pi = SoftmaxPolicy(env)
    pi.set_theta('S0', [0, 1, 0, 1])
    pi.set_theta('S1', [0, 1, 0, 1])
    pi.set_theta('S2', [0, 1, 0, 1])
    
    print("=" * 80)
    print("目标策略π: 偏好'下'和'右' (θ=[0,1,0,1])")
    print("=" * 80)
    print(f"π(上|S0): {pi.get_prob('S0', 'up'):.4f}")
    print(f"π(下|S0): {pi.get_prob('S0', 'down'):.4f}")
    print(f"π(左|S0): {pi.get_prob('S0', 'left'):.4f}")
    print(f"π(右|S0): {pi.get_prob('S0', 'right'):.4f}")
    
    # 定义基础子策略
    random_policy = UniformPolicy(env)  # 随机策略：每个动作25%
    
    prefer_right_policy = SoftmaxPolicy(env)  # 偏好右策略：theta=[0,0,0,1]
    prefer_right_policy.set_theta('S0', [0, 0, 0, 1])
    prefer_right_policy.set_theta('S1', [0, 0, 0, 1])
    prefer_right_policy.set_theta('S2', [0, 0, 0, 1])
    
    prefer_down_policy = SoftmaxPolicy(env)  # 偏好下策略：theta=[0,1,0,0]
    prefer_down_policy.set_theta('S0', [0, 1, 0, 0])
    prefer_down_policy.set_theta('S1', [0, 1, 0, 0])
    prefer_down_policy.set_theta('S2', [0, 1, 0, 0])
    
    prefer_left_policy = SoftmaxPolicy(env)  # 偏好左策略：theta=[0,0,1,0]
    prefer_left_policy.set_theta('S0', [0, 0, 1, 0])
    prefer_left_policy.set_theta('S1', [0, 0, 1, 0])
    prefer_left_policy.set_theta('S2', [0, 0, 1, 0])

    prefer_up_policy = SoftmaxPolicy(env)  # 偏好下策略：theta=[0,1,0,0]
    prefer_up_policy.set_theta('S0', [1, 0, 0, 0])
    prefer_up_policy.set_theta('S1', [1, 0, 0, 0])
    prefer_up_policy.set_theta('S2', [1, 0, 0, 0])
    
    # 情况A：μ = 100%随机策略
    print("\n" + "=" * 80)
    print("情况A: μ = 100%随机策略")
    print("=" * 80)
    
    policy_weights_A = {
        'random': (1.0, random_policy)
    }
    mu_A, mixed_probs_A, theta_A = create_mixed_policy(env, policy_weights_A)
    
    print("混合策略组成: 100%随机策略")
    print("混合策略的概率分布:")
    for a in env.actions:
        print(f"  {a}: {mixed_probs_A[a]:.4f}")
    print(f"对应的theta值: {theta_A}")
    
    stats_A = compute_gradient_stats(pi, mu_A, env, state='S0', target_action='right')
    
    print(f"\nμ(上|S0): {mu_A.get_prob('S0', 'up'):.4f}")
    print(f"μ(下|S0): {mu_A.get_prob('S0', 'down'):.4f}")
    print(f"μ(左|S0): {mu_A.get_prob('S0', 'left'):.4f}")
    print(f"μ(右|S0): {mu_A.get_prob('S0', 'right'):.4f}")
    print(f"\nV(S0): {stats_A['V']['S0']:.4f}")
    print(f"A(S0, 右): {stats_A['A']['S0']['right']:.4f}")
    print(f"\n重要性采样比:")
    for a, rho in stats_A['rho_values'].items():
        print(f"  {a}: {rho:.4f}")
    
    print(f"\nIS估计量期望: {stats_A['E_g_is']:.6f}")
    print(f"IS估计量方差: {stats_A['Var_g_is']:.6f}")
    print(f"WIS估计量期望: {stats_A['E_g_wis']:.6f}")
    print(f"WIS估计量方差: {stats_A['Var_g_wis']:.6f}")
    print(f"PPO估计量期望: {stats_A['E_g_ppo']:.6f}")
    print(f"PPO估计量方差: {stats_A['Var_g_ppo']:.6f}")
    print(f"SAPO估计量期望: {stats_A['E_g_sapo']:.6f}")
    print(f"SAPO估计量方差: {stats_A['Var_g_sapo']:.6f}")
    
    # 情况B：μ = 40%随机策略 + 30%偏好右策略 + 30%偏好下策略
    print("\n" + "=" * 80)
    print("情况B: μ = 40%随机策略 + 30%偏好右策略 + 30%偏好下策略")
    print("=" * 80)
    
    policy_weights_B = {
        'random': (0.4, random_policy),
        'prefer_right': (0.3, prefer_right_policy),
        'prefer_down': (0.3, prefer_down_policy)
    }
    mu_B, mixed_probs_B, theta_B = create_mixed_policy(env, policy_weights_B)
    
    print("混合策略组成:")
    for name, (weight, _) in policy_weights_B.items():
        print(f"  {weight*100:.0f}% {name}")
    print("混合策略的概率分布:")
    for a in env.actions:
        print(f"  {a}: {mixed_probs_B[a]:.4f}")
    print(f"对应的theta值: {theta_B}")
    
    stats_B = compute_gradient_stats(pi, mu_B, env, state='S0', target_action='right')
    
    print(f"\nμ(上|S0): {mu_B.get_prob('S0', 'up'):.4f}")
    print(f"μ(下|S0): {mu_B.get_prob('S0', 'down'):.4f}")
    print(f"μ(左|S0): {mu_B.get_prob('S0', 'left'):.4f}")
    print(f"μ(右|S0): {mu_B.get_prob('S0', 'right'):.4f}")
    print(f"\nV(S0): {stats_B['V']['S0']:.4f}")
    print(f"A(S0, 右): {stats_B['A']['S0']['right']:.4f}")
    print(f"\n重要性采样比:")
    for a, rho in stats_B['rho_values'].items():
        print(f"  {a}: {rho:.4f}")
    
    print(f"\nIS估计量期望: {stats_B['E_g_is']:.6f}")
    print(f"IS估计量方差: {stats_B['Var_g_is']:.6f}")
    print(f"WIS估计量期望: {stats_B['E_g_wis']:.6f}")
    print(f"WIS估计量方差: {stats_B['Var_g_wis']:.6f}")
    print(f"PPO估计量期望: {stats_B['E_g_ppo']:.6f}")
    print(f"PPO估计量方差: {stats_B['Var_g_ppo']:.6f}")
    print(f"SAPO估计量期望: {stats_B['E_g_sapo']:.6f}")
    print(f"SAPO估计量方差: {stats_B['Var_g_sapo']:.6f}")
    
    # 情况C：μ = 20%随机策略 + 40%偏好右策略 + 40%偏好下策略
    print("\n" + "=" * 80)
    print("情况C: μ = 20%随机策略 + 40%偏好右策略 + 40%偏好下策略")
    print("=" * 80)
    
    policy_weights_C = {
        'random': (0.2, random_policy),
        'prefer_right': (0.4, prefer_right_policy),
        'prefer_down': (0.4, prefer_down_policy)
    }
    mu_C, mixed_probs_C, theta_C = create_mixed_policy(env, policy_weights_C)
    
    print("混合策略组成:")
    for name, (weight, _) in policy_weights_C.items():
        print(f"  {weight*100:.0f}% {name}")
    print("混合策略的概率分布:")
    for a in env.actions:
        print(f"  {a}: {mixed_probs_C[a]:.4f}")
    print(f"对应的theta值: {theta_C}")
    
    stats_C = compute_gradient_stats(pi, mu_C, env, state='S0', target_action='right')
    
    print(f"\nμ(上|S0): {mu_C.get_prob('S0', 'up'):.4f}")
    print(f"μ(下|S0): {mu_C.get_prob('S0', 'down'):.4f}")
    print(f"μ(左|S0): {mu_C.get_prob('S0', 'left'):.4f}")
    print(f"μ(右|S0): {mu_C.get_prob('S0', 'right'):.4f}")
    print(f"\nV(S0): {stats_C['V']['S0']:.4f}")
    print(f"A(S0, 右): {stats_C['A']['S0']['right']:.4f}")
    print(f"\n重要性采样比:")
    for a, rho in stats_C['rho_values'].items():
        print(f"  {a}: {rho:.4f}")
    
    print(f"\nIS估计量期望: {stats_C['E_g_is']:.6f}")
    print(f"IS估计量方差: {stats_C['Var_g_is']:.6f}")
    print(f"WIS估计量期望: {stats_C['E_g_wis']:.6f}")
    print(f"WIS估计量方差: {stats_C['Var_g_wis']:.6f}")
    print(f"PPO估计量期望: {stats_C['E_g_ppo']:.6f}")
    print(f"PPO估计量方差: {stats_C['Var_g_ppo']:.6f}")
    print(f"SAPO估计量期望: {stats_C['E_g_sapo']:.6f}")
    print(f"SAPO估计量方差: {stats_C['Var_g_sapo']:.6f}")
    
    # 情况D：μ = π（策略mu与pi一致）
    print("\n" + "=" * 80)
    print("情况D: μ = π（策略mu与pi一致）")
    print("=" * 80)
    
    # 直接使用pi策略作为mu
    mu_D = pi
    
    print("策略说明: μ = π（完全一致）")
    print("策略的概率分布:")
    for a in env.actions:
        print(f"  {a}: {mu_D.get_prob('S0', a):.4f}")
    
    stats_D = compute_gradient_stats(pi, mu_D, env, state='S0', target_action='right')
    
    print(f"\nμ(上|S0): {mu_D.get_prob('S0', 'up'):.4f}")
    print(f"μ(下|S0): {mu_D.get_prob('S0', 'down'):.4f}")
    print(f"μ(左|S0): {mu_D.get_prob('S0', 'left'):.4f}")
    print(f"μ(右|S0): {mu_D.get_prob('S0', 'right'):.4f}")
    print(f"\nV(S0): {stats_D['V']['S0']:.4f}")
    print(f"A(S0, 右): {stats_D['A']['S0']['right']:.4f}")
    print(f"\n重要性采样比:")
    for a, rho in stats_D['rho_values'].items():
        print(f"  {a}: {rho:.4f}")
    
    print(f"\nIS估计量期望: {stats_D['E_g_is']:.6f}")
    print(f"IS估计量方差: {stats_D['Var_g_is']:.6f}")
    print(f"WIS估计量期望: {stats_D['E_g_wis']:.6f}")
    print(f"WIS估计量方差: {stats_D['Var_g_wis']:.6f}")
    print(f"PPO估计量期望: {stats_D['E_g_ppo']:.6f}")
    print(f"PPO估计量方差: {stats_D['Var_g_ppo']:.6f}")
    print(f"SAPO估计量期望: {stats_D['E_g_sapo']:.6f}")
    print(f"SAPO估计量方差: {stats_D['Var_g_sapo']:.6f}")
    
    # 总结对比表格
    print("\n" + "=" * 80)
    print("总结对比 - 期望")
    print("=" * 80)
    
    # 打印期望相关的表头
    header_exp = f"{'情况':<20} {'E_g_is':<15}"
    header_exp += f" {'E_g_wis(σ=0.1)':<18} {'E_g_wis(σ=0.3)':<18} {'E_g_wis(σ=0.5)':<18} {'E_g_wis(σ=1.0)':<18} {'E_g_wis(σ=2.0)':<18}"
    header_exp += f" {'E_g_ppo':<15} {'E_g_sapo':<15}"
    print(header_exp)
    print("-" * 120)
    
    # 打印期望相关的数据
    for case_name, stats in [('A (100%随机)', stats_A), ('B (40%随机+30%右+30%下)', stats_B), 
                              ('C (20%随机+40%右+40%下)', stats_C), ('D (μ=π)', stats_D)]:
        row = f"{case_name:<20} {stats['E_g_is']:>14.6f}"
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
    header_var = f"{'情况':<20} {'Var_g_is':<15}"
    header_var += f" {'Var_g_wis(σ=0.1)':<18} {'Var_g_wis(σ=0.3)':<18} {'Var_g_wis(σ=0.5)':<18} {'Var_g_wis(σ=1.0)':<18} {'Var_g_wis(σ=2.0)':<18}"
    header_var += f" {'Var_g_ppo':<15} {'Var_g_sapo':<15}"
    print(header_var)
    print("-" * 120)
    
    # 打印方差相关的数据
    for case_name, stats in [('A (100%随机)', stats_A), ('B (40%随机+30%右+30%下)', stats_B), 
                              ('C (20%随机+40%右+40%下)', stats_C), ('D (μ=π)', stats_D)]:
        row = f"{case_name:<20} {stats['Var_g_is']:>14.6f}"
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
    header_bias = f"{'情况':<20} {'|E_g_wis(σ=0.1)-E_g_is|':<25} {'|E_g_wis(σ=0.3)-E_g_is|':<25} {'|E_g_wis(σ=0.5)-E_g_is|':<25} {'|E_g_wis(σ=1.0)-E_g_is|':<25} {'|E_g_wis(σ=2.0)-E_g_is|':<25}"
    header_bias += f" {'|E_g_ppo-E_g_is|':<20} {'|E_g_sapo-E_g_is|':<22}"
    print(header_bias)
    print("-" * 150)
    
    # 打印偏差相关的数据（使用绝对值）
    for case_name, stats in [('A (100%随机)', stats_A), ('B (40%随机+30%右+30%下)', stats_B), 
                              ('C (20%随机+40%右+40%下)', stats_C), ('D (μ=π)', stats_D)]:
        row = f"{case_name:<20}"
        row += f" {abs(stats['E_g_wis_sigma_0.1'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_0.3'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_0.5'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_1.0'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_wis_sigma_2.0'] - stats['E_g_is']):>24.6f}"
        row += f" {abs(stats['E_g_ppo'] - stats['E_g_is']):>19.6f}"
        row += f" {abs(stats['E_g_sapo'] - stats['E_g_is']):>21.6f}"
        print(row)
    
    # 帕累托最优分析
    print("\n" + "=" * 80)
    print("帕累托最优分析（基于偏差绝对值和方差）")
    print("=" * 80)
    print("说明：帕累托最优算法是指在偏差和方差两个指标上，不存在其他算法同时优于它的算法")
    print("（偏差和方差都是越小越好）\n")
    
    all_cases = [
        ('A (100%随机)', stats_A),
        ('B (40%随机+30%右+30%下)', stats_B),
        ('C (20%随机+40%右+40%下)', stats_C),
        ('D (μ=π)', stats_D)
    ]
    
    for case_name, stats in all_cases:
        pareto_optimal, algorithms = find_pareto_optimal(stats)
        print(f"{case_name}:")
        print(f"  帕累托最优算法: {', '.join(pareto_optimal)}")
        print(f"  算法详细指标:")
        for alg_name in sorted(algorithms.keys()):
            alg_metrics = algorithms[alg_name]
            is_pareto = "✓" if alg_name in pareto_optimal else " "
            print(f"    [{is_pareto}] {alg_name:<15} 偏差: {alg_metrics['bias']:>10.6f}, 方差: {alg_metrics['variance']:>10.6f}")
        print()
    
    # 绘制偏差-方差图
    print("\n" + "=" * 80)
    print("绘制偏差-方差图...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 绘制第一个子图：2x2网格世界环境（田字格，边缘不出头）
    ax_grid = axes[0, 0]
    ax_grid.set_xlim(0, 2)
    ax_grid.set_ylim(0, 2)
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    
    # 绘制田字格：外框 + 中间十字线，线段正好落在[0,2]边界上
    for i in range(3):
        # 水平线段：从 x=0 到 x=2
        ax_grid.plot([0, 2], [i, i], color='black', linewidth=2)
        # 垂直线段：从 y=0 到 y=2
        ax_grid.plot([i, i], [0, 2], color='black', linewidth=2)
    
    # 标注状态名称（使用下标）
    # 左上角 S₀
    ax_grid.text(0.5, 1.5, 'S$_0$', fontsize=20, fontweight='bold', ha='center', va='center')
    # 右上角 S₁
    ax_grid.text(1.5, 1.5, 'S$_1$', fontsize=20, fontweight='bold', ha='center', va='center')
    # 左下角 S₂
    ax_grid.text(0.5, 0.5, 'S$_2$', fontsize=20, fontweight='bold', ha='center', va='center')
    # 右下角 S_G
    ax_grid.text(1.5, 0.5, 'S$_G$', fontsize=20, fontweight='bold', ha='center', va='center')
    
    ax_grid.set_title('GridWorld 2×2 Environment', fontsize=12, fontweight='bold')
    
    # Algorithm names and corresponding colors and markers
    # 颜色与 TensorBoard 绘图示例保持一致：
    # PPO: 深蓝色 #1f77b4, GIPO: 橙色 #ff7f0e, SAPO: 绿色 #2ca02c
    # 不同 σ 使用同一颜色、不同形状区分
    algorithm_styles = {
        'No-Clip':      {'color': '#d62728', 'marker': 'X', 'size': 100},  # 红色，区分于其它算法
        'GIPO(σ=0.1)':  {'color': '#ff7f0e', 'marker': 'o', 'size': 80},
        'GIPO(σ=0.3)':  {'color': '#ff7f0e', 'marker': 'P', 'size': 80},  # 五边形
        'GIPO(σ=0.5)':  {'color': '#ff7f0e', 'marker': 's', 'size': 80},
        'GIPO(σ=1.0)':  {'color': '#ff7f0e', 'marker': '^', 'size': 80},
        'GIPO(σ=2.0)':  {'color': '#ff7f0e', 'marker': 'v', 'size': 80},
        'PPO':          {'color': '#1f77b4', 'marker': 'D', 'size': 80},
        'SAPO':         {'color': '#2ca02c', 'marker': 'p', 'size': 80}
    }
    
    # Case name mapping to English
    case_name_map = {
        'A (100%随机)': 'Case A',
        'B (40%随机+30%右+30%下)': 'Case B',
        'C (20%随机+40%右+40%下)': 'Case C',
        'D (μ=π)': 'Case D (μ=π)'
    }
    
    # Plot subplots for first 3 cases only (放在其他3个子图中)
    plot_positions = [(0, 1), (1, 0), (1, 1)]
    plot_cases = all_cases[:3]
    plot_data = []
    global_biases = []
    global_variances = []

    for case_name, stats in plot_cases:
        pareto_optimal, algorithms = find_pareto_optimal(stats)
        plot_data.append((case_name, pareto_optimal, algorithms))
        global_biases.extend([alg['bias'] for alg in algorithms.values()])
        global_variances.extend([alg['variance'] for alg in algorithms.values()])

    # Unified axis range for all 3 case subplots.
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

    for idx, (case_name, pareto_optimal, algorithms) in enumerate(plot_data):
        ax = axes[plot_positions[idx]]

        # Plot all algorithms:
        # draw larger markers first, smaller markers later (on top).
        # For same-size overlap, draw PPO earlier so GIPO points stay visible.
        sorted_alg_names = sorted(
            algorithms.keys(),
            key=lambda name: (
                -algorithm_styles[name]['size'],      # larger first
                0 if name == 'PPO' else 1,            # PPO earlier in ties
                name
            )
        )
        for alg_name in sorted_alg_names:
            alg_metrics = algorithms[alg_name]
            style = algorithm_styles[alg_name]
            is_pareto = alg_name in pareto_optimal

            scatter_kwargs = {
                'c': style['color'],
                'marker': style['marker'],
                's': style['size'],  # fixed marker size for all algorithms
                'alpha': 1.0,  # keep colors identical to legend
                'label': alg_name,
                'zorder': 3 if is_pareto else 2
            }
            if is_pareto and pareto_border:
                scatter_kwargs['edgecolors'] = 'black'
                scatter_kwargs['linewidths'] = 2
            ax.scatter(alg_metrics['bias'], alg_metrics['variance'], **scatter_kwargs)
        
        # Plot Pareto frontier (connect Pareto optimal points)
        pareto_points = [(algorithms[alg]['bias'], algorithms[alg]['variance']) 
                         for alg in pareto_optimal]
        if len(pareto_points) > 1:
            # Sort by bias
            pareto_points.sort(key=lambda x: x[0])
            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]
            # 虚线风格与示例中的网格线风格一致（--，适中粗细）
            ax.plot(pareto_x, pareto_y, linestyle='--', color='black',
                    alpha=0.6, linewidth=2, label='Pareto Frontier')
        
        # 轴标签和标题风格向示例对齐：较大的字体和加粗标题
        ax.set_xlabel('Bias', fontsize=14, fontweight='bold')
        ax.set_ylabel('Variance', fontsize=14, fontweight='bold')
        ax.set_title(case_name_map.get(case_name, case_name),
                     fontsize=16, fontweight='bold')
        # 刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=12)
        # 网格线使用虚线与示例一致
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Use unified global axis range across all 3 case subplots.
        ax.set_xlim(*global_xlim)
        ax.set_ylim(*global_ylim)
    
    # Create handles for shared legend using the algorithm styles
    import matplotlib.lines as mlines
    legend_handles = []
    legend_labels = []
    
    # matplotlib的ncol参数是按列填充的（从上到下，然后从左到右）
    # 要实现每行4个的效果，需要重新组织顺序
    # 期望的显示（按行）：
    # 行1: GIPO(σ=0.1), GIPO(σ=0.3), GIPO(σ=0.5), GIPO(σ=1.0)
    # 行2: GIPO(σ=2.0), No-Clip, PPO, SAPO
    # 行3: Pareto Frontier
    
    # 由于matplotlib按列填充，共3行4列，需要按列的顺序添加
    # 列1（从上到下）: GIPO(σ=0.1), GIPO(σ=2.0), Pareto Frontier
    # 列2（从上到下）: GIPO(σ=0.3), No-Clip, (空)
    # 列3（从上到下）: GIPO(σ=0.5), PPO, (空)
    # 列4（从上到下）: GIPO(σ=1.0), SAPO, (空)
    
    legend_order_by_column = [
        # 列1
        'GIPO(σ=0.1)',
        'GIPO(σ=2.0)',
        'Pareto Frontier',
        # 列2
        'GIPO(σ=0.3)',
        'No-Clip',
        None,  # 占位符
        # 列3
        'GIPO(σ=0.5)',
        'PPO',
        None,  # 占位符
        # 列4
        'GIPO(σ=1.0)',
        'SAPO',
        None,  # 占位符
    ]
    
    for alg_name in legend_order_by_column:
        if alg_name is None:
            # 添加空白占位符
            handle = mlines.Line2D([0], [0], marker='', color='none', linestyle='')
            legend_handles.append(handle)
            legend_labels.append('')
        elif alg_name == 'Pareto Frontier':
            # Pareto Frontier 是线型
            handle = mlines.Line2D(
                [0], [0],
                linestyle='--',
                color='black',
                alpha=0.6,
                linewidth=2
            )
            legend_handles.append(handle)
            legend_labels.append(alg_name)
        elif alg_name in algorithm_styles:
            # 其他算法是标记点
            style = algorithm_styles[alg_name]
            handle = mlines.Line2D(
                [0], [0],
                marker=style['marker'],
                color='none',
                markerfacecolor=style['color'],
                markeredgecolor='none',
                markersize=math.sqrt(style['size']),
                alpha=1.0,
            )
            legend_handles.append(handle)
            legend_labels.append(alg_name)
    
    # Create a shared legend below the title，字体大小与示例一致风格（相对较大）
    # 调整为每行4个：共9个项目，第一排4个，第二排4个，第三排1个
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        ncol=4,  # 每行4个
        fontsize=14,
        framealpha=0.9,
        bbox_to_anchor=(0.1, 0.96, 0.8, 0.1),  # 降低y坐标，从0.98改为0.96
        handlelength=2.5,
        handletextpad=0.8,
        columnspacing=2.0,  # 增加列间距
    )
    
    # Increase spacing between subplots to prevent overlap
    # 由于图例位置降低，可以增加子图区域，从0.90改为0.92
    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=2.0, w_pad=1.5, h_pad=1.5)
    
    # 保存图片（同时导出 PDF 和 PNG）
    os.makedirs(output_dir, exist_ok=True)
    output_pdf = os.path.join(output_dir, 'bias_variance_analysis.pdf')
    output_png = os.path.join(output_dir, 'bias_variance_analysis.png')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    print(f"图片已保存为: {output_pdf}")
    print(f"图片已保存为: {output_png}")
    
    # 显示图片
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir, pareto_border=args.pareto_border)
