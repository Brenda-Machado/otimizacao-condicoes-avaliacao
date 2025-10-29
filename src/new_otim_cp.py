"""
Author: Brenda Silva Machado.
Modified: Visualizações aprimoradas adicionadas (minimalista)

Env: CartPoleV-1.
Adiciona múltiplas visões 3D para controle e comparações detalhadas
"""

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
from policy import Policy
from cartpole import CartPoleEnv

def get_action(obs, param1, param2):
    val = param1 * obs[2] + param2 * obs[3]
    return int(val > 0)

def run_episode(env, param1, param2, max_steps=500, noise_range=None, custom_state=None):
    obs, _ = env.reset()
    
    if custom_state is not None:
        env.unwrapped.state = np.array(custom_state)

    total_reward = 0
    for i in range(max_steps):
        action = get_action(obs, param1, param2)
        if noise_range:
            action = np.clip(action + np.random.uniform(*noise_range), 0, 1)
            action = int(round(action))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def run_episode_exp(env, param1, param2, max_steps=500, noise_range=None, custom_state=None, metodo=None):
    obs, _ = env.reset()
    
    if custom_state is not None:
        env.unwrapped.state = np.array(custom_state)

    states_rewards = []

    for i in range(max_steps):
        action = get_action(obs, param1, param2)
        if noise_range:
            action = np.clip(action + np.random.uniform(*noise_range), 0, 1)
            action = int(round(action))
        obs, reward, terminated, truncated, _ = env.step(action)

        ang = obs[2]
        ang_vel = obs[3]
        states_rewards.append((ang, ang_vel, reward))

        if terminated or truncated:
            break
    
    return states_rewards

def run_episode_exp_pesos(env, param1, param2, max_steps=500, noise_range=None, custom_state=None, metodo=None):
    obs, _ = env.reset()
    
    if custom_state is not None:
        env.unwrapped.state = np.array(custom_state)

    states_rewards = []

    for i in range(max_steps):
        action = get_action(obs, param1, param2)
        if noise_range:
            action = np.clip(action + np.random.uniform(*noise_range), 0, 1)
            action = int(round(action))
        obs, reward, terminated, truncated, _ = env.step(action)

        ang = obs[2]
        ang_vel = obs[3]
        states_rewards.append((ang, ang_vel, reward))

        if terminated or truncated:
            break
    
    return states_rewards

# ==================== FUNÇÕES DE VISUALIZAÇÃO APRIMORADAS ====================

def load_and_interpolate(filepath, grid_x, grid_y):
    """Carrega dados e interpola em uma grade regular"""
    data = np.load(filepath)
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic', fill_value=np.nan)
    return grid_z

def plot_results_multiple_views(results, exp, name):
    """Cria visualização com múltiplos ângulos"""
    X, Y, Z = results[:, 0], results[:, 1], results[:, 2]
    fig = plt.figure(figsize=(20, 15))
    
    views = [
        (30, 45, "Vista Padrão (30°, 45°)"),
        (10, 45, "Vista Baixa (10°, 45°)"),
        (60, 45, "Vista Alta (60°, 45°)"),
        (30, 0, "Vista Frontal (30°, 0°)"),
        (30, 90, "Vista Lateral (30°, 90°)"),
        (90, 0, "Vista de Cima (90°, 0°)")
    ]
    
    for idx, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none', antialiased=True)
        ax.set_xlabel('Theta', fontsize=10)
        ax.set_ylabel('Theta Dot', fontsize=10)
        ax.set_zlabel('Fitness', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        if idx == 3:
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.suptitle(f'Fitness Landscape - {name} (Múltiplas Visões)', fontsize=16, fontweight='bold', y=0.98)
    save_dir = os.path.expanduser(f'~/tcc/plots/cartpole/{exp}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}_multiple_views.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Múltiplas visões salvas: {save_path}")

def plot_comparison_diff(control_path, exp_path, exp_name, param_value, exp_dir):
    """Cria visualização completa de diferença"""
    if not os.path.exists(exp_path):
        print(f"  ⚠ Arquivo não encontrado: {exp_path}")
        return
    
    control_data = np.load(control_path)
    exp_data = np.load(exp_path)
    
    # Criar grade regular
    grid_resolution = 100
    grid_theta = np.linspace(-3, 3, grid_resolution)
    grid_theta_dot = np.linspace(-0.2, 0.2, grid_resolution)
    grid_x, grid_y = np.meshgrid(grid_theta, grid_theta_dot)
    
    control_z = load_and_interpolate(control_path, grid_x, grid_y)
    exp_z = load_and_interpolate(exp_path, grid_x, grid_y)
    diff_z = exp_z - control_z
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Superfície 3D da diferença
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    vmax_abs = np.nanmax(np.abs(diff_z))
    surf = ax1.plot_surface(grid_x, grid_y, diff_z, cmap='RdBu_r', alpha=0.8, antialiased=True, 
                           vmin=-vmax_abs, vmax=vmax_abs)
    ax1.set_xlabel('Theta', fontsize=10)
    ax1.set_ylabel('Theta Dot', fontsize=10)
    ax1.set_zlabel('Δ Fitness', fontsize=10)
    ax1.set_title(f'Diferença 3D: {exp_name} - Controle\n(param={param_value})', fontsize=11, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Δ Fitness')
    
    # 2. Heatmap 2D da diferença
    ax2 = fig.add_subplot(2, 3, 2)
    im = ax2.contourf(grid_x, grid_y, diff_z, levels=50, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
    ax2.set_xlabel('Theta', fontsize=10)
    ax2.set_ylabel('Theta Dot', fontsize=10)
    ax2.set_title(f'Heatmap 2D - Diferença\n(param={param_value})', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax2, label='Δ Fitness')
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap 2D do controle
    ax3 = fig.add_subplot(2, 3, 3)
    im_control = ax3.contourf(grid_x, grid_y, control_z, levels=50, cmap='viridis')
    ax3.set_xlabel('Theta', fontsize=10)
    ax3.set_ylabel('Theta Dot', fontsize=10)
    ax3.set_title('Heatmap 2D - Controle (Referência)', fontsize=11, fontweight='bold')
    fig.colorbar(im_control, ax=ax3, label='Fitness')
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap 2D do experimento
    ax4 = fig.add_subplot(2, 3, 4)
    im_exp = ax4.contourf(grid_x, grid_y, exp_z, levels=50, cmap='viridis')
    ax4.set_xlabel('Theta', fontsize=10)
    ax4.set_ylabel('Theta Dot', fontsize=10)
    ax4.set_title(f'Heatmap 2D - {exp_name}\n(param={param_value})', fontsize=11, fontweight='bold')
    fig.colorbar(im_exp, ax=ax4, label='Fitness')
    ax4.grid(True, alpha=0.3)
    
    # 5. Histograma das diferenças
    ax5 = fig.add_subplot(2, 3, 5)
    diff_flat = diff_z.flatten()
    diff_flat = diff_flat[~np.isnan(diff_flat)]
    ax5.hist(diff_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax5.axvline(x=np.mean(diff_flat), color='green', linestyle='--', linewidth=2, 
                label=f'Média: {np.mean(diff_flat):.2f}')
    ax5.set_xlabel('Δ Fitness', fontsize=10)
    ax5.set_ylabel('Frequência', fontsize=10)
    ax5.set_title('Distribuição das Diferenças', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Estatísticas
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    ESTATÍSTICAS DA COMPARAÇÃO
    {'='*40}
    
    Experimento: {exp_name}
    Parâmetro: {param_value}
    
    DIFERENÇAS (Experimento - Controle):
    Média: {np.nanmean(diff_flat):.4f}
    Mediana: {np.nanmedian(diff_flat):.4f}
    Desvio Padrão: {np.nanstd(diff_flat):.4f}
    Mínimo: {np.nanmin(diff_flat):.4f}
    Máximo: {np.nanmax(diff_flat):.4f}
    
    CONTROLE:
    Média: {np.nanmean(control_z):.4f}
    Desvio Padrão: {np.nanstd(control_z):.4f}
    
    EXPERIMENTO:
    Média: {np.nanmean(exp_z):.4f}
    Desvio Padrão: {np.nanstd(exp_z):.4f}
    
    MUDANÇA RELATIVA:
    {(np.nanmean(exp_z) - np.nanmean(control_z)) / abs(np.nanmean(control_z)) * 100:.2f}%
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_dir = os.path.expanduser(f'~/tcc/plots/cartpole/{exp_dir}/comparisons')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'comparison_{exp_name}_{param_value}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparação salva: comparison_{exp_name}_{param_value}.png")

def get_experiment_path(exp_dir, param):
    """Retorna o caminho do arquivo de dados do experimento"""
    base = '~/tcc/data/cartpole'
    if exp_dir == 'exp_1':
        return os.path.expanduser(f'{base}/exp_1/fitness_landscape_ep_{param}.npy')
    elif exp_dir == 'exp_2':
        return os.path.expanduser(f'{base}/exp_2/fitness_landscape_d_{param}.npy')
    elif exp_dir == 'exp_3':
        return os.path.expanduser(f'{base}/exp_3/fitness_landscape_n_{param}.npy')
    elif exp_dir == 'exp_4':
        return os.path.expanduser(f'{base}/exp_4/fitness_landscape_r_{param}.npy')
    return None

def create_summary_comparison_grid(exp_name, param_values, exp_dir):
    """Cria uma grade de comparação mostrando todos os parâmetros"""
    control_path = os.path.expanduser('~/tcc/data/cartpole/exp_controle/fitness_landscape.npy')
    
    if not os.path.exists(control_path):
        print(f"  ⚠ Controle não encontrado: {control_path}")
        return
    
    n_params = len(param_values)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Criar grade regular
    grid_resolution = 100
    grid_theta = np.linspace(-3, 3, grid_resolution)
    grid_theta_dot = np.linspace(-0.2, 0.2, grid_resolution)
    grid_x, grid_y = np.meshgrid(grid_theta, grid_theta_dot)
    
    control_z = load_and_interpolate(control_path, grid_x, grid_y)
    
    # Calcular min/max global
    all_diffs = []
    for param in param_values:
        exp_path = get_experiment_path(exp_dir, param)
        if os.path.exists(exp_path):
            exp_z = load_and_interpolate(exp_path, grid_x, grid_y)
            diff_z = exp_z - control_z
            all_diffs.append(diff_z)
    
    if all_diffs:
        vmin = min(np.nanmin(d) for d in all_diffs)
        vmax = max(np.nanmax(d) for d in all_diffs)
        vmax_abs = max(abs(vmin), abs(vmax))
    else:
        print(f"  ⚠ Nenhum dado encontrado para {exp_name}")
        return
    
    for idx, param in enumerate(param_values):
        if idx >= len(axes):
            break
        ax = axes[idx]
        exp_path = get_experiment_path(exp_dir, param)
        
        if os.path.exists(exp_path):
            exp_z = load_and_interpolate(exp_path, grid_x, grid_y)
            diff_z = exp_z - control_z
            im = ax.contourf(grid_x, grid_y, diff_z, levels=50, cmap='RdBu_r', 
                           vmin=-vmax_abs, vmax=vmax_abs)
            ax.set_xlabel('Theta', fontsize=9)
            ax.set_ylabel('Theta Dot', fontsize=9)
            ax.set_title(f'{exp_name}\nParam: {param}', fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Δ Fitness')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Dados não encontrados\n{param}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.axis('off')
    
    for idx in range(len(param_values), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Comparação em Grade: {exp_name} vs Controle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_dir = os.path.expanduser(f'~/tcc/plots/cartpole/{exp_dir}/comparisons')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'grid_comparison_{exp_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grade de comparação salva: grid_comparison_{exp_name}.png")

# ==================== EXPERIMENTOS (SEM ALTERAÇÕES) ====================

def experimento_controle():
    """Combinação de condições"""
    print("\n" + "="*60)
    print("EXPERIMENTO CONTROLE")
    print("="*60)
    
    step = 0.025
    results = []
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)

    for t in tqdm(theta, desc="Exp Controle"):
        for td in theta_dot:
            episode_rewards = []
            for trial in range(10):
                state = [td, t]
                reward, _, state_reward = policy.rollout(env=env, ntrials=1, custom_state=state)
                episode_rewards.append(reward)
            avg_reward = np.mean(episode_rewards)
            results.append((t, td, avg_reward))
    
    env.close()
    results = np.array(results)
    path = os.path.expanduser('~/tcc/data/cartpole/exp_controle/fitness_landscape.npy')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, results)
    
    plot_results(results=results, exp='exp_controle', name='controle')
    plot_results_multiple_views(results=results, exp='exp_controle', name='controle')
    print("✓ Experimento controle finalizado")

def experimento_1_n_episodios():
    """Variação do numero de episódios"""
    print("\n" + "="*60)
    print("EXPERIMENTO 1: VARIAÇÃO DO NÚMERO DE EPISÓDIOS")
    print("="*60)
    
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    episodios = [2, 5, 10, 15, 20, 50]
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)
    control_path = os.path.expanduser('~/tcc/data/cartpole/exp_controle/fitness_landscape.npy')

    for ep in episodios:
        print(f"\nProcessando: n_episodes = {ep}")
        results = []
        for t in tqdm(theta, desc=f"Exp 1 - ep {ep}"):
            for td in theta_dot:
                episode_rewards = []
                for trial in range(ep):
                    state = [td, t]
                    reward, _, _ = policy.rollout(env=env, ntrials=1, custom_state=state)
                    episode_rewards.append(reward)
                avg_reward = np.mean(episode_rewards)
                results.append((t, td, avg_reward))
        
        results = np.array(results)
        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_1/fitness_landscape_ep_{ep}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)
        plot_results(results=results, exp='exp_1', name=f'ep_{ep}')
        
        if os.path.exists(control_path):
            plot_comparison_diff(control_path, path, 'N_Episodios', f'{ep}', 'exp_1')
    
    env.close()
    create_summary_comparison_grid('N_Episodios', episodios, 'exp_1')
    print("✓ Experimento 1 finalizado")

def experimento_2_duracao():
    """Variação da duração do episódio."""
    print("\n" + "="*60)
    print("EXPERIMENTO 2: VARIAÇÃO DA DURAÇÃO DO EPISÓDIO")
    print("="*60)
    
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    duracao = [50, 100, 200, 300, 400, 500]
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)
    control_path = os.path.expanduser('~/tcc/data/cartpole/exp_controle/fitness_landscape.npy')

    for d in duracao:
        print(f"\nProcessando: maxsteps = {d}")
        results = []
        for t in tqdm(theta, desc=f"Exp 2 - dur {d}"):
            for td in theta_dot:
                episode_rewards = []
                for trial in range(10):
                    state = [td, t]
                    reward, _, _ = policy.rollout(env=env, ntrials=1, custom_state=state, custom_maxsteps=d)
                    episode_rewards.append(reward)
                avg_reward = np.mean(episode_rewards)
                results.append((t, td, avg_reward))
        
        results = np.array(results)
        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_2/fitness_landscape_d_{d}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)
        name = 'mstep_' + str(d)
        plot_results(results=results, exp='exp_2', name=name)
        
        if os.path.exists(control_path):
            plot_comparison_diff(control_path, path, 'Duracao', f'{d}', 'exp_2')
    
    env.close()
    create_summary_comparison_grid('Duracao', duracao, 'exp_2')
    print("✓ Experimento 2 finalizado")

def experimento_3_ruido():
    """Variação do ruído na ação."""
    print("\n" + "="*60)
    print("EXPERIMENTO 3: VARIAÇÃO DO RUÍDO NA AÇÃO")
    print("="*60)
    
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    noise = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)
    control_path = os.path.expanduser('~/tcc/data/cartpole/exp_controle/fitness_landscape.npy')

    for n in noise:
        print(f"\nProcessando: noise = {n}")
        results = []
        for t in tqdm(theta, desc=f"Exp 3 - noise {n}"):
            for td in theta_dot:
                episode_rewards = []
                for trial in range(10):
                    state = [td, t]
                    reward, _, _ = policy.rollout(env=env, ntrials=1, custom_state=state, custom_noise=n)
                    episode_rewards.append(reward)
                avg_reward = np.mean(episode_rewards)
                results.append((t, td, avg_reward))
        
        results = np.array(results)
        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_3/fitness_landscape_n_{str(n)}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)
        name = 'noise_' + str(n)
        plot_results(results=results, exp='exp_3', name=name)
        
        if os.path.exists(control_path):
            plot_comparison_diff(control_path, path, 'Ruido', f'{n}', 'exp_3')
    
    env.close()
    create_summary_comparison_grid('Ruido', noise, 'exp_3')
    print("✓ Experimento 3 finalizado")

def experimento_4_condicoes():
    """Variação das condições iniciais."""
    print("\n" + "="*60)
    print("EXPERIMENTO 4: VARIAÇÃO DAS CONDIÇÕES INICIAIS")
    print("="*60)
    
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    interval_ranges_theta = [3, 2, 1]
    interval_ranges_theta_dot = [0.2, 0.1, 0.05]
    control_path = os.path.expanduser('~/tcc/data/cartpole/exp_controle/fitness_landscape.npy')

    param_values = []
    for t_range in interval_ranges_theta:
        for td_range in interval_ranges_theta_dot:
            print(f"\nProcessando: ranges = theta±{t_range}, theta_dot±{td_range}")
            
            theta = np.linspace(-t_range, t_range, num=int((2*t_range) / step) + 1)
            theta_dot = np.linspace(-td_range, td_range, num=int((2*td_range) / step) + 1)
            results = []

            for t in tqdm(theta, desc=f"Exp 4 - {t_range}_{td_range}"):
                for td in theta_dot:
                    episode_rewards = []
                    for trial in range(10):
                        state = [td, t]
                        reward, _, _ = policy.rollout(env=env, ntrials=1, custom_state=state)
                        episode_rewards.append(reward)
                    avg_reward = np.mean(episode_rewards)
                    results.append((t, td, avg_reward))

            results = np.array(results)
            ranges = f"{td_range}_{t_range}"
            param_values.append(ranges)
            path = os.path.expanduser(f'~/tcc/data/cartpole/exp_4/fitness_landscape_r_{ranges}.npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, results)
            name = 'ranges_' + ranges
            plot_results(results=results, exp='exp_4', name=name)
            
            if os.path.exists(control_path):
                plot_comparison_diff(control_path, path, 'Condicoes', ranges, 'exp_4')

    env.close()
    create_summary_comparison_grid('Condicoes', param_values, 'exp_4')
    print("✓ Experimento 4 finalizado")

def experimento_5_fitness():
    """Fitness com diferentes métricas"""
    print("\n" + "="*60)
    print("EXPERIMENTO 5: VARIAÇÃO DE MÉTRICAS DE FITNESS")
    print("="*60)
    
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    fitness_methods = ['mean', 'min', 'max', 'median', 'std']
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)

    for method in fitness_methods:
        print(f"\nProcessando: fitness_method = {method}")
        results = []
        for t in tqdm(theta, desc=f"Exp 5 - {method}"):
            for td in theta_dot:
                episode_rewards = []
                for trial in range(10):
                    state = [td, t]
                    reward, _, _ = policy.rollout(env=env, ntrials=1, custom_state=state)
                    episode_rewards.append(reward)
                
                if episode_rewards:
                    if method == 'mean':
                        final_fitness = np.mean(episode_rewards)
                    elif method == 'min':
                        final_fitness = np.min(episode_rewards)
                    elif method == 'max':
                        final_fitness = np.max(episode_rewards)
                    elif method == 'median':
                        final_fitness = np.median(episode_rewards)
                    elif method == 'std':
                        final_fitness = np.std(episode_rewards)
                    results.append((t, td, final_fitness))
        
        results = np.array(results)
        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_5/fitness_landscape_{method}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)
        name = f'fitness_{method}'
        plot_results(results=results, exp='exp_5', name=name)
    
    env.close()
    print("✓ Experimento 5 finalizado")

def experimento_6_pesos():
    """Fitness com pesos sobre min, mean e max"""
    print("\n" + "="*60)
    print("EXPERIMENTO 6: VARIAÇÃO DE PESOS NO FITNESS")
    print("="*60)
    
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    peso_combinations = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.3, 0.4, 0.3),
        (0.5, 0.3, 0.2),
        (0.1, 0.2, 0.7),
    ]
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)

    for w_min, w_mean, w_max in peso_combinations:
        print(f"\nProcessando: pesos = ({w_min}, {w_mean}, {w_max})")
        results = []
        for t in tqdm(theta, desc=f"Exp 6 - {w_min}_{w_mean}_{w_max}"):
            for td in theta_dot:
                episode_rewards = []
                for trial in range(10):
                    state = [td, t]
                    reward, _, _ = policy.rollout(env=env, ntrials=1, custom_state=state)
                    episode_rewards.append(reward)
                
                if episode_rewards:
                    min_fitness = np.min(episode_rewards)
                    mean_fitness = np.mean(episode_rewards)
                    max_fitness = np.max(episode_rewards)
                    weighted_fitness = w_min * min_fitness + w_mean * mean_fitness + w_max * max_fitness
                    results.append((t, td, weighted_fitness))

        results = np.array(results)
        peso_str = f"{w_min}_{w_mean}_{w_max}".replace('.', '')
        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_6/fitness_landscape_w_{peso_str}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)
        name = f'pesos_{peso_str}'
        plot_results(results=results, exp='exp_6', name=name)
    
    env.close()
    print("✓ Experimento 6 finalizado")

def otim_weights_action_func(param_range):
    env = gym.make('CartPole-v1')
    param_range = np.arange(param_range)
    results = []

    for param1 in tqdm(param_range, desc="Varredura param1"):
        for param2 in param_range:
            total_reward = 0
            episodes = 5 
            print(f"\nTestando parâmetros: param1={param1:.3f}, param2={param2:.3f}") 

            for _ in range(episodes):
                obs, _ = env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action = get_action(obs, param1, param2)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                total_reward += ep_reward

            avg_reward = total_reward / episodes
            results.append((param1, param2, avg_reward))

    env.close()
    results = np.array(results)
    best_index = np.argmax(results[:, 2])
    best_params = results[best_index]
    print(f"\nMelhores parâmetros encontrados:")
    print(f"param1 = {best_params[0]}, param2 = {best_params[1]} --> recompensa média = {best_params[2]}")
    return results, best_index, best_params

def plot_results(results, exp, name):
    import matplotlib
    matplotlib.use('Agg')
    
    X = results[:, 0]
    Y = results[:, 1]
    Z = results[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    n_points = len(X)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if n_points < 3:
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=100, alpha=0.8)
    elif n_points < 10:
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=80, alpha=0.8)
        try:
            ax.plot_trisurf(X, Y, Z, cmap='viridis', alpha=0.3)
        except:
            pass
    else:
        surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', alpha=0.8)

    save_dir = os.path.expanduser(f'~/tcc/plots/cartpole/{exp}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.png')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    except Exception as e:
        print(f"Erro ao salvar gráfico {save_path}: {e}")
    
    plt.close(fig)

def run_all_experimentos():
    experimento_controle()
    experimento_1_n_episodios()
    experimento_2_duracao()
    experimento_3_ruido()
    experimento_4_condicoes()
    

if __name__ == "__main__":
    run_all_experimentos()