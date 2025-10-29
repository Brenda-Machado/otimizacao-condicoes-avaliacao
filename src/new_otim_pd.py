"""
Author: Brenda Silva Machado.

Env: PendulumV-1.

Action function: 

Observation: (x,y) and angular velocity

Default parameters:

episodes = 10
maxsteps = 500
noise = ?
fitness = mean
interval_initial = ([-pi, pi], [-1,1])

VERSÃO CORRIGIDA - Todos os experimentos ajustados para fitness landscape consistente
"""

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from policy import Policy
from tqdm import tqdm
import os
from pendulum import PendulumEnv

def collect_initial_states_fitness_pendulum(env, policy, n_episodes, **kwargs):
    results = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()

        if 'custom_state' in kwargs and kwargs['custom_state'] is not None:
            env.unwrapped.state = np.array(kwargs['custom_state'])
            obs = env._get_obs() 
        elif 'custom_bounds' in kwargs and kwargs['custom_bounds'] is not None:
            obs, _ = env.reset_custom(custom_bounds=kwargs['custom_bounds'])
        
        initial_theta = np.arctan2(obs[1], obs[0])
        initial_theta_dot = obs[2]

        total_reward = 0
        for step in range(kwargs.get('max_steps', 500)):
            action = policy.get_action(obs)
            if 'custom_noise' in kwargs:
                action = action + np.random.normal(0, kwargs['custom_noise'])
                action = np.clip(action, -2.0, 2.0) 
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        fitness = total_reward
        results.append((initial_theta, initial_theta_dot, fitness))
    
    return results

def experimento_controle():
    """Combinação de condições iniciais"""
    step = 0.1  
    results = []
    env = PendulumEnv()
    policy = Policy(input_size=3)

    theta_range = np.arange(-np.pi, np.pi + step, step)     
    theta_dot_range = np.arange(-1.0, 1.0 + step, step)        

    for theta_val in tqdm(theta_range, desc="Exp Controle"):
        for theta_dot_val in theta_dot_range:
            
            episode_fitness = []
            for trial in range(5): 
                custom_state = [theta_val, theta_dot_val]
                
                trial_results = collect_initial_states_fitness_pendulum(
                    env, policy, n_episodes=1, 
                    custom_state=custom_state,
                    max_steps=500,
                    custom_noise=0.1
                )
                
                if trial_results:
                    episode_fitness.append(trial_results[0][2]) 
            
            if episode_fitness:
                avg_fitness = np.mean(episode_fitness)
                results.append((theta_val, theta_dot_val, avg_fitness))
    
    env.close()

    results = np.array(results)

    path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum/exp_controle/fitness_landscape.npy')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, results)
    
    plot_results(results=results, exp='exp_controle', name='controle')

def experimento_1_n_episodios():
    """Variação do número de episódios"""
    step = 0.1
    env = PendulumEnv()
    policy = Policy(input_size=3)
    episodios = [2, 5, 10, 15, 20, 50]

    theta_range = np.arange(-np.pi, np.pi + step, step)     
    theta_dot_range = np.arange(-1.0, 1.0 + step, step)

    for ep in episodios:
        print(f"Experimento 1: n_episodes = {ep}")
        results = []
        
        for theta_val in tqdm(theta_range, desc=f"Exp 1 - ep {ep}"):
            for theta_dot_val in theta_dot_range:
                
                episode_fitness = []
                for trial in range(ep): 
                    custom_state = [theta_val, theta_dot_val]
                    
                    trial_results = collect_initial_states_fitness_pendulum(
                        env, policy, n_episodes=1,
                        custom_state=custom_state,
                        max_steps=500,
                        custom_noise=0.1
                    )
                    
                    if trial_results:
                        episode_fitness.append(trial_results[0][2])
                
                if episode_fitness:
                    avg_fitness = np.mean(episode_fitness)
                    results.append((theta_val, theta_dot_val, avg_fitness))
            
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_1/fitness_landscape_ep_{ep}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        plot_results(results=results, exp='exp_1', name=f'ep_{ep}')

def experimento_2_duracao():
    """Variação da duração do episódio"""
    step = 0.1
    env = PendulumEnv()
    policy = Policy(input_size=3)
    duracao = [50, 100, 200, 300, 400, 500]

    theta_range = np.arange(-np.pi, np.pi + step, step)     
    theta_dot_range = np.arange(-1.0, 1.0 + step, step)

    for d in duracao:
        print(f"Experimento 2: maxsteps = {d}")
        results = []
        
        for theta_val in tqdm(theta_range, desc=f"Exp 2 - dur {d}"):
            for theta_dot_val in theta_dot_range:
                
                episode_fitness = []
                for trial in range(5): 
                    custom_state = [theta_val, theta_dot_val]
                    
                    trial_results = collect_initial_states_fitness_pendulum(
                        env, policy, n_episodes=1,
                        max_steps=d,
                        custom_state=custom_state,
                        custom_noise=0.1
                    )
                    
                    if trial_results:
                        episode_fitness.append(trial_results[0][2])
                
                if episode_fitness:
                    avg_fitness = np.mean(episode_fitness)
                    results.append((theta_val, theta_dot_val, avg_fitness))
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_2/fitness_landscape_d_{d}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'mstep_' + str(d)
        plot_results(results=results, exp='exp_2', name=name)

def experimento_3_ruido():
    """Variação do ruído na ação"""
    step = 0.1
    env = PendulumEnv()
    policy = Policy(input_size=3)
    noise = [0.001, 0.01, 0.05, 0.1, 0.5, 1]

    theta_range = np.arange(-np.pi, np.pi + step, step)     
    theta_dot_range = np.arange(-1.0, 1.0 + step, step)

    for n in noise:
        print(f"Experimento 3: noise = {n}")
        results = []
        
        for theta_val in tqdm(theta_range, desc=f"Exp 3 - noise {n}"):
            for theta_dot_val in theta_dot_range:
                
                episode_fitness = []
                for trial in range(5): 
                    custom_state = [theta_val, theta_dot_val]
                    
                    trial_results = collect_initial_states_fitness_pendulum(
                        env, policy, n_episodes=1,
                        max_steps=500,
                        custom_state=custom_state,
                        custom_noise=n
                    )
                    
                    if trial_results:
                        episode_fitness.append(trial_results[0][2])
                
                if episode_fitness:
                    avg_fitness = np.mean(episode_fitness)
                    results.append((theta_val, theta_dot_val, avg_fitness))
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_3/fitness_landscape_n_{str(n)}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'noise_' + str(n)
        plot_results(results=results, exp='exp_3', name=name)

def experimento_4_condicoes():
    """Variação das condições iniciais"""
    step = 0.1
    env = PendulumEnv()
    policy = Policy(input_size=3)
    interval_ranges_theta = [np.pi/4, np.pi/2, np.pi]      
    interval_ranges_theta_dot = [2.0, 4.0, 8.0]            

    for t in interval_ranges_theta:
        for td in interval_ranges_theta_dot:
            print(f"Experimento 4: ranges = theta±{t:.2f}, theta_dot±{td}")
            
            theta_range = np.arange(-t, t + step, step)
            theta_dot_range = np.arange(-td, td + step, step)
            results = []
            
            for theta_val in tqdm(theta_range, desc=f"Exp 4 - {t:.2f}_{td}"):
                for theta_dot_val in theta_dot_range:
                    
                    episode_fitness = []
                    for trial in range(5):
                        custom_state = [theta_val, theta_dot_val]
                        
                        trial_results = collect_initial_states_fitness_pendulum(
                            env, policy, n_episodes=1,
                            max_steps=500,
                            custom_state=custom_state,
                            custom_noise=0.1
                        )
                        
                        if trial_results:
                            episode_fitness.append(trial_results[0][2])
                    
                    if episode_fitness:
                        avg_fitness = np.mean(episode_fitness)
                        results.append((theta_val, theta_dot_val, avg_fitness))

            env.close()
            results = np.array(results)

            ranges = f"{t:.2f}_{td}"

            path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_4/fitness_landscape_r_{ranges}.npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, results)

            name = 'ranges_' + ranges.replace('.', '_')
            plot_results(results=results, exp='exp_4', name=name)

def experimento_5_fitness():
    """Fitness com diferentes métricas - CORRIGIDO"""
    step = 0.1
    env = PendulumEnv()
    policy = Policy(input_size=3)
    
    fitness_methods = ['mean', 'min', 'max', 'median', 'std']
    
    theta_range = np.arange(-np.pi, np.pi + step, step)     
    theta_dot_range = np.arange(-1.0, 1.0 + step, step)
    
    for method in fitness_methods:
        print(f"Experimento 5: fitness_method = {method}")
        results = []
        
        for theta_val in tqdm(theta_range, desc=f"Exp 5 - {method}"):
            for theta_dot_val in theta_dot_range:
                
                episode_fitness = []
                for trial in range(10):
                    custom_state = [theta_val, theta_dot_val]
                    
                    trial_results = collect_initial_states_fitness_pendulum(
                        env, policy, n_episodes=1,
                        max_steps=500,
                        custom_state=custom_state,
                        custom_noise=0.1
                    )
                    
                    if trial_results:
                        episode_fitness.append(trial_results[0][2])
                
                if episode_fitness:
                    if method == 'mean':
                        final_fitness = np.mean(episode_fitness)
                    elif method == 'min':
                        final_fitness = np.min(episode_fitness)
                    elif method == 'max':
                        final_fitness = np.max(episode_fitness)
                    elif method == 'median':
                        final_fitness = np.median(episode_fitness)
                    elif method == 'std':
                        final_fitness = np.std(episode_fitness)
                    
                    results.append((theta_val, theta_dot_val, final_fitness))
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_5/fitness_landscape_{method}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = f'fitness_{method}'
        plot_results(results=results, exp='exp_5', name=name)

def experimento_6_pesos():
    """Fitness com pesos sobre min, mean e max - CORRIGIDO"""
    step = 0.1
    env = PendulumEnv()
    policy = Policy(input_size=3)
    
    peso_combinations = [
        (1.0, 0.0, 0.0),  
        (0.0, 1.0, 0.0),  
        (0.0, 0.0, 1.0),  
        (0.3, 0.4, 0.3),  
        (0.5, 0.3, 0.2),  
        (0.1, 0.2, 0.7),  
    ]
    
    theta_range = np.arange(-np.pi, np.pi + step, step)     
    theta_dot_range = np.arange(-1.0, 1.0 + step, step)
    
    for w_min, w_mean, w_max in peso_combinations:
        print(f"Experimento 6: pesos = ({w_min}, {w_mean}, {w_max})")
        results = []
        
        for theta_val in tqdm(theta_range, desc=f"Exp 6 - {w_min}_{w_mean}_{w_max}"):
            for theta_dot_val in theta_dot_range:
                
                episode_fitness = []
                for trial in range(10):
                    custom_state = [theta_val, theta_dot_val]
                    
                    trial_results = collect_initial_states_fitness_pendulum(
                        env, policy, n_episodes=1,
                        max_steps=500,
                        custom_state=custom_state,
                        custom_noise=0.1
                    )
                    
                    if trial_results:
                        episode_fitness.append(trial_results[0][2])
                
                if episode_fitness:
                    min_fitness = np.min(episode_fitness)
                    mean_fitness = np.mean(episode_fitness)
                    max_fitness = np.max(episode_fitness)
                    
                    weighted_fitness = w_min * min_fitness + w_mean * mean_fitness + w_max * max_fitness
                    results.append((theta_val, theta_dot_val, weighted_fitness))

        env.close()
        results = np.array(results)

        peso_str = f"{w_min}_{w_mean}_{w_max}".replace('.', '')
        
        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_6/fitness_landscape_w_{peso_str}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = f'pesos_{peso_str}'
        plot_results(results=results, exp='exp_6', name=name)

def plot_results(results, exp, name):
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
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='Blues', s=100, alpha=0.8)
        
    elif n_points < 10:
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='Blues', s=80, alpha=0.8)
        
        try:
            ax.plot_trisurf(X, Y, Z, cmap='Blues', alpha=0.3)
        except:
            pass
        
    else:
        surf = ax.plot_trisurf(X, Y, Z, cmap='Blues', alpha=0.8)

    save_dir = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 
    
def run_all_experimentos():
    """Executa todos os experimentos corrigidos"""
    
    experimento_controle()
    experimento_1_n_episodios()
    experimento_2_duracao()
    experimento_3_ruido()
    experimento_4_condicoes()
    # experimento_5_fitness()
    # experimento_6_pesos()

"""
Visualizações adicionais para análise de fitness landscapes.
Sem alterar as funções existentes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_controle_multiangulo(results_path):
    """Plota o experimento controle em múltiplos ângulos."""
    results = np.load(results_path)
    X, Y, Z = results[:, 0], results[:, 1], results[:, 2]
    
    angulos = [
        (30, 45, "padrao"),
        (0, 0, "frontal"),
        (0, 90, "lateral"),
        (90, 0, "topo"),
        (45, 135, "diagonal1"),
        (60, 225, "diagonal2")
    ]
    
    fig = plt.figure(figsize=(18, 12))
    
    for idx, (elev, azim, nome) in enumerate(angulos, 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        ax.plot_trisurf(X, Y, Z, cmap='Blues', alpha=0.8)
        ax.set_xlabel('Theta')
        ax.set_ylabel('Theta_dot')
        ax.set_zlabel('Fitness')
        ax.set_title(f'Visão {nome} (elev={elev}, azim={azim})')
        ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    save_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/plots/pendulum/exp_controle/multiangulo.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_diff(ref_results, var_results):
    """Calcula diferença entre landscapes."""
    ref_dict = {(r[0], r[1]): r[2] for r in ref_results}
    
    diffs = []
    for var_r in var_results:
        key = (var_r[0], var_r[1])
        if key in ref_dict:
            diff = var_r[2] - ref_dict[key]
            diffs.append([var_r[0], var_r[1], diff])
    
    return np.array(diffs)

def plot_diff_3d(diff_results, exp, name):
    """Plota diferença em 3D."""
    X, Y, Z = diff_results[:, 0], diff_results[:, 1], diff_results[:, 2]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_trisurf(X, Y, Z, cmap='RdBu_r', alpha=0.8)
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_zlabel('Diff Fitness')
    ax.set_title(f'Diferença: {name} vs Controle')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    save_path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}/diff_3d_{name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_diff_heatmap(diff_results, exp, name):
    """Plota diferença como heatmap 2D."""
    X, Y, Z = diff_results[:, 0], diff_results[:, 1], diff_results[:, 2]
    
    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    
    Z_grid = np.full((len(y_unique), len(x_unique)), np.nan)
    
    for i, x_val in enumerate(x_unique):
        for j, y_val in enumerate(y_unique):
            mask = (X == x_val) & (Y == y_val)
            if mask.any():
                Z_grid[j, i] = Z[mask][0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(Z_grid, cmap='RdBu_r', aspect='auto', 
                   extent=[x_unique.min(), x_unique.max(), 
                          y_unique.min(), y_unique.max()],
                   origin='lower')
    
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_title(f'Heatmap Diferença: {name} vs Controle')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Diff Fitness')
    
    save_path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}/diff_heatmap_{name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_experiment_1():
    """Compara variações de episódios com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    episodios = [2, 5, 10, 15, 20, 50]
    
    for ep in episodios:
        var = np.load(f'{base_path}/exp_1/fitness_landscape_ep_{ep}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_1', f'ep_{ep}')
        plot_diff_heatmap(diff, 'exp_1', f'ep_{ep}')

def compare_experiment_2():
    """Compara variações de duração com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    duracoes = [50, 100, 200, 300, 400, 500]
    
    for d in duracoes:
        var = np.load(f'{base_path}/exp_2/fitness_landscape_d_{d}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_2', f'mstep_{d}')
        plot_diff_heatmap(diff, 'exp_2', f'mstep_{d}')

def compare_experiment_3():
    """Compara variações de ruído com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    noises = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    
    for n in noises:
        var = np.load(f'{base_path}/exp_3/fitness_landscape_n_{str(n)}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_3', f'noise_{str(n)}')
        plot_diff_heatmap(diff, 'exp_3', f'noise_{str(n)}')

def compare_experiment_4():
    """Compara variações de condições iniciais com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    interval_ranges_theta = [np.pi/4, np.pi/2, np.pi]
    interval_ranges_theta_dot = [2.0, 4.0, 8.0]
    
    for t in interval_ranges_theta:
        for td in interval_ranges_theta_dot:
            ranges = f"{t:.2f}_{td}"
            var = np.load(f'{base_path}/exp_4/fitness_landscape_r_{ranges}.npy')
            diff = compute_diff(ref, var)
            
            plot_diff_3d(diff, 'exp_4', f'ranges_{ranges.replace(".", "_")}')
            plot_diff_heatmap(diff, 'exp_4', f'ranges_{ranges.replace(".", "_")}')


def compare_experiment_5():
    """Compara diferentes métricas de fitness."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_5/fitness_landscape_mean.npy')
    
    methods = ['min', 'max', 'median', 'std']
    
    for method in methods:
        var = np.load(f'{base_path}/exp_5/fitness_landscape_{method}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_5', f'fitness_{method}')
        plot_diff_heatmap(diff, 'exp_5', f'fitness_{method}')

def compare_experiment_6():
    """Compara diferentes pesos."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_6/fitness_landscape_w_010000.npy')
    
    pesos = ['100000', '001000', '030403', '050302', '010207']
    
    for p in pesos:
        var = np.load(f'{base_path}/exp_6/fitness_landscape_w_{p}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_6', f'pesos_{p}')
        plot_diff_heatmap(diff, 'exp_6', f'pesos_{p}')

def create_comparison_grid(exp_num, variations):
    """Cria grade comparativa para um experimento."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    n_vars = len(variations)
    rows = (n_vars + 2) // 3
    
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
    axes = axes.flatten()
    
    for idx, (path, label) in enumerate(variations):
        var = np.load(f'{base_path}/{path}')
        diff = compute_diff(ref, var)
        
        X, Y, Z = diff[:, 0], diff[:, 1], diff[:, 2]
        x_unique = np.unique(X)
        y_unique = np.unique(Y)
        
        Z_grid = np.full((len(y_unique), len(x_unique)), np.nan)
        for i, x_val in enumerate(x_unique):
            for j, y_val in enumerate(y_unique):
                mask = (X == x_val) & (Y == y_val)
                if mask.any():
                    Z_grid[j, i] = Z[mask][0]
        
        im = axes[idx].imshow(Z_grid, cmap='RdBu_r', aspect='auto',
                             extent=[x_unique.min(), x_unique.max(),
                                    y_unique.min(), y_unique.max()],
                             origin='lower')
        axes[idx].set_title(label)
        axes[idx].set_xlabel('Theta')
        axes[idx].set_ylabel('Theta_dot')
        plt.colorbar(im, ax=axes[idx])
    
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/exp_{exp_num}/comparison_grid.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_all_visualizations():
    """Executa todas as visualizações adicionais."""
    
    # Multi-ângulo controle
    print("Gerando visualizações multi-ângulo do controle...")
    controle_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum/exp_controle/fitness_landscape.npy')
    plot_controle_multiangulo(controle_path)
    
    # Comparações exp 1
    print("Comparando experimento 1 (episódios)...")
    compare_experiment_1()
    variations_1 = [(f'exp_1/fitness_landscape_ep_{ep}.npy', f'{ep} eps') 
                    for ep in [2, 5, 10, 15, 20, 50]]
    create_comparison_grid(1, variations_1)
    
    # Comparações exp 2
    print("Comparando experimento 2 (duração)...")
    compare_experiment_2()
    variations_2 = [(f'exp_2/fitness_landscape_d_{d}.npy', f'{d} steps') 
                    for d in [50, 100, 200, 300, 400, 500]]
    create_comparison_grid(2, variations_2)
    
    # Comparações exp 3
    print("Comparando experimento 3 (ruído)...")
    compare_experiment_3()
    variations_3 = [(f'exp_3/fitness_landscape_n_{str(n)}.npy', f'noise={n}') 
                    for n in [0.001, 0.01, 0.05, 0.1, 0.5, 1]]
    create_comparison_grid(3, variations_3)
    
    print("Visualizações concluídas!")

    compare_experiment_4()
    variations_4 = []
    for t in [np.pi/4, np.pi/2, np.pi]:
        for td in [2.0, 4.0, 8.0]:
            ranges = f"{t:.2f}_{td}"
            variations_4.append((f'exp_4/fitness_landscape_r_{ranges}.npy', 
                               f'θ±{t:.2f}, θ_dot±{td}'))
    create_comparison_grid(4, variations_4)


if __name__ == "__main__":
    run_all_experimentos()
    run_all_visualizations()