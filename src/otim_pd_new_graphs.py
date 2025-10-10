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

if __name__ == "__main__":
    run_all_experimentos()