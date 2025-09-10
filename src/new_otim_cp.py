import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from policy import Policy
from tqdm import tqdm
import os
from cartpole import CartPoleEnv
from scipy.interpolate import griddata


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

def collect_initial_states_fitness(env, policy, n_episodes=10, **kwargs):
    results = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        if 'custom_state' in kwargs and kwargs['custom_state'] is not None:
            env.unwrapped.state = np.array(kwargs['custom_state'])
            obs = np.array(kwargs['custom_state'])
        elif 'custom_bounds' in kwargs and kwargs['custom_bounds'] is not None:
            obs, _ = env.reset_custom(custom_bounds=kwargs['custom_bounds'])

        initial_theta = obs[2]
        initial_theta_dot = obs[3]

        total_reward = 0
        for step in range(kwargs.get('max_steps', 500)):
            action = policy.get_action(obs)
            custom_noise = kwargs.get('custom_noise', 0.1)
            obs, reward, terminated, truncated, _ = env.step(action, custom_noise)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        fitness = total_reward
        results.append((initial_theta, initial_theta_dot, fitness))
    
    return results

def experimento_controle():
    """Combinação de condições iniciais"""
    step = 0.1  # Resolução ajustada
    results = []
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    
    theta_range = np.arange(-0.15, 0.16, step)      # ±8.6° (dentro do limite)
    theta_dot_range = np.arange(-2.0, 2.1, step)  

    for theta_val in tqdm(theta_range, desc="Exp Controle"):
        for theta_dot_val in theta_dot_range:
            
            episode_fitness = []
            for trial in range(5): 
                # [cart_pos, cart_vel, pole_angle, pole_ang_vel]
                custom_state = [0.0, 0.0, theta_val, theta_dot_val]
                
                trial_results = collect_initial_states_fitness(
                    env, policy, n_episodes=10, 
                    custom_state=custom_state,
                    max_steps=500,
                    custom_noise=0.1
                )
                
                if trial_results:
                    episode_fitness.append(trial_results[0][2])  
            
            if episode_fitness:
                avg_fitness = np.mean(episode_fitness)
                # (theta_inicial, theta_dot_inicial, fitness_medio)
                results.append((theta_val, theta_dot_val, avg_fitness))
    
    env.close()

    results = np.array(results)
    
    path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/cartpole/exp_controle/fitness_landscape.npy')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, results)
    
    plot_results(results=results, exp='exp_controle', name='controle')

def experimento_1_n_episodios():
    """Variação do número de episódios"""
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    episodios = [2, 5, 10, 15, 20, 50]

    for ep in episodios:
        print(f"Experimento 1: n_episodes = {ep}")
        
        results = collect_initial_states_fitness(
            env, policy, n_episodes=ep,
            max_steps=500,
            custom_noise=0.1
        )

        for i, (theta, theta_dot, fitness) in enumerate(results):
            print(f"Episódio {i+1} | Theta inicial: {theta:.3f} | Fitness: {fitness}")
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/cartpole/exp_1/fitness_landscape_ep_{ep}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        plot_results(results=results, exp='exp_1', name=f'ep_{ep}')

def experimento_2_duracao():
    """Variação da duração do episódio"""
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    duracao = [50, 100, 200, 300, 400, 500]

    for d in duracao:
        print(f"Experimento 2: maxsteps = {d}")
        
        results = collect_initial_states_fitness(
            env, policy, n_episodes=10,
            max_steps=d,  
            custom_noise=0.1
        )

        for i, (theta, theta_dot, fitness) in enumerate(results):
            print(f"Episódio {i+1} | Fitness: {fitness} | Max_steps: {d}")
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/cartpole/exp_2/fitness_landscape_d_{d}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'mstep_' + str(d)
        plot_results(results=results, exp='exp_2', name=name)

def experimento_3_ruido():
    """Variação do ruído na ação"""
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    noise = [0.001, 0.01, 0.05, 0.1, 0.5, 1]

    for n in noise:
        print(f"Experimento 3: noise = {n}")

        results = collect_initial_states_fitness(
            env, policy, n_episodes=10,
            max_steps=500,
            custom_noise=n 
        )

        for i, (theta, theta_dot, fitness) in enumerate(results):
            print(f"Episódio {i+1} | Fitness: {fitness} | Noise: {n}")
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/cartpole/exp_3/fitness_landscape_n_{str(n)}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'noise_' + str(n)
        plot_results(results=results, exp='exp_3', name=name)

def experimento_4_condicoes():
    """Variação das condições iniciais"""
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    interval_ranges_theta = [0.05, 0.1, 0.15]      
    interval_ranges_theta_dot = [0.5, 1.0, 2.0]    

    for t in interval_ranges_theta:
        for td in interval_ranges_theta_dot:
            print(f"Experimento 4: ranges = theta±{t}, theta_dot±{td}")
            
            results = collect_initial_states_fitness(
                env, policy, n_episodes=10,
                max_steps=500,
                custom_bounds=(t, td),  # (theta_range, theta_dot_range)
                custom_noise=0.1
            )

            for i, (theta, theta_dot, fitness) in enumerate(results):
                print(f"Episódio {i+1} | Theta: {theta:.3f} | Theta_dot: {theta_dot:.3f} | Fitness: {fitness}")

            env.close()
            results = np.array(results)

            ranges = str(t) + "_" + str(td)

            path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/cartpole/exp_4/fitness_landscape_r_{ranges}.npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, results)

            name = 'ranges_' + ranges
            plot_results(results=results, exp='exp_4', name=name)

def experimento_5_fitness():
    """Fitness com diferentes métricas - CORRIGIDO"""
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    
    fitness_methods = ['mean', 'min', 'max', 'median', 'std']
    
    for method in fitness_methods:
        print(f"Experimento 5: fitness_method = {method}")
        
        all_results = []
        
        for ep in range(10):
            episode_results = collect_initial_states_fitness(
                env, policy, n_episodes=1,
                max_steps=500,
                custom_noise=0.1
            )
            all_results.extend(episode_results)
        
        # Agrupar por estados iniciais similares e calcular fitness agregado
        results_dict = {}
        for theta, theta_dot, fitness in all_results:
            # Arredondar estados para agrupamento
            key = (round(theta, 2), round(theta_dot, 2))
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(fitness)
        
        # Calcular fitness final baseado no método
        final_results = []
        for (theta, theta_dot), fitness_list in results_dict.items():
            if method == 'mean':
                final_fitness = np.mean(fitness_list)
            elif method == 'min':
                final_fitness = np.min(fitness_list)
            elif method == 'max':
                final_fitness = np.max(fitness_list)
            elif method == 'median':
                final_fitness = np.median(fitness_list)
            elif method == 'std':
                final_fitness = np.std(fitness_list)
            
            final_results.append((theta, theta_dot, final_fitness))
                
        env.close()
        results = np.array(final_results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/cartpole/exp_5/fitness_landscape_{method}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = f'fitness_{method}'
        plot_results(results=results, exp='exp_5', name=name)

def experimento_6_pesos():
    """Fitness com pesos sobre min, mean e max - CORRIGIDO"""
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    
    # Diferentes combinações de pesos [w_min, w_mean, w_max]
    peso_combinations = [
        (1.0, 0.0, 0.0),  # Apenas mínimo
        (0.0, 1.0, 0.0),  # Apenas média
        (0.0, 0.0, 1.0),  # Apenas máximo
        (0.3, 0.4, 0.3),  # Balanceado
        (0.5, 0.3, 0.2),  # Foco no mínimo
        (0.1, 0.2, 0.7),  # Foco no máximo
    ]
    
    for w_min, w_mean, w_max in peso_combinations:
        print(f"Experimento 6: pesos = ({w_min}, {w_mean}, {w_max})")
        
        # Coletar múltiplos episódios para calcular min, mean, max
        all_results = []
        
        # Gerar diferentes estados iniciais
        for trial in range(5):
            episode_results = collect_initial_states_fitness(
                env, policy, n_episodes=10,  # 5 episódios por estado inicial
                max_steps=500,
                custom_noise=0.1
            )
            all_results.extend(episode_results)
        
        # Agrupar por estados iniciais similares
        results_dict = {}
        for theta, theta_dot, fitness in all_results:
            key = (round(theta, 2), round(theta_dot, 2))
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(fitness)
        
        # Calcular fitness ponderado
        final_results = []
        for (theta, theta_dot), fitness_list in results_dict.items():
            if len(fitness_list) >= 3:  # Apenas se tiver dados suficientes
                min_fitness = np.min(fitness_list)
                mean_fitness = np.mean(fitness_list)
                max_fitness = np.max(fitness_list)
                
                # Fitness final = combinação ponderada
                weighted_fitness = w_min * min_fitness + w_mean * mean_fitness + w_max * max_fitness
                final_results.append((theta, theta_dot, weighted_fitness))

        env.close()
        results = np.array(final_results)

        peso_str = f"{w_min}_{w_mean}_{w_max}".replace('.', '')
        
        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/cartpole/exp_6/fitness_landscape_w_{peso_str}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = f'pesos_{peso_str}'
        plot_results(results=results, exp='exp_6', name=name)

# def plot_results(results, exp, name):
#     X = results[:, 0] # eixo X: ângulo
#     Y = results[:, 1] # eixo Y: velocidade angular
#     Z = results[:, 2] # eixo Z: Fitness   

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_trisurf(X, Y, Z, cmap='viridis')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     save_dir = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/cartpole/{exp}')
#     os.makedirs(save_dir, exist_ok=True)

#     save_path = os.path.join(save_dir, f'{name}.png')
#     plt.savefig(save_path)

# def plot_results(results, exp, name):
    # """Mantém a função de plot original para compatibilidade"""
    # X = results[:, 0]  # eixo X: theta (ângulo inicial)
    # Y = results[:, 1]  # eixo Y: theta_dot (velocidade angular inicial)
    # Z = results[:, 2]  # eixo Z: Fitness   

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(X, Y, Z, cmap='viridis')

    # ax.set_xlabel('Theta - Ângulo Inicial (rad)')
    # ax.set_ylabel('Theta_dot - Vel. Angular Inicial (rad/s)')
    # ax.set_zlabel('Fitness (Duração)')
    # ax.set_title(f'Fitness Landscape - {exp}/{name}')

    # save_dir = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/cartpole/{exp}')
    # os.makedirs(save_dir, exist_ok=True)

    # save_path = os.path.join(save_dir, f'{name}.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()  # Fechar figura para economizar memória
    
    # # Imprimir estatísticas
    # print(f"Fitness - Min: {np.min(Z):.2f}, Max: {np.max(Z):.2f}, Média: {np.mean(Z):.2f}")

def plot_results(results, exp, name):
    X = results[:, 0]  # eixo X: theta (ângulo inicial)
    Y = results[:, 1]  # eixo Y: theta_dot (velocidade angular inicial)
    Z = results[:, 2]  # eixo Z: Fitness   

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    n_points = len(X)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if n_points < 3:
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=100, alpha=0.8)
        # plt.colorbar(scatter, ax=ax, label='Fitness')
        
    elif n_points < 10:
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=80, alpha=0.8)

        try:
            ax.plot_trisurf(X, Y, Z, cmap='viridis', alpha=0.3)
        except:
            pass
        
        # plt.colorbar(scatter, ax=ax, label='Fitness')
        
    else:
        surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', alpha=0.8)
        # plt.colorbar(surf, ax=ax, label='Fitness')

    save_dir = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/cartpole/{exp}')
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