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
    """Função auxiliar para coletar (theta_inicial, theta_dot_inicial, fitness) no Pendulum"""
    results = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        
        # Aplicar configurações customizadas se existirem
        if 'custom_state' in kwargs and kwargs['custom_state'] is not None:
            env.unwrapped.state = np.array(kwargs['custom_state'])
            obs = env._get_obs()  # Recalcular observação após mudança de estado
        elif 'custom_bounds' in kwargs and kwargs['custom_bounds'] is not None:
            # Reset com bounds customizados
            obs, _ = env.reset_custom(custom_bounds=kwargs['custom_bounds'])
        
        # Capturar estado inicial (theta, theta_dot) do estado interno
        initial_theta = env.unwrapped.state[0]      # Ângulo
        initial_theta_dot = env.unwrapped.state[1]  # Velocidade angular
        
        # Executar episódio
        total_reward = 0
        for step in range(kwargs.get('max_steps', 500)):
            action = policy.get_action(obs)
            
            # Aplicar ruído se especificado
            if 'custom_noise' in kwargs:
                action = action + np.random.normal(0, kwargs['custom_noise'])
                action = np.clip(action, -2.0, 2.0)  # Limites da ação do Pendulum
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Fitness = recompensa total (mais negativa = pior)
        fitness = total_reward
        
        # Armazenar (theta_inicial, theta_dot_inicial, fitness)
        results.append((initial_theta, initial_theta_dot, fitness))
    
    return results

def experimento_controle():
    """Combinação de condições iniciais - CORRIGIDO"""
    step = 0.15  # Resolução ajustada
    results = []
    env = PendulumEnv()
    policy = Policy(input_size=3)
    
    # CORREÇÃO: Ranges válidos para Pendulum
    theta_range = np.arange(-np.pi, np.pi + step, step)     # Ângulo completo [-π, π]
    theta_dot_range = np.arange(-8.0, 8.1, step*4)         # Velocidades realistas

    for theta_val in tqdm(theta_range, desc="Exp Controle"):
        for theta_dot_val in theta_dot_range:
            
            episode_fitness = []
            for trial in range(3):  # Múltiplas execuções por condição
                # Estado inicial: [theta, theta_dot]
                custom_state = [theta_val, theta_dot_val]
                
                trial_results = collect_initial_states_fitness_pendulum(
                    env, policy, n_episodes=1, 
                    custom_state=custom_state,
                    max_steps=500,
                    custom_noise=0.1
                )
                
                if trial_results:
                    episode_fitness.append(trial_results[0][2])  # fitness
            
            if episode_fitness:
                avg_fitness = np.mean(episode_fitness)
                # Salvar como (theta_inicial, theta_dot_inicial, fitness_medio)
                results.append((theta_val, theta_dot_val, avg_fitness))
    
    env.close()

    results = np.array(results)
    
    # Salvar dados
    path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum/exp_controle/fitness_landscape.npy')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, results)
    
    plot_results(results=results, exp='exp_controle', name='controle')

def experimento_1_n_episodios():
    """Variação do número de episódios - CORRIGIDO"""
    env = PendulumEnv()
    policy = Policy(input_size=3)
    episodios = [2, 5, 10, 15, 20, 50]

    for ep in episodios:
        print(f"Experimento 1: n_episodes = {ep}")
        
        # CORREÇÃO: Coletar estados iniciais vs fitness
        results = collect_initial_states_fitness_pendulum(
            env, policy, n_episodes=ep,
            max_steps=500,
            custom_noise=0.1
        )

        for i, (theta, theta_dot, fitness) in enumerate(results):
            print(f"Episódio {i+1} | Theta inicial: {theta:.3f} | Fitness: {fitness:.2f}")
            
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_1/fitness_landscape_ep_{ep}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        plot_results(results=results, exp='exp_1', name=f'ep_{ep}')

def experimento_2_duracao():
    """Variação da duração do episódio - CORRIGIDO"""
    env = PendulumEnv()
    policy = Policy(input_size=3)
    duracao = [50, 100, 200, 300, 400, 500]

    for d in duracao:
        print(f"Experimento 2: maxsteps = {d}")
        
        # CORREÇÃO: Coletar estados iniciais vs fitness
        results = collect_initial_states_fitness_pendulum(
            env, policy, n_episodes=10,
            max_steps=d,  # Variação da duração
            custom_noise=0.1
        )

        for i, (theta, theta_dot, fitness) in enumerate(results):
            print(f"Episódio {i+1} | Fitness: {fitness:.2f} | Max_steps: {d}")
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_2/fitness_landscape_d_{d}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'mstep_' + str(d)
        plot_results(results=results, exp='exp_2', name=name)

def experimento_3_ruido():
    """Variação do ruído na ação - CORRIGIDO"""
    env = PendulumEnv()
    policy = Policy(input_size=3)
    noise = [0.001, 0.01, 0.05, 0.1, 0.5, 1]

    for n in noise:
        print(f"Experimento 3: noise = {n}")
        
        # CORREÇÃO: Coletar estados iniciais vs fitness
        results = collect_initial_states_fitness_pendulum(
            env, policy, n_episodes=10,
            max_steps=500,
            custom_noise=n  # Variação do ruído
        )

        for i, (theta, theta_dot, fitness) in enumerate(results):
            print(f"Episódio {i+1} | Fitness: {fitness:.2f} | Noise: {n}")
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_3/fitness_landscape_n_{str(n)}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'noise_' + str(n)
        plot_results(results=results, exp='exp_3', name=name)

def experimento_4_condicoes():
    """Variação das condições iniciais - CORRIGIDO"""
    env = PendulumEnv()
    policy = Policy(input_size=3)
    
    # CORREÇÃO: Ranges mais realistas para Pendulum
    interval_ranges_theta = [np.pi/4, np.pi/2, np.pi]      # π/4, π/2, π radianos
    interval_ranges_theta_dot = [2.0, 4.0, 8.0]            # Velocidades realistas

    for t in interval_ranges_theta:
        for td in interval_ranges_theta_dot:
            print(f"Experimento 4: ranges = theta±{t:.2f}, theta_dot±{td}")
            
            # CORREÇÃO: Coletar estados iniciais vs fitness
            results = collect_initial_states_fitness_pendulum(
                env, policy, n_episodes=10,
                max_steps=500,
                custom_bounds=(t, td),  # (theta_range, theta_dot_range)
                custom_noise=0.1
            )

            for i, (theta, theta_dot, fitness) in enumerate(results):
                print(f"Episódio {i+1} | Theta: {theta:.3f} | Theta_dot: {theta_dot:.3f} | Fitness: {fitness:.2f}")

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
    env = PendulumEnv()
    policy = Policy(input_size=3)
    
    # Diferentes formas de calcular fitness
    fitness_methods = ['mean', 'min', 'max', 'median', 'std']
    
    for method in fitness_methods:
        print(f"Experimento 5: fitness_method = {method}")
        
        # Coletar múltiplos episódios para cada estado inicial
        all_results = []
        
        for ep in range(15):  # Mais episódios para estatística robusta
            episode_results = collect_initial_states_fitness_pendulum(
                env, policy, n_episodes=1,
                max_steps=500,
                custom_noise=0.1
            )
            all_results.extend(episode_results)
        
        # Agrupar por estados iniciais similares e calcular fitness agregado
        results_dict = {}
        for theta, theta_dot, fitness in all_results:
            # Arredondar estados para agrupamento
            key = (round(theta, 1), round(theta_dot, 1))
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(fitness)
        
        # Calcular fitness final baseado no método
        final_results = []
        for (theta, theta_dot), fitness_list in results_dict.items():
            if len(fitness_list) >= 2:  # Apenas se tiver dados suficientes
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

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_5/fitness_landscape_{method}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = f'fitness_{method}'
        plot_results(results=results, exp='exp_5', name=name)

def experimento_6_pesos():
    """Fitness com pesos sobre min, mean e max - CORRIGIDO"""
    env = PendulumEnv()
    policy = Policy(input_size=3)
    
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
        for trial in range(25):  # Mais trials para estatística robusta
            episode_results = collect_initial_states_fitness_pendulum(
                env, policy, n_episodes=4,  # 4 episódios por estado inicial
                max_steps=500,
                custom_noise=0.1
            )
            all_results.extend(episode_results)
        
        # Agrupar por estados iniciais similares
        results_dict = {}
        for theta, theta_dot, fitness in all_results:
            key = (round(theta, 1), round(theta_dot, 1))
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
        
        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_6/fitness_landscape_w_{peso_str}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = f'pesos_{peso_str}'
        plot_results(results=results, exp='exp_6', name=name)

def plot_results(results, exp, name):
    """Função de plot robusta que lida com diferentes quantidades de dados"""
    X = results[:, 0]  # eixo X: theta (ângulo inicial)
    Y = results[:, 1]  # eixo Y: theta_dot (velocidade angular inicial)
    Z = results[:, 2]  # eixo Z: Fitness   

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Verificar se há dados suficientes para diferentes tipos de plot
    n_points = len(X)
    
    if n_points < 3:
        # Poucos pontos: usar scatter plot
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='Blues', s=100, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label='Fitness')
        print(f"Usando scatter plot (apenas {n_points} pontos)")
        
    elif n_points < 10:
        # Poucos pontos: usar scatter + linhas conectoras
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='Blues', s=80, alpha=0.8)
        
        # Tentar trisurf se possível
        try:
            ax.plot_trisurf(X, Y, Z, cmap='Blues', alpha=0.3)
        except:
            # Se trisurf falhar, usar apenas scatter
            pass
        
        plt.colorbar(scatter, ax=ax, label='Fitness')
        print(f"Usando scatter + trisurf (apenas {n_points} pontos)")
        
    else:
        # Muitos pontos: usar trisurf normalmente
        surf = ax.plot_trisurf(X, Y, Z, cmap='Blues', alpha=0.8)
        plt.colorbar(surf, ax=ax, label='Fitness')
        print(f"Usando trisurf ({n_points} pontos)")

    ax.set_xlabel('Theta - Ângulo Inicial (rad)')
    ax.set_ylabel('Theta_dot - Vel. Angular Inicial (rad/s)')
    ax.set_zlabel('Fitness (Recompensa Total)')
    ax.set_title(f'Fitness Landscape - {exp}/{name}\n({n_points} pontos de dados)')

    save_dir = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Fechar figura para economizar memória
    
    # Imprimir estatísticas
    print(f"Fitness - Min: {np.min(Z):.2f}, Max: {np.max(Z):.2f}, Média: {np.mean(Z):.2f}")

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