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

def run_episode(env, param1, param2, max_steps=500, noise_range=None, custom_state=None):
    obs, _ = env.reset()
    total_reward = 0
    
    for i in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(custom_state)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def experimento_controle():
    """Combinação de condições"""
    step = 0.025
    results = []
    env = PendulumEnv()
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)

    rewards = []

    for t in tqdm(theta, desc="Theta"):
        for td in theta_dot:
            state = np.array([t,td])
            reward = run_episode(env, 0.25, 0.25, custom_state=state)
            rewards.append(reward)
            results.append((t, td, reward))
    env.close()

    results = np.array(results)
    best_index = np.argmax(results[:, 2])
    best_params = results[best_index]

    print(f"\nMelhores parâmetros encontrados:")
    print(f"ângulo = {best_params[0]}, velocidade angular = {best_params[1]} --> recompensa média = {best_params[2]}")

    plot_results(results=results, exp='exp_controle', name=f'controle')

    return results, best_index, best_params

def experimento_1_n_episodios():
    """Variação do numero de episódios"""
    env = PendulumEnv()
    policy = Policy()
    episodios = [2, 5, 10, 15, 20,50]
    state_reward = []

    for ep in episodios:
        print(f"Experimento 1: n_episodes = {ep}")
        state_reward = []
        for i in range(ep):
            recompensa, steps, episode_data = policy.rollout(env=env, ntrials=1)
            state_reward.extend(episode_data)
            print(f"Episódio {i+1} | Recompensa: {recompensa:.2f} | Passos: {steps}")
            
        env.close()
        results = np.array(state_reward)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_1/states_rewards_ep_{ep}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, state_reward)

        plot_results(results=results, exp='exp_1', name=f'ep_{ep}')
        plot_results_umbounded(results=results, exp='exp_1', name=f'ep_{ep}_unb')

def experimento_2_duracao():
    """Variação da duração do episódio."""
    env = PendulumEnv()
    policy = Policy()
    duracao = [50, 100, 200, 300, 400, 500]
    state_reward = []

    for d in duracao:
        print(f"Experimento 2: maxsteps = {d}")
        state_reward = []

        for e in range(10):
            recompensa, steps, episode_data = policy.rollout(env=env, custom_maxsteps=d)
            state_reward.extend(episode_data)

            print(f"Episódio {e+1} | Recompensa: {recompensa:.2f} | Passos: {steps}")
                
        env.close()
        results = np.array(state_reward)

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_2/states_rewards_d_{d}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, state_reward)

        name = 'mstep_' + str(d)
        name_unb = 'mstep_' + str(d) + '_unb'

        plot_results(results=results, exp='exp_2', name=name)
        plot_results_umbounded(results=results, exp='exp_2', name=name_unb)

def experimento_3_ruido():
    """Variação do ruído na ação."""
    pass

def experimento_4_condicoes():
    """Variação das condições iniciais."""
    env = PendulumEnv()
    policy = Policy()
    interval_ranges = [[(-0.05, 0.05), (-0.05, 0.05), (-0.2, 0.2), (-2.0, 2.0)], [(-0.05, 0.05), (-0.05, 0.05), (-0.2, 0.2), (-1.0, 1.0)], [(-0.05, 0.05), (-0.05, 0.05), (-0.2, 0.2), (-3.0, 3.0)], [(-0.05, 0.05), (-0.05, 0.05), (-0.15, 0.15), (-3.0, 3.0)], [(-0.05, 0.05), (-0.05, 0.05), (-0.1, 0.1), (-3.0, 3.0)]]

    for r in interval_ranges:
        print(f"Experimento 4: ranges = {r}")
        state_reward = []

        for e in range(10):
            recompensa, steps, episode_data = policy.rollout(env=env,custom_state=r)
            state_reward.extend(episode_data)

            print(f"Episódio {e+1} | Recompensa: {recompensa:.2f} | Passos: {steps}")

        env.close()
        results = np.array(state_reward)

        ranges = str(r[2][1]) + "_" + str((r[3][1]))

        path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/pendulum/exp_4/states_rewards_r_{ranges}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, state_reward)

        name = 'ranges_' + ranges
        name_unb = 'ranges_' + ranges + '_unb'

        plot_results(results=results, exp='exp_4', name=name)
        plot_results_umbounded(results=results, exp='exp_4', name=name_unb)

# def experimento_5_fitness():
#     """Fitness com pesos sobre min, mean e max."""
#     env = gym.make("CartPole-v1")
#     policy = Policy()
#     pesos = [()]
#     state_reward = []

#     for p in pesos:
#         print(f"Experimento 5: pesos = {p}")
#         state_reward = []

#         for e in range(10):
#             recompensa, steps, episode_data = policy.rollout(env=env, custom_maxsteps=d)
#             state_reward.extend(episode_data)

#             print(f"Episódio {e+1} | Recompensa: {recompensa:.2f} | Passos: {steps}")
                
#         env.close()
#         results = np.array(state_reward)

#         path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/data/cartpole/exp_5/states_rewards_d_{p}.npy')
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         np.save(path, state_reward)

#         name = 'pesos_' + str(p)
#         name_unb = 'pesos_' + str(p) + '_unb'

#         plot_results(results=results, exp='exp_5', name=name)
#         plot_results_umbounded(results=results, exp='exp_5', name=name_unb)

# def experimento_6_pesos():
#     env = gym.make('CartPole-v1')
#     rewards = [run_episode(env, param1, param2) for _ in range(n_episodios)]
#     env.close()

#     min_r = np.min(rewards)
#     mean_r = np.mean(rewards)
#     max_r = np.max(rewards)

#     w_min, w_mean, w_max = pesos
#     fitness_final = w_min * min_r + w_mean * mean_r + w_max * max_r
#     return fitness_final


# def otim_weights_action_func(param_range):
#     env = gym.make('CartPole-v1')

#     param_range = np.arange(param_range)
#     results = []

#     for param1 in tqdm(param_range, desc="Varredura param1"):
#         for param2 in param_range:
#             total_reward = 0
#             episodes = 5 

#             print(f"\nTestando parâmetros: param1={param1:.3f}, param2={param2:.3f}") 

#             for _ in range(episodes):
#                 obs, _ = env.reset()
#                 done = False
#                 ep_reward = 0

#                 while not done:
#                     action = get_action(obs, param1, param2)
#                     obs, reward, terminated, truncated, _ = env.step(action)
#                     ep_reward += reward
#                     done = terminated or truncated

#                 total_reward += ep_reward

#             avg_reward = total_reward / episodes
#             results.append((param1, param2, avg_reward))

#     env.close()

#     results = np.array(results)
#     best_index = np.argmax(results[:, 2])
#     best_params = results[best_index]

#     print(f"\nMelhores parâmetros encontrados:")
#     print(f"param1 = {best_params[0]}, param2 = {best_params[1]} --> recompensa média = {best_params[2]}")

#     return results, best_index, best_params


def plot_results(results, exp, name):
    X = results[:, 0] 
    Y = results[:, 1] 
    Z = results[:, 2]      

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.2, 0.2)

    save_dir = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{name}.png')
    # plt.show()
    plt.savefig(save_path)

def plot_results_umbounded(results, exp, name):
    X = results[:, 0] 
    Y = results[:, 1] 
    Z = results[:, 2]    

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    save_dir = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{name}.png')
    # plt.show()
    plt.savefig(save_path)


def run_all_experimentos():
    # experimento_controle()
    experimento_1_n_episodios()
    experimento_2_duracao()
    # experimento_3_ruido()
    experimento_4_condicoes()
    # experimento_5_fitness()
    # experimento_6_pesos()

if __name__ == "__main__":
    run_all_experimentos()
