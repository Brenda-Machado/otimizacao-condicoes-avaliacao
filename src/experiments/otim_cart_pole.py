"""
Author: Brenda Silva Machado.

Env: CartPoleV-1.

Action function: sign(param1_observation[2]+param2_observation[3]).

Observation: cart position, cart velocity, pole angle, pole angular velocity

Possible otimizations:
    * Weights of the action function;
    * All possible combinations of the Pole Angle and the Pole Velocity;
    *

Otim TO-DO:
    * Variação da duração do episódio de avaliação;
    * Variação do ruído a ser adicionado ao motor do agente;
    * Variação do intervalo das condições iniciais;
    * Avaliação do peso de cada episódio avaliativo;
    * Modificação do peso das componentes de fitness.
"""

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def experimento_controle(param1, param2, step=0.025):
    results = []
    env = gym.make('CartPole-v1')
    pos = 0
    vel = 0
    ang = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)
    ang_vel = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)

    rewards = []

    for a in tqdm(ang, desc="ângulo"):
        for av in ang_vel:
            state = [pos, vel, a, av]
            reward = run_episode(env, param1, param2, custom_state=state)
            rewards.append(reward)
            results.append((a, av, reward))
    env.close()

    results = np.array(results)
    best_index = np.argmax(results[:, 2])
    best_params = results[best_index]

    print(f"\nMelhores parâmetros encontrados:")
    print(f"ângulo = {best_params[0]}, velocidade angular = {best_params[1]} --> recompensa média = {best_params[2]}")

    return results, best_index, best_params

def experimento_1(param1, param2, n_episodios):
    env = gym.make('CartPole-v1')
    results = [] 

    for _ in tqdm(range(n_episodios), desc="Episódios de avaliação"):
        episode_data = run_episode_exp(env, param1, param2) 
        results.extend(episode_data)

    env.close()
    results = np.array(results)
    return results

def experimento_2(param1, param2, duration):
    env = gym.make('CartPole-v1')
    results = []

    for _ in tqdm(range(10), desc="Episódios de avaliação"):
        episode_data = run_episode_exp(env, param1, param2, max_steps=duration)
        results.extend(episode_data)

    env.close()
    results = np.array(results)
    return results

def experimento_3(param1, param2, noise_range=(-0.05, 0.05)):
    env = gym.make('CartPole-v1')
    results = []
    for _ in tqdm(range(10), desc="Episódios de avaliação"):
        episode_data = run_episode_exp(env, param1, param2, noise_range=noise_range)
        results.extend(episode_data)
    
    env.close()
    results = np.array(results)
    return results

def experimento_4(param1, param2, ranges=[(0, 0), (0, 0), (-0.2, 0.2), (-3.0, 3.0)]):
    env = gym.make('CartPole-v1')
    state = [np.random.uniform(low, high) for low, high in ranges]
    results = []
    for _ in tqdm(range(10), desc="Episódios de avaliação"):
        episode_data = run_episode_exp(env, param1, param2, custom_state=state)
        results.extend(episode_data)
    
    env.close()
    results = np.array(results)
    return results

def experimento_5(param1, param2, metodo='media'):
    env = gym.make('CartPole-v1')
    results = []
    
    for _ in tqdm(range(10), desc="Episódios de avaliação"):
        episode_data = run_episode_exp(env, param1, param2, metodo=metodo)
        results.extend(episode_data)

    env.close()
    results = np.array(results)
    return results

def experimento_6(param1, param2, n_episodios=10, pesos=(0.3, 0.4, 0.3)):
    env = gym.make('CartPole-v1')
    rewards = [run_episode(env, param1, param2) for _ in range(n_episodios)]
    env.close()

    min_r = np.min(rewards)
    mean_r = np.mean(rewards)
    max_r = np.max(rewards)

    w_min, w_mean, w_max = pesos
    fitness_final = w_min * min_r + w_mean * mean_r + w_max * max_r
    return fitness_final


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

def plot_results_otim_weights(results):

    X = results[:, 0]
    Y = results[:, 1]
    Z = results[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='viridis')

    ax.set_xlabel('param1 (peso pole angle)')
    ax.set_ylabel('param2 (peso angular velocity)')
    ax.set_zlabel('Recompensa Média')
    ax.set_title('Otimização de Parâmetros para CartPole-v1')

    plt.show()

def plot_results(results):
    X = results[:, 0] # eixo X: ângulo
    Y = results[:, 1] # eixo Y: velocidade angular
    Z = results[:, 2] # eixo Z: Fitness     

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Ângulo (a)')
    ax.set_ylabel('Velocidade Angular (av)')
    ax.set_zlabel('Recompensa Média')

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-3, 3)

    plt.show()
