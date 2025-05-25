"""
Author: Brenda Silva Machado.

Env: CartPoleV-1.

Action function: sign(param1_observation[2]+param2_observation[3]).

Observation: cart position, cart velocity, pole angle, pole angular velocity

Possible otimizations:
    * Weights of the action function;

Otim TO-DO:
    * Todas as possíveis combinações condições iniciais;
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

def experimento_controle(param1, param2, step=0.25):
    results = {}
    env = gym.make('CartPole-v1')
    pos = np.arange(-2.4, 2.4 + step, step)
    vel = np.arange(-3.0, 3.0 + step, step)
    ang = np.arange(-0.21, 0.21 + step, step)
    ang_vel = np.arange(-3.0, 3.0 + step, step)

    rewards = []
    for p in tqdm(pos, desc="pos"):
        for v in vel:
            for a in ang:
                for av in ang_vel:
                    state = [p, v, a, av]
                    reward = run_episode(env, param1, param2, custom_state=state)
                    rewards.append(reward)
                    results[p,v,a,av] = reward
    env.close()

    best_state = max(results, key=results.get)
    best_reward = results[best_state]

    plot_results_controle(results)

    # print("\nMelhores condições iniciais encontradas:")
    # print(f"p = {best_state[0]}, v = {best_state[1]}, a = {best_state[2]}, av = {best_state[3]} --> recompensa = {best_reward}")

    return results, best_state, best_reward

def experimento_1(param1, param2, n_episodios=10):
    env = gym.make('CartPole-v1')
    rewards = [run_episode(env, param1, param2) for _ in range(n_episodios)]
    env.close()
    return np.mean(rewards)

def experimento_2(param1, param2, duration=500):
    env = gym.make('CartPole-v1')
    reward = run_episode(env, param1, param2, max_steps=duration)
    env.close()
    return reward

def experimento_3(param1, param2, noise_range=(-0.05, 0.05)):
    env = gym.make('CartPole-v1')
    reward = run_episode(env, param1, param2, noise_range=noise_range)
    env.close()
    return reward

def experimento_4(param1, param2, ranges=[(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.0, 3.0)]):
    env = gym.make('CartPole-v1')
    state = [np.random.uniform(low, high) for low, high in ranges]
    reward = run_episode(env, param1, param2, custom_state=state)
    env.close()
    return reward

def experimento_5(param1, param2, n_episodios=10, metodo='media'):
    env = gym.make('CartPole-v1')
    rewards = [run_episode(env, param1, param2) for _ in range(n_episodios)]
    env.close()

    if metodo == 'media':
        return np.mean(rewards)
    elif metodo == 'max':
        return np.max(rewards)
    elif metodo == 'min':
        return np.min(rewards)
    else:
        raise ValueError("Método deve ser 'media', 'max' ou 'min'")

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

def plot_results_controle(results):
    xs, ys, zs, cs = [], [], [], []

    for (p, v, a, av), reward in results.items():
        xs.append(p)     # eixo X: posição
        ys.append(a)     # eixo Y: ângulo
        zs.append(av)    # eixo Z: velocidade angular
        cs.append(reward)  # cor: recompensa

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, c=cs, cmap='viridis', marker='o')
    plt.colorbar(sc, ax=ax, label='Recompensa')

    ax.set_xlabel('Posição (p)')
    ax.set_ylabel('Ângulo (a)')
    ax.set_zlabel('Velocidade Angular (av)')
    ax.set_title('Recompensa em função do Estado Inicial no CartPole')

    plt.show()
