"""
Análise de otimização para CartPoleV-1 para o TCC

Ação é determinada pela seguinte função: sign(param1_observation[2]+param2_observation[3])

Observation: cart position, cart velocity, pole angle, pole angular velocity

Ação é discreta, 0 ou 1, a função sign vai dar a ação automaticamente.

Variar os parâmetros (param1 e param2) de -2 a 2 de 0.025 em 0.025.
"""

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_action(obs, param1, param2):
    val = param1 * obs[2] + param2 * obs[3]
    return int(val > 0)

env = gym.make('CartPole-v1')

param_range = np.arange(-2, 2, 0.025)
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
