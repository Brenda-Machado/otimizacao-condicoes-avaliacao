"""
Experiments to evaluate the impact of the Evaluation Conditions have on the fitness landscape.

Env: CartPole-v0.
Action function: sign(param1_observation[2]+param2_observation[3]).
Observation: cart position, cart velocity, pole angle, pole angular velocity.

Author: Brenda Silva Machado.

cartpole_experiments.py
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from policy import Policy
from tqdm import tqdm
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

def experimento_controle():
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

def experimento_1_n_episodios():
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    episodios = [2, 5, 10, 15, 20, 50]
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)

    for ep in episodios:
        print(f"Experimento 1: n_episodes = {ep}")

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
            
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_1/fitness_landscape_ep_{ep}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        plot_results(results=results, exp='exp_1', name=f'ep_{ep}')

def experimento_2_duracao():
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    duracao = [50, 100, 200, 300, 400, 500]
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)

    for d in duracao:
        print(f"Experimento 2: maxsteps = {d}")

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
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_2/fitness_landscape_d_{d}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'mstep_' + str(d)
        plot_results(results=results, exp='exp_2', name=name)

def experimento_3_ruido():
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    noise = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    theta = np.linspace(-3, 3, num=int((3 - (-3)) / step) + 1)
    theta_dot = np.linspace(-0.2, 0.2, num=int((0.2 - (-0.2)) / step) + 1)

    for n in noise:
        print(f"Experimento 3: noise = {n}")

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
                
        env.close()
        results = np.array(results)

        path = os.path.expanduser(f'~/tcc/data/cartpole/exp_3/fitness_landscape_n_{str(n)}.npy')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, results)

        name = 'noise_' + str(n)
        plot_results(results=results, exp='exp_3', name=name)

def experimento_4_condicoes():
    step = 0.025
    env = CartPoleEnv()
    policy = Policy(input_size=4)
    interval_ranges_theta = [3, 2, 1]
    interval_ranges_theta_dot = [0.2, 0.1, 0.05]

    for t_range in interval_ranges_theta:
        for td_range in interval_ranges_theta_dot:

            print(f"Experimento 4: ranges = theta±{t_range}, theta_dot±{td_range}")
            
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

            env.close()
            results = np.array(results)

            ranges = f"{td_range}_{t_range}"
            path = os.path.expanduser(f'~/tcc/data/cartpole/exp_4/fitness_landscape_r_{ranges}.npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, results)

            name = 'ranges_' + ranges
            plot_results(results=results, exp='exp_4', name=name)


def plot_results(results, exp, name):
    import matplotlib
    matplotlib.use('Agg')
    
    X = results[:, 0] # eixo X: ângulo
    Y = results[:, 1] # eixo Y: velocidade angular
    Z = results[:, 2] # eixo Z: Fitness   

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    n_points = len(X)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([-3, 3])         
    ax.set_ylim([-0.2, 0.2])      
    ax.set_zlim([0, 500])

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