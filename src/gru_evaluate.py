"""
Policy Evaluation with GRU-based Neural Controller

Uses advanced GRU architecture based on:
Pagliuca, P., Milano, N., & Nolfi, S. (2020). "Efficacy of modern 
neuroevolutionary strategies for continuous control optimization". 
Frontiers in Robotics and AI, 7, 98.

Author: Brenda Silva Machado

gru_evaluate.py
"""

import numpy as np
import time
import pickle
from open_ai_es import OpenAIES
from gru_policy import CartPolePolicy, PendulumPolicy
from pendulum import PendulumEnv 
from cartpole import CartPoleEnv

def optimize_policy(env_class, policy_class, 
                   generations=1000,
                   population_size=101,
                   learning_rate=0.01,
                   noise_std=0.02,
                   weight_decay=0.0,
                   seed=42,
                   verbose=True,
                   num_episodes=2,
                   max_steps=None,
                   theta_range=None,
                   theta_dot_range=None,
                   motor_noise=None):
    
    env = env_class()
    policy = policy_class()
    
    es = OpenAIES(
        param_count=policy.get_param_count(),
        population_size=population_size,
        learning_rate=learning_rate,
        noise_std=noise_std,
        weight_decay=weight_decay,
        seed=seed
    )
    
    fitness_history = []
    start_time = time.time()
    
    if verbose:
        print(f"OpenAI-ES Optimization with GRU Controller")
        print(f"  Seed: {seed}")
        print(f"  Generations: {generations}")
        print(f"  Population size: {population_size*2} (ES: +/- perturbations)")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Noise std: {noise_std}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Policy parameters: {policy.get_param_count()}")
        print(f"  Policy type: {policy.__class__.__name__}")
        
        if theta_range is not None:
            print(f"  Theta range: ±{theta_range:.4f}")
        if theta_dot_range is not None:
            print(f"  Theta_dot range: ±{theta_dot_range:.4f}")
        if motor_noise is not None:
            print(f"  Motor noise: {motor_noise:.4f}")
        print(f"  Episodes per evaluation: {num_episodes}")
        if max_steps is not None:
            print(f"  Max steps per episode: {max_steps}")
    
    for generation in range(generations):
        samples, noise = es.ask()
        
        fitness_list = []
        for params in samples:
            policy.set_params(params)
            fitness = evaluate_policy(env, policy, num_episodes, max_steps, 
                                    theta_range, theta_dot_range, motor_noise)
            fitness_list.append(fitness)
        
        es.tell(fitness_list, noise)
        
        avg_fitness = np.mean(fitness_list)
        max_fitness = np.max(fitness_list)
        fitness_history.append({
            'generation': generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'best_fitness': es.best_fitness
        })
        
        if verbose and (generation % 10 == 0 or generation == generations - 1):
            elapsed = time.time() - start_time
            print(f"Gen {generation:4d} | avg_fitness={avg_fitness:8.2f} | "
                  f"max_fitness={max_fitness:8.2f} | best={es.best_fitness:8.2f} | "
                  f"time={elapsed:7.1f}s")
    
    if verbose:
        total_time = time.time() - start_time

        print(f"Optimization completed in {total_time:.1f}s")
        print(f"Best fitness achieved: {es.best_fitness:.2f}")
    
    return es.get_best_params(), fitness_history


def evaluate_policy(env, policy, num_episodes=2, max_steps=1000, 
                   theta_range=None, theta_dot_range=None, motor_noise=None):
    total_reward = 0
    
    for episode in range(num_episodes):
        policy.controller.reset_hidden_state()
        
        if theta_range is not None or theta_dot_range is not None:

            if theta_range is not None:
                theta = np.random.uniform(-theta_range, theta_range)
            else:
                theta = np.random.uniform(-np.pi, np.pi)
                
            if theta_dot_range is not None:
                theta_dot = np.random.uniform(-theta_dot_range, theta_dot_range)
            else:
                theta_dot = np.random.uniform(-1.0, 1.0)
 
            if hasattr(env, 'reset_custom'):
                if isinstance(env, PendulumEnv):
                    custom_state = [theta, theta_dot]
                elif isinstance(env, CartPoleEnv):
                    custom_state = [theta_dot, theta]
                else:
                    custom_state = [theta, theta_dot]
                
                result = env.reset_custom(custom_state=custom_state)
            else:
                result = env.reset()
        else:
            result = env.reset()
        
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        
        episode_reward = 0
        
        for step in range(max_steps):
            action = policy.get_action(obs)
            
            if motor_noise is not None:
                if isinstance(action, (int, np.integer)):
                    action_float = float(action) + np.random.normal(0, motor_noise)
                    action = int(np.clip(np.round(action_float), 0, 1))
                else:
                    action = action + np.random.normal(0, motor_noise)
                    action = np.clip(action, -2.0, 2.0)
            
            step_result = env.step(action)
            
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            episode_reward += reward
            
            if done or terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def run_pendulum_evolution(theta_range=None, theta_dot_range=None, motor_noise=None, 
                          generations=1000, verbose=True):

    best_params, history = optimize_policy(
        env_class=PendulumEnv,
        policy_class=PendulumPolicy,
        generations=generations,
        population_size=101,
        learning_rate=0.02,
        noise_std=0.05,
        seed=42,
        verbose=verbose,
        num_episodes=17,
        max_steps=999,
        theta_range=1.6039171496925475,
        theta_dot_range=3.8240345672052407,
        motor_noise=0.109262991047962
    )
    
    return best_params, history


def run_cartpole_evolution(theta_range=None, theta_dot_range=None, motor_noise=None,
                          generations=1000, verbose=True):

    best_params, history = optimize_policy(
        env_class=CartPoleEnv,
        policy_class=CartPolePolicy,
        generations=generations,
        population_size=101,
        learning_rate=0.05,
        noise_std=0.1,
        seed=123,
        verbose=verbose,
        num_episodes=48,
        max_steps=653,
        theta_range=1.0619792925512106,
        theta_dot_range=7.203153416131942,
        motor_noise=0.030988660252392
    )
    
    return best_params, history


def save_results(pendulum_params, cartpole_params, pendulum_history, cartpole_history, 
                filename_suffix=""):

    np.save(f'pendulum_best_params{filename_suffix}.npy', pendulum_params)
    np.save(f'cartpole_best_params{filename_suffix}.npy', cartpole_params)
    
    with open(f'evolution_history{filename_suffix}.pkl', 'wb') as f:
        pickle.dump({
            'pendulum': pendulum_history,
            'cartpole': cartpole_history
        }, f)
    

def run_experiment_with_conditions(theta_range_p=None, theta_dot_range_p=None, 
                                  motor_noise_p=None,
                                  theta_range_c=None, theta_dot_range_c=None,
                                  motor_noise_c=None,
                                  experiment_name="default",
                                  generations=1000):    
    start_time = time.time()
    
    print("Running Pendulum optimization...")
    pendulum_params, pendulum_history = run_pendulum_evolution(
        theta_range=theta_range_p,
        theta_dot_range=theta_dot_range_p,
        motor_noise=motor_noise_p,
        generations=generations,
        verbose=True
    )
    
    print("\nRunning CartPole optimization...")
    cartpole_params, cartpole_history = run_cartpole_evolution(
        theta_range=theta_range_c,
        theta_dot_range=theta_dot_range_c,
        motor_noise=motor_noise_c,
        generations=generations,
        verbose=True
    )
    
    filename_suffix = f"_{experiment_name}" if experiment_name != "default" else ""
    save_results(pendulum_params, cartpole_params, pendulum_history, cartpole_history,
                filename_suffix)
    
    total_time = time.time() - start_time

    print(f"Experiment {experiment_name} completed")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Pendulum - Best fitness: {pendulum_history[-1]['best_fitness']:.2f}")
    print(f"CartPole - Best fitness: {cartpole_history[-1]['best_fitness']:.2f}")

def main():
    run_experiment_with_conditions(
        theta_range_p=1.725973,
        theta_dot_range_p=1.369452,
        motor_noise_p=0.089886,
        theta_range_c=0.725378,
        theta_dot_range_c=3.218251,
        motor_noise_c=0.113388,
        experiment_name="baseline_gru",
        generations=1000
    )


if __name__ == "__main__":
    main()