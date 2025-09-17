"""

Author: Brenda Silva Machado.

Evaluate

"""

import numpy as np
import time
from open_ai_es import OpenAIES
from policy import CartPolePolicy, PendulumPolicy
from pendulum import PendulumEnv 
from cartpole import CartPoleEnv

def optimize_policy(env_class, policy_class, 
                   generations=100,
                   population_size=20,
                   learning_rate=0.01,
                   noise_std=0.02,
                   weight_decay=0.0,
                   seed=42,
                   verbose=True,
                   num_episodes=3,
                   max_steps=None):
    
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
        print(f"OpenAI-ES: seed {seed}, generations {generations}, pop_size {population_size*2}")
        print(f"Parameters: lr={learning_rate}, noise_std={noise_std}, param_count={policy.get_param_count()}")
        print("-" * 80)
    
    for generation in range(generations):
        samples, noise = es.ask()
        
        fitness_list = []
        for params in samples:
            policy.set_params(params)
            fitness = evaluate_policy(env, policy)
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
            print(f"Gen {generation:3d}: avg={avg_fitness:7.2f}, max={max_fitness:7.2f}, "
                  f"best={es.best_fitness:7.2f}, time={elapsed:.1f}s")
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.1f}s")
        print(f"Best fitness: {es.best_fitness:.2f}")
    
    return es.get_best_params(), fitness_history


def evaluate_policy(env, policy, num_episodes=1, max_steps=1000):
    total_reward = 0
    
    for episode in range(num_episodes):
        result = env.reset()
        
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        
        episode_reward = 0
        
        for step in range(max_steps):
            action = policy.get_action(obs)
            
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

def run_pendulum_evolution():

    best_params, history = optimize_policy(
        env_class=PendulumEnv,
        policy_class=PendulumPolicy,
        generations=80,
        population_size=15,
        learning_rate=0.02,
        noise_std=0.05,
        seed=42,
        num_episodes=3,
        max_steps=200
    )
    
    return best_params, history


def run_cartpole_evolution():
    
    best_params, history = optimize_policy(
        env_class=CartPoleEnv,
        policy_class=CartPolePolicy,
        generations=50,
        population_size=20,
        learning_rate=0.05,
        noise_std=0.1,
        seed=123,
        num_episodes=2,
        max_steps=500
    )
    
    return best_params, history


def save_results(pendulum_params, cartpole_params, pendulum_history, cartpole_history):
    np.save('pendulum_best_params.npy', pendulum_params)
    np.save('cartpole_best_params.npy', cartpole_params)
    
    import pickle
    with open('evolution_history.pkl', 'wb') as f:
        pickle.dump({
            'pendulum': pendulum_history,
            'cartpole': cartpole_history
        }, f)
        
        

def main():
    start_time = time.time()
    pendulum_params, pendulum_history = run_pendulum_evolution()
    cartpole_params, cartpole_history = run_cartpole_evolution()
    
    save_results(pendulum_params, cartpole_params, pendulum_history, cartpole_history)
    
    total_time = time.time() - start_time
    print(f"Tempo total: {total_time:.1f}s")
    print(f"Pendulum - Melhor fitness: {pendulum_history[-1]['best_fitness']:.2f}")
    print(f"CartPole - Melhor fitness: {cartpole_history[-1]['best_fitness']:.2f}")

if __name__ == "__main__":
    main()

