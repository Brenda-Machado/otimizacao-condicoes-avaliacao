"""

Author: Brenda Silva Machado.

Evaluate

"""

import numpy as np
import time
from open_ai_es import OpenAIES
from policy import CartPolePolicy, PendulumPolicy

def optimize_policy(env_class, policy_class, 
                   generations=100,
                   population_size=20,
                   learning_rate=0.01,
                   noise_std=0.02,
                   weight_decay=0.0,
                   seed=42,
                   verbose=True):
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
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = policy.get_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
                
        total_reward += episode_reward
    
    return total_reward / num_episodes