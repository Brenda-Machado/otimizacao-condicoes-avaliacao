"""
Run irace to optimize OpenAI-ES Evaluation Conditions with GRU Policy

Baseado em Brenda Silva Machado - adaptado para usar Policy GRU avançada

Based on: Pagliuca, P., Milano, N., & Nolfi, S. (2020). "Efficacy of modern 
neuroevolutionary strategies for continuous control optimization". 
Frontiers in Robotics and AI, 7, 98.

Author: Brenda Silva Machado
"""

import numpy as np
import time
from irace import irace, ParameterSpace, Scenario, Experiment
from irace.params import Real, Integer
from open_ai_es import OpenAIES
from advanced_policy import PendulumPolicy, CartPolePolicy
from pendulum import PendulumEnv
from cartpole import CartPoleEnv


class ParameterizedEnvironment:
    def __init__(self, base_env_class):
        self.base_env_class = base_env_class
        self.base_env = None
        self.current_params = {}

    def set_parameters(self, params_dict):
        self.current_params = params_dict
        self.base_env = self.base_env_class()
        for param, value in params_dict.items():
            if hasattr(self.base_env, param):
                setattr(self.base_env, param, value)

    def reset(self, seed=None):
        if self.base_env is None:
            raise ValueError("Parâmetros não foram definidos! Chame set_parameters() primeiro.")
        return self.base_env.reset(seed=seed)

    def step(self, action):
        return self.base_env.step(action)

    def reset_custom(self, custom_state=None, custom_bounds=None):
        if hasattr(self.base_env, 'reset_custom'):
            return self.base_env.reset_custom(custom_state=custom_state, 
                                             custom_bounds=custom_bounds)
        return self.reset()

    def __getattr__(self, name):
        if self.base_env is not None:
            return getattr(self.base_env, name)
        else:
            raise AttributeError(f"Ambiente não inicializado")


def evaluate_configuration(env_class, policy_class, params_dict, 
                         es_generations=30, es_population_size=10, verbose=False):
    param_env = ParameterizedEnvironment(env_class)
    param_env.set_parameters({})
    
    policy = policy_class()
    
    es = OpenAIES(
        param_count=policy.get_param_count(),
        population_size=es_population_size,
        learning_rate=0.02,
        noise_std=0.05,
        seed=42
    )
    
    best_fitness = -np.inf
    num_episodes = int(params_dict.get('num_episodes', 3))
    maxsteps = int(params_dict.get('maxsteps', 500))
    theta_range = params_dict.get('theta_range', None)
    theta_dot_range = params_dict.get('theta_dot_range', None)
    motor_noise = params_dict.get('motor_noise', None)
    
    for generation in range(es_generations):
        samples, noise = es.ask()
        fitness_list = []
        
        for params in samples:
            policy.set_params(params)
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
                    
                    if isinstance(env_class, type) and issubclass(env_class, PendulumEnv):
                        custom_state = [theta, theta_dot]
                    elif isinstance(env_class, type) and issubclass(env_class, CartPoleEnv):
                        custom_state = [theta_dot, theta]
                    else:
                        custom_state = [theta, theta_dot]
                    
                    try:
                        obs, _ = param_env.reset_custom(custom_state=custom_state)
                    except:
                        obs, _ = param_env.reset()
                else:
                    obs, _ = param_env.reset()
                
                episode_reward = 0
                
                for step in range(maxsteps):
                    action = policy.get_action(obs)
                    
                    if motor_noise is not None:
                        if isinstance(action, (int, np.integer)):
                            action_float = float(action) + np.random.normal(0, motor_noise)
                            action = int(np.clip(np.round(action_float), 0, 1))
                        else:
                            action = action + np.random.normal(0, motor_noise)
                            action = np.clip(action, -2.0, 2.0)
                    
                    result = param_env.step(action)
                    
                    if len(result) == 5:
                        obs, reward, terminated, truncated, _ = result
                    elif len(result) == 4:
                        obs, reward, terminated, _ = result
                        truncated = terminated
                    else:
                        raise ValueError(f"Resultado inesperado com {len(result)} elementos")
                    
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                total_reward += episode_reward
            
            avg_reward = total_reward / num_episodes
            fitness_list.append(avg_reward)
        
        es.tell(fitness_list, noise)
        best_fitness = max(best_fitness, max(fitness_list))
        
        if verbose and generation % 10 == 0:
            print(f"  Gen {generation}: best={max(fitness_list):.3f}")
    
    return best_fitness


def target_runner_pendulum(experiment: Experiment, scenario: Scenario) -> float:
    params = experiment.configuration
    params_dict = {
        'num_episodes': int(params['num_episodes']),
        'maxsteps': int(params['maxsteps']),
        'motor_noise': float(params['motor_noise']),
        'theta_range': float(params['theta_range']),
        'theta_dot_range': float(params['theta_dot_range'])
    }
    
    if scenario.verbose >= 200:
        print(f"Avaliando configuração Pendulum (GRU): {params_dict}")
    
    try:
        fitness = evaluate_configuration(
            PendulumEnv, 
            PendulumPolicy, 
            params_dict,
            es_generations=25,
            es_population_size=8,
            verbose=scenario.verbose >= 300
        )
        
        if scenario.verbose >= 200:
            print(f"Fitness obtido: {fitness:.3f}")
        
        return fitness
        
    except Exception as e:
        print(f"Erro na avaliação: {e}")
        import traceback
        traceback.print_exc()
        return -1000.0


def target_runner_cartpole(experiment: Experiment, scenario: Scenario) -> float:
    params = experiment.configuration
    params_dict = {
        'num_episodes': int(params['num_episodes']),
        'maxsteps': int(params['maxsteps']),
        'motor_noise': float(params['motor_noise']),
        'theta_range': float(params['theta_range']),
        'theta_dot_range': float(params['theta_dot_range'])
    }
    
    if scenario.verbose >= 200:
        print(f"Avaliando configuração CartPole (GRU): {params_dict}")
    
    try:
        fitness = evaluate_configuration(
            CartPoleEnv, 
            CartPolePolicy, 
            params_dict,
            es_generations=20,
            es_population_size=10,
            verbose=scenario.verbose >= 300
        )
        
        if scenario.verbose >= 200:
            print(f"Fitness obtido: {fitness:.3f}")
        
        return fitness
        
    except Exception as e:
        print(f"Erro na avaliação: {e}")
        import traceback
        traceback.print_exc()
        return -500.0


def optimize_pendulum_parameters():
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO PENDULUM - PYIRACE (GRU Policy)")
    print("=" * 60)
    
    parameter_space = ParameterSpace([
        Integer('num_episodes', 1, 50),
        Integer('maxsteps', 100, 1000),
        Real('motor_noise', 0.001, 0.2),
        Real('theta_range', 0.1, 4.0),
        Real('theta_dot_range', 0.5, 8.0),
    ])
    
    scenario = Scenario(
        max_experiments=200,
        verbose=100,
        seed=42
    )
    
    print("Espaço de parâmetros:")
    print(parameter_space)
    print(f"\nExecutando {scenario.max_experiments} experimentos...")
    
    start_time = time.time()
    result = irace(target_runner_pendulum, parameter_space, scenario, return_df=True)
    elapsed_time = time.time() - start_time
    
    print(f"\nOtimização concluída em {elapsed_time:.1f}s")
    print("\nMelhores configurações encontradas:")
    print(result.head(10))
    
    result.to_csv('pendulum_irace_results_gru.csv', index=False)
    print("\nResultados salvos em 'pendulum_irace_results_gru.csv'")
    
    return result


def optimize_cartpole_parameters():
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO CARTPOLE - PYIRACE (GRU Policy)")
    print("=" * 60)
    
    parameter_space = ParameterSpace([
        Integer('num_episodes', 1, 50),
        Integer('maxsteps', 100, 1000),
        Real('motor_noise', 0.001, 0.2),
        Real('theta_range', 0.05, 4.0),
        Real('theta_dot_range', 0.5, 8.0),
    ])
    
    scenario = Scenario(
        max_experiments=500,
        verbose=100,
        seed=123
    )
    
    print("Espaço de parâmetros:")
    print(parameter_space)
    print(f"\nExecutando {scenario.max_experiments} experimentos...")
    
    start_time = time.time()
    result = irace(target_runner_cartpole, parameter_space, scenario, return_df=True)
    elapsed_time = time.time() - start_time
    
    print(f"\nOtimização concluída em {elapsed_time:.1f}s")
    print("\nMelhores configurações encontradas:")
    print(result.head(10))
    
    result.to_csv('cartpole_irace_results_gru.csv', index=False)
    print("\nResultados salvos em 'cartpole_irace_results_gru.csv'")
    
    return result


def test_best_configuration(results_df, env_class, policy_class, env_name):
    best_config = results_df.iloc[0]
    
    print(f"\n{env_name} - Melhor configuração:")
    for param in ['num_episodes', 'maxsteps', 'motor_noise', 'theta_range', 'theta_dot_range']:
        if param in best_config:
            print(f"  {param}: {best_config[param]:.4f}")
    
    print(f"Fitness da melhor configuração: {best_config.get('cost', 'N/A')}")

    params_dict = {
        'num_episodes': int(best_config['num_episodes']),
        'maxsteps': int(best_config['maxsteps']),
        'motor_noise': float(best_config['motor_noise']),
        'theta_range': float(best_config['theta_range']),
        'theta_dot_range': float(best_config['theta_dot_range'])
    }

    print(f"\nValidando {env_name} com otimização estendida...")
    detailed_fitness = evaluate_configuration(
        env_class, 
        policy_class, 
        params_dict,
        es_generations=50, 
        es_population_size=15,
        verbose=True
    )
    
    return params_dict, detailed_fitness


def main():
    start_total = time.time()
    
    pendulum_results = optimize_pendulum_parameters()
    cartpole_results = optimize_cartpole_parameters()
    
    pendulum_best, pendulum_fitness = test_best_configuration(
        pendulum_results, PendulumEnv, PendulumPolicy, "Pendulum"
    )
    
    cartpole_best, cartpole_fitness = test_best_configuration(
        cartpole_results, CartPoleEnv, CartPolePolicy, "CartPole"
    )
    
    total_time = time.time() - start_total
    
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"Tempo total: {total_time:.1f}s")
    
    print(f"\nPendulum - Melhor fitness: {pendulum_fitness:.3f}")
    print("Melhores parâmetros:")
    for param, value in pendulum_best.items():
        print(f"  {param}: {value}")
    
    print(f"\nCartPole - Melhor fitness: {cartpole_fitness:.3f}")
    print("Melhores parâmetros:")
    for param, value in cartpole_best.items():
        print(f"  {param}: {value}")
    

if __name__ == '__main__':
    main()