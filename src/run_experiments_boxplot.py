"""
Multi-Seed Experiment Runner for Statistical Comparison (Parallelized)

Executa 10 replicações (seeds) de cada algoritmo para análise estatística robusta.
Gera boxplots, testes estatísticos e análise de significância.
Usa multiprocessing para executar seeds em paralelo.

Author: Brenda Silva Machado
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from scipy import stats
from multiprocessing import Pool, cpu_count
import os
from evaluate_advanced import optimize_policy
from adaptative_algo_opt import AdaptiveES
from pendulum import PendulumEnv
from cartpole import CartPoleEnv
from advanced_policy import PendulumPolicy, CartPolePolicy


class MultiSeedExperiment:
    def __init__(self, num_seeds=10, n_jobs=None):
        self.num_seeds = num_seeds
        self.seeds = list(range(42, 42 + num_seeds))  # Seeds: 42, 43, ..., 51
        self.n_jobs = n_jobs if n_jobs is not None else max(1, cpu_count() - 1)
        self.results = {
            'pendulum': {'baseline': [], 'adaptive': []},
            'cartpole': {'baseline': [], 'adaptive': []}
        }
        self.histories = {
            'pendulum': {'baseline': [], 'adaptive': []},
            'cartpole': {'baseline': [], 'adaptive': []}
        }
        
        print(f"Paralelização: {self.n_jobs} processos (de {cpu_count()} CPUs disponíveis)")
    
    def run_baseline(self, env_class, policy_class, env_name, generations=1000,
                     population_size=101, learning_rate=0.02, noise_std=0.05,
                     num_episodes=26, max_steps=980, **env_params):
        print(f"\n{'='*80}")
        print(f"BASELINE - {env_name.upper()} - {self.num_seeds} SEEDS (PARALELO)")
        print(f"{'='*80}\n")
        
        # Prepare arguments for each seed
        args_list = []
        for seed in self.seeds:
            args = {
                'env_class': env_class,
                'policy_class': policy_class,
                'seed': seed,
                'generations': generations,
                'population_size': population_size,
                'learning_rate': learning_rate,
                'noise_std': noise_std,
                'num_episodes': num_episodes,
                'max_steps': max_steps,
                'env_params': env_params
            }
            args_list.append(args)
        
        start_time = time.time()
        
        with Pool(processes=self.n_jobs) as pool:
            results_list = pool.map(_run_baseline_single_seed, args_list)
        
        total_time = time.time() - start_time
        
        results = []
        histories = []

        for seed, (fitness, history) in zip(self.seeds, results_list):
            if fitness is not None:
                results.append(fitness)
                histories.append(history)
                print(f"✓ Seed {seed}: fitness={fitness:.2f}")
            else:
                results.append(np.nan)
                histories.append(None)
                print(f"✗ Seed {seed}: FALHOU")
        
        self.results[env_name]['baseline'] = results
        self.histories[env_name]['baseline'] = histories
        
        print(f"\nTempo total (paralelo): {total_time:.1f}s")
        self._print_summary(results, f"Baseline {env_name}")

        return results, histories
    
    def run_adaptive(self, env_class, policy_class, param_bounds, env_name,
                    es_generations=1000, es_population=5, agent_generations=1,
                    learning_rate=0.02, noise_std=0.05, weight_decay=0.0):

        print(f"\n{'='*80}")
        print(f"ADAPTATIVO - {env_name.upper()} - {self.num_seeds} SEEDS (PARALELO)")
        print(f"{'='*80}\n")
        
        args_list = []

        for seed in self.seeds:
            args = {
                'env_class': env_class,
                'policy_class': policy_class,
                'param_bounds': param_bounds,
                'seed': seed,
                'es_generations': es_generations,
                'es_population': es_population,
                'agent_generations': agent_generations,
                'learning_rate': learning_rate,
                'noise_std': noise_std,
                'weight_decay': weight_decay
            }
            args_list.append(args)
        
        start_time = time.time()
        
        with Pool(processes=self.n_jobs) as pool:
            results_list = pool.map(_run_adaptive_single_seed, args_list)
        
        total_time = time.time() - start_time
        results = []
        histories = []

        for seed, (fitness, history, best_params) in zip(self.seeds, results_list):
            if fitness is not None:
                results.append(fitness)
                histories.append(history)
                print(f"✓ Seed {seed}: fitness={fitness:.2f}")
                print(f"  Params: {best_params}")
            else:
                results.append(np.nan)
                histories.append(None)
                print(f"✗ Seed {seed}: FALHOU")
        
        self.results[env_name]['adaptive'] = results
        self.histories[env_name]['adaptive'] = histories
        
        print(f"\nTempo total (paralelo): {total_time:.1f}s")
        self._print_summary(results, f"Adaptativo {env_name}")

        return results, histories
    
    def _print_summary(self, results, name):
        valid_results = [r for r in results if not np.isnan(r)]
        
        print(f"\n{'='*80}")
        print(f"RESUMO: {name}")
        print(f"{'='*80}")
        print(f"Média: {np.mean(valid_results):.2f}")
        print(f"Mediana: {np.median(valid_results):.2f}")
        print(f"Desvio padrão: {np.std(valid_results):.2f}")
        print(f"Mínimo: {np.min(valid_results):.2f}")
        print(f"Máximo: {np.max(valid_results):.2f}")
        print(f"Seeds válidas: {len(valid_results)}/{len(results)}")
        print(f"{'='*80}\n")
    
    def save_results(self, filepath='multi_seed_results.pkl'):
        data = {
            'seeds': self.seeds,
            'results': self.results,
            'histories': self.histories
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_results(self, filepath='multi_seed_results.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.seeds = data['seeds']
        self.results = data['results']
        self.histories = data['histories']
    
    def plot_boxplots(self, save_prefix='boxplot'):
        for env_name in ['pendulum', 'cartpole']:
            baseline_results = [r for r in self.results[env_name]['baseline'] if not np.isnan(r)]
            adaptive_results = [r for r in self.results[env_name]['adaptive'] if not np.isnan(r)]
            
            if not baseline_results or not adaptive_results:
                print(f"Dados insuficientes para {env_name}")
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            ax = axes[0]
            
            data = [baseline_results, adaptive_results]
            labels = ['Baseline\n(Fixo)', 'Adaptativo\n(ES)']
            colors = ['#3498db', '#e74c3c']
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                           widths=0.6, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='yellow', 
                                        markeredgecolor='black', markersize=8))
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            for whisker in bp['whiskers']:
                whisker.set(linewidth=1.5)
            
            for cap in bp['caps']:
                cap.set(linewidth=1.5)
            
            for median in bp['medians']:
                median.set(color='black', linewidth=2)
            
            ax.set_ylabel('Fitness Final', fontsize=12, fontweight='bold')
            ax.set_title(f'{env_name.capitalize()} - Comparação de Performance\n(10 seeds)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, (results, color) in enumerate(zip(data, colors), 1):
                y = results
                x = np.random.normal(i, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.4, s=30, color=color, edgecolors='black', linewidth=0.5)
            
            ax = axes[1]
            ax.axis('off')
            
            if len(baseline_results) >= 3 and len(adaptive_results) >= 3:
                statistic, p_value = stats.mannwhitneyu(baseline_results, adaptive_results,
                                                        alternative='two-sided')

                mean_diff = np.mean(adaptive_results) - np.mean(baseline_results)
                pooled_std = np.sqrt((np.var(baseline_results) + np.var(adaptive_results)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            else:
                p_value = np.nan
                cohens_d = np.nan
            
            summary_text = f"ANÁLISE ESTATÍSTICA\n\n"
            summary_text += f"{'─'*40}\n"
            summary_text += f"BASELINE (Fixo)\n"
            summary_text += f"{'─'*40}\n"
            summary_text += f"Média:     {np.mean(baseline_results):8.2f}\n"
            summary_text += f"Mediana:   {np.median(baseline_results):8.2f}\n"
            summary_text += f"Std:       {np.std(baseline_results):8.2f}\n"
            summary_text += f"Min:       {np.min(baseline_results):8.2f}\n"
            summary_text += f"Max:       {np.max(baseline_results):8.2f}\n\n"
            
            summary_text += f"{'─'*40}\n"
            summary_text += f"ADAPTATIVO (ES)\n"
            summary_text += f"{'─'*40}\n"
            summary_text += f"Média:     {np.mean(adaptive_results):8.2f}\n"
            summary_text += f"Mediana:   {np.median(adaptive_results):8.2f}\n"
            summary_text += f"Std:       {np.std(adaptive_results):8.2f}\n"
            summary_text += f"Min:       {np.min(adaptive_results):8.2f}\n"
            summary_text += f"Max:       {np.max(adaptive_results):8.2f}\n\n"
            
            summary_text += f"{'─'*40}\n"
            summary_text += f"COMPARAÇÃO\n"
            summary_text += f"{'─'*40}\n"
            summary_text += f"Δ Média:   {mean_diff:8.2f}\n"
            
            if not np.isnan(p_value):
                summary_text += f"p-value:   {p_value:8.4f}\n"
                
                if p_value < 0.001:
                    sig = "*** (p < 0.001)"
                elif p_value < 0.01:
                    sig = "** (p < 0.01)"
                elif p_value < 0.05:
                    sig = "* (p < 0.05)"
                else:
                    sig = "n.s. (p ≥ 0.05)"
                
                summary_text += f"Signif.:   {sig}\n"
                summary_text += f"Cohen's d: {cohens_d:8.3f}\n\n"
                
                if abs(cohens_d) < 0.2:
                    effect = "Pequeno"
                elif abs(cohens_d) < 0.5:
                    effect = "Médio"
                elif abs(cohens_d) < 0.8:
                    effect = "Grande"
                else:
                    effect = "Muito Grande"
                
                summary_text += f"Tamanho do efeito: {effect}\n"
            
            ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            
            save_path = f"{save_prefix}_{env_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Boxplot salvo: {save_path}")
            
            plt.show()


def run_full_experiment_pendulum():
    experiment = MultiSeedExperiment(num_seeds=10)
    experiment.run_baseline(
        env_class=PendulumEnv,
        policy_class=PendulumPolicy,
        env_name='pendulum',
        generations=1000,
        population_size=101,
        learning_rate=0.02,
        noise_std=0.05,
        num_episodes=26,
        max_steps=980,
        theta_range=1.725973,
        theta_dot_range=1.369452,
        motor_noise=0.089886
    )
    
    param_bounds = {
        'n_episodes': (2, 50),
        'max_steps': (100, 1000),
        'noise_level': (0.001, 0.1),
        'theta_range': (1.0, 3.14),
        'theta_dot_range': (0.01, 2.0)
    }
    
    experiment.run_adaptive(
        env_class=PendulumEnv,
        policy_class=PendulumPolicy,
        param_bounds=param_bounds,
        env_name='pendulum',
        es_generations=1000,
        es_population=5,
        agent_generations=1,
        learning_rate=0.02,
        noise_std=0.05,
        weight_decay=0.0
    )
    
    experiment.save_results('pendulum_multi_seed.pkl')
    experiment.plot_boxplots(save_prefix='pendulum_boxplot')
    
    return experiment


def _run_baseline_single_seed(args):
    try:
        env_class = args['env_class']
        policy_class = args['policy_class']
        seed = args['seed']
        generations = args['generations']
        population_size = args['population_size']
        learning_rate = args['learning_rate']
        noise_std = args['noise_std']
        num_episodes = args['num_episodes']
        max_steps = args['max_steps']
        env_params = args['env_params']
        
        print(f"[Seed {seed}] Iniciando baseline...")
        
        best_params, history = optimize_policy(
            env_class=env_class,
            policy_class=policy_class,
            generations=generations,
            population_size=population_size,
            learning_rate=learning_rate,
            noise_std=noise_std,
            seed=seed,
            verbose=False, 
            num_episodes=num_episodes,
            max_steps=max_steps,
            **env_params
        )
        
        final_fitness = history[-1]['best_fitness']
        print(f"[Seed {seed}] Baseline concluído: {final_fitness:.2f}")
        
        return (final_fitness, history)
        
    except Exception as e:
        print(f"[Seed {seed}] ERRO: {e}")
        import traceback
        traceback.print_exc()
        return (None, None)


def _run_adaptive_single_seed(args):
    try:
        env_class = args['env_class']
        policy_class = args['policy_class']
        param_bounds = args['param_bounds']
        seed = args['seed']
        es_generations = args['es_generations']
        es_population = args['es_population']
        agent_generations = args['agent_generations']
        learning_rate = args['learning_rate']
        noise_std = args['noise_std']
        weight_decay = args['weight_decay']
        
        print(f"[Seed {seed}] Iniciando adaptativo...")
        
        es = AdaptiveES(
            param_bounds,
            population_size=es_population,
            learning_rate=learning_rate,
            noise_std=noise_std,
            weight_decay=weight_decay,
            seed=seed
        )
        
        es_history = []
        
        for gen in range(es_generations):
            samples, noise = es.ask()
            fitness_scores = []
            
            for individual in samples:
                env_params = es._params_to_dict(individual)
                
                _, agent_history = optimize_policy(
                    env_class=env_class,
                    policy_class=policy_class,
                    generations=agent_generations,
                    population_size=51,
                    learning_rate=learning_rate,
                    noise_std=noise_std,
                    verbose=False,
                    seed=seed,
                    num_episodes=int(env_params.get('n_episodes', 5)),
                    max_steps=int(env_params.get('max_steps', 500)),
                    theta_range=env_params.get('theta_range', None),
                    theta_dot_range=env_params.get('theta_dot_range', None),
                    motor_noise=env_params.get('noise_level', None)
                )
                
                fitness = agent_history[-1]['best_fitness']
                fitness_scores.append(fitness)
            
            es.tell(fitness_scores, noise)
            
            result = es.fitness_history[-1]
            es_history.append(result)
            
            if gen % 100 == 0:
                print(f"[Seed {seed}] Gen {gen}/{es_generations}: best={es.best_fitness:.2f}")
        
        final_fitness = es.best_fitness
        best_params = es.get_best_params()
        
        print(f"[Seed {seed}] Adaptativo concluído: {final_fitness:.2f}")
        
        return (final_fitness, es_history, best_params)
        
    except Exception as e:
        print(f"[Seed {seed}] ERRO: {e}")
        import traceback
        traceback.print_exc()
        return (None, None, None)


def run_full_experiment_cartpole():
    experiment = MultiSeedExperiment(num_seeds=10)
    experiment.run_baseline(
        env_class=CartPoleEnv,
        policy_class=CartPolePolicy,
        env_name='cartpole',
        generations=1000,
        population_size=101,
        learning_rate=0.05,
        noise_std=0.1,
        num_episodes=5,
        max_steps=100,
        theta_range=0.725378,
        theta_dot_range=3.218251,
        motor_noise=0.113388
    )
    
    param_bounds = {
        'n_episodes': (2, 30),
        'max_steps': (100, 500),
        'noise_level': (0.001, 0.15),
        'theta_range': (0.05, 1.0),
        'theta_dot_range': (0.1, 4.0)
    }
    
    experiment.run_adaptive(
        env_class=CartPoleEnv,
        policy_class=CartPolePolicy,
        param_bounds=param_bounds,
        env_name='cartpole',
        es_generations=100,
        es_population=10,
        agent_generations=10,
        learning_rate=0.01,
        noise_std=0.03,
        weight_decay=0.001
    )
    
    experiment.save_results('cartpole_multi_seed.pkl')
    experiment.plot_boxplots(save_prefix='cartpole_boxplot')
    
    return experiment


if __name__ == "__main__":
    print("Experimento Multi-Seed")
    print("1 - Pendulum")
    print("2 - CartPole")
    print("3 - Ambos")
    
    choice = input("Escolha (1/2/3): ").strip()
    
    if choice == "1":
        run_full_experiment_pendulum()
    elif choice == "2":
        run_full_experiment_cartpole()
    else:
        run_full_experiment_pendulum()
        run_full_experiment_cartpole()
