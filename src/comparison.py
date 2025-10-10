"""
Comparison between baseline e adaptive algorithm
Author: Brenda Silva Machado
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from evaluate import optimize_policy
from adaptative_algo_opt import AdaptiveAlgorithm
from pendulum import PendulumEnv
from policy import PendulumPolicy


class ExperimentComparison:
    
    def __init__(self):
        self.results = {}
    
    def load_baseline_from_file(self, history_file='evolution_history.pkl', 
                                params_file='pendulum_best_params.npy',
                                name="baseline"):
        try:
            with open(history_file, 'rb') as f:
                history_data = pickle.load(f)
            
            best_params = np.load(params_file)
            
            if 'pendulum' in history_data:
                history = history_data['pendulum']
            else:
                history = history_data
            
            self.results[name] = {
                'type': 'baseline',
                'best_params': best_params,
                'history': history,
                'final_fitness': history[-1]['best_fitness']
            }
            
            print(f"✓ Baseline carregado de {history_file}")
            print(f"  Fitness final: {history[-1]['best_fitness']:.2f}")
            print(f"  Gerações: {len(history)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Erro ao carregar baseline: {e}")
            return False
    
    def run_adaptive(self, env_class, policy_class, param_bounds, 
                     ga_generations=10, ga_population=20, agent_generations=100,
                     learning_rate=0.02, noise_std=0.05,
                     name="adaptive", save_prefix="adaptive"):
        
        print(f"\n{'='*80}")
        print(f"INICIANDO ALGORITMO ADAPTATIVO")
        print(f"{'='*80}")
        print(f"GA: {ga_generations} gerações, população={ga_population}")
        print(f"Agente: {agent_generations} gerações OpenAI-ES por avaliação")
        print(f"Parâmetros a adaptar: {list(param_bounds.keys())}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        ga = AdaptiveAlgorithm(param_bounds, population_size=ga_population)
        ga_history = []
        param_evolution = {param: [] for param in param_bounds.keys()}
        
        for gen in range(ga_generations):
            gen_start = time.time()
            print(f"\n{'─'*80}")
            print(f"GA GERAÇÃO {gen}/{ga_generations-1}")
            print(f"{'─'*80}")
            
            fitness_scores = []
            
            for idx, individual in enumerate(ga.population):
                ind_start = time.time()
                env_params = ga._params_to_dict(individual)
                
                print(f"\n  [{idx+1}/{ga_population}] Avaliando configuração:")
                print(f"    n_episodes={int(env_params.get('n_episodes', 5))}, "
                      f"max_steps={int(env_params.get('max_steps', 500))}, "
                      f"noise={env_params.get('noise_level', 0):.3f}")
                print(f"    theta_range={env_params.get('theta_range', 0):.3f}, "
                      f"theta_dot_range={env_params.get('theta_dot_range', 0):.3f}")
                
                try:
                    _, agent_history = optimize_policy(
                        env_class=env_class,
                        policy_class=policy_class,
                        generations=agent_generations,
                        population_size=51,
                        learning_rate=learning_rate,
                        noise_std=noise_std,
                        verbose=False,
                        num_episodes=int(env_params.get('n_episodes', 5)),
                        max_steps=int(env_params.get('max_steps', 500)),
                        theta_range=env_params.get('theta_range', None),
                        theta_dot_range=env_params.get('theta_dot_range', None),
                        motor_noise=env_params.get('noise_level', None)
                    )
                    
                    fitness = agent_history[-1]['best_fitness']
                    fitness_scores.append(fitness)
                    
                    ind_time = time.time() - ind_start
                    print(f"    → Fitness: {fitness:.2f} (tempo: {ind_time:.1f}s)")
                    
                except Exception as e:
                    print(f"    ✗ ERRO na avaliação: {e}")
                    fitness_scores.append(-1e6)
            
            result = ga.evolve(fitness_scores)
            ga_history.append(result)
            
            for param in param_bounds.keys():
                param_evolution[param].append(result['best_params'][param])
            
            gen_time = time.time() - gen_start
            elapsed_total = time.time() - start_time
            
            print(f"\n{'─'*80}")
            print(f"RESULTADO GA GERAÇÃO {gen}:")
            print(f"  Melhor fitness:  {result['best_fitness']:.2f}")
            print(f"  Fitness médio:   {result['avg_fitness']:.2f}")
            print(f"  Tempo geração:   {gen_time:.1f}s")
            print(f"  Tempo total:     {elapsed_total:.1f}s")
            print(f"\n  Melhores parâmetros:")
            for k, v in result['best_params'].items():
                print(f"    {k}: {v:.4f}")
            print(f"{'─'*80}")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"ALGORITMO ADAPTATIVO CONCLUÍDO")
        print(f"{'='*80}")
        print(f"Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Melhor fitness alcançado: {ga.best_fitness:.2f}")
        print(f"\nMelhores parâmetros finais:")
        for k, v in ga.get_best_params().items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*80}\n")
        
        self._save_adaptive_results(ga, ga_history, param_evolution, save_prefix)
        
        self.results[name] = {
            'type': 'adaptive',
            'ga_history': ga_history,
            'param_evolution': param_evolution,
            'best_env_params': ga.get_best_params(),
            'final_fitness': ga.best_fitness,
            'total_time': total_time
        }
        
        return ga.get_best_params(), ga_history
    
    def _save_adaptive_results(self, ga, ga_history, param_evolution, prefix):
        env_params = ga.get_best_params()
        np.save(f'{prefix}_best_env_params.npy', 
                np.array([env_params[k] for k in sorted(env_params.keys())]))
        
        with open(f'{prefix}_history.pkl', 'wb') as f:
            pickle.dump({
                'ga_history': ga_history,
                'param_evolution': param_evolution,
                'best_env_params': env_params,
                'final_fitness': ga.best_fitness
            }, f)
        
        print(f"✓ Resultados salvos:")
        print(f"  - {prefix}_best_env_params.npy")
        print(f"  - {prefix}_history.pkl")
    
    def print_comparison(self):
        print(f"\n{'='*80}")
        print("COMPARAÇÃO DE RESULTADOS")
        print(f"{'='*80}\n")
        
        for name, data in self.results.items():
            print(f"{name.upper()}:")
            print(f"  Tipo: {data['type']}")
            print(f"  Fitness Final: {data['final_fitness']:.2f}")
            
            if data['type'] == 'adaptive':
                print(f"  Tempo total: {data.get('total_time', 0):.1f}s")
                print(f"  Parâmetros adaptados:")
                for k, v in data['best_env_params'].items():
                    print(f"    {k}: {v:.4f}")
            print()

        if len(self.results) >= 2:
            fitnesses = {name: data['final_fitness'] for name, data in self.results.items()}
            best_name = max(fitnesses, key=fitnesses.get)
            worst_name = min(fitnesses, key=fitnesses.get)
            
            improvement = fitnesses[best_name] - fitnesses[worst_name]
            improvement_pct = (improvement / abs(fitnesses[worst_name])) * 100
            
            print(f"{'='*80}")
            print(f"MELHOR ABORDAGEM: {best_name.upper()}")
            print(f"Melhoria absoluta: {improvement:.2f}")
            print(f"Melhoria relativa: {improvement_pct:.1f}%")
            print(f"{'='*80}")
    
    def plot_fitness_comparison(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        ax = axes[0]
        for name, data in self.results.items():
            if data['type'] == 'baseline':
                gens = [h['generation'] for h in data['history']]
                fitness = [h['best_fitness'] for h in data['history']]
                ax.plot(gens, fitness, label=f"{name} (fixo)", linewidth=2)
            elif data['type'] == 'adaptive':
                gens = list(range(len(data['ga_history'])))
                fitness = [h['best_fitness'] for h in data['ga_history']]
                ax.plot(gens, fitness, label=f"{name} (adaptativo)", 
                       linewidth=2, linestyle='--', marker='o', markersize=6)
        
        ax.set_xlabel('Geração', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Comparação de Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        names = list(self.results.keys())
        fitnesses = [self.results[n]['final_fitness'] for n in names]
        colors = ['#3498db' if self.results[n]['type'] == 'baseline' else '#e74c3c' 
                  for n in names]
        
        bars = ax.bar(names, fitnesses, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Fitness Final', fontsize=12)
        ax.set_title('Fitness Final Comparado', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, fit in zip(bars, fitnesses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{fit:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_param_evolution(self, adaptive_name='adaptive', save_path=None):
        if adaptive_name not in self.results:
            print(f"Resultado '{adaptive_name}' não encontrado!")
            return
        
        data = self.results[adaptive_name]
        if data['type'] != 'adaptive':
            print(f"'{adaptive_name}' não é um experimento adaptativo!")
            return
        
        param_evolution = data['param_evolution']
        n_params = len(param_evolution)
        
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for ax, (param_name, values) in zip(axes, param_evolution.items()):
            generations = list(range(len(values)))
            ax.plot(generations, values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Geração GA', fontsize=11)
            ax.set_ylabel(param_name, fontsize=11)
            ax.set_title(f'Evolução: {param_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5, 
                      label=f'Média: {mean_val:.3f}')
            ax.fill_between(generations, mean_val-std_val, mean_val+std_val, 
                           alpha=0.2, color='red', label=f'±σ: {std_val:.3f}')
            ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_param_statistics(self, adaptive_name='adaptive'):
        if adaptive_name not in self.results:
            print(f"Estatísticas não disponíveis para '{adaptive_name}'")
            return
        
        data = self.results[adaptive_name]
        if data['type'] != 'adaptive':
            return
        
        param_evolution = data['param_evolution']
        
        print(f"\n{'='*80}")
        print(f"ESTATÍSTICAS DE VARIAÇÃO DOS PARÂMETROS ({adaptive_name})")
        print(f"{'='*80}\n")
        
        for param_name, values in param_evolution.items():
            print(f"{param_name}:")
            print(f"  Inicial:  {values[0]:.4f}")
            print(f"  Final:    {values[-1]:.4f}")
            print(f"  Média:    {np.mean(values):.4f} ± {np.std(values):.4f}")
            print(f"  Range:    [{np.min(values):.4f}, {np.max(values):.4f}]")
            print(f"  Variação: {np.max(values) - np.min(values):.4f}")
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            print(f"  Coef.Var: {cv:.4f}")
            print()


if __name__ == "__main__":
    comparison = ExperimentComparison()
    
    comparison.load_baseline_from_file(
        history_file='evolution_history.pkl',
        params_file='pendulum_best_params.npy',
        name="baseline_fixed"
    )
    
    param_bounds = {
        'n_episodes': (10, 40),
        'max_steps': (500, 1000),
        'noise_level': (0.05, 0.15),
        'theta_range': (1.0, 3.0),
        'theta_dot_range': (1.0, 2.0)
    }
    
    comparison.run_adaptive(
        env_class=PendulumEnv,
        policy_class=PendulumPolicy,
        param_bounds=param_bounds,
        ga_generations=1000,        
        ga_population=20,           
        agent_generations=1,
        learning_rate=0.02,
        noise_std=0.05,
        name="adaptive_ga",
        save_prefix="adaptive_pendulum"
    )

    
    comparison.print_comparison()
    comparison.print_param_statistics('adaptive_ga')
    comparison.plot_fitness_comparison(save_path='fitness_comparison.png')
    comparison.plot_param_evolution('adaptive_ga', save_path='param_evolution.png')
