"""
Módulo de Integração - Otimizador Adaptativo com Experimentos Existentes

Este módulo integra o AdaptiveExperimentOptimizer com os experimentos de
CartPole e Pendulum, permitindo adaptação online de parâmetros durante
o treinamento.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Any
from adaptive_experiment_optimizer import AdaptiveExperimentOptimizer
from cartpole import CartPoleEnv
from pendulum import PendulumEnv
from policy import Policy
import os

class AdaptiveExperimentRunner:
    
    def __init__(self, env_name: str, save_dir: str = None):
        self.env_name = env_name.lower()
        self.save_dir = save_dir or f"~/adaptive_experiments_{self.env_name}"
        self.save_dir = os.path.expanduser(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.optimizer = AdaptiveExperimentOptimizer(env_name=self.env_name)
        self.env = self._create_environment()
        self.policy = self._create_policy()
        
        self.adaptation_metrics = {
            'cycle': [],
            'fitness_mean': [],
            'fitness_std': [],
            'param_stability': [],
            'adaptation_success': []
        }
    
    def _create_environment(self):
        if self.env_name == 'cartpole':
            return CartPoleEnv()
        elif self.env_name == 'pendulum':
            return PendulumEnv()
        else:
            raise ValueError(f"Ambiente não suportado: {self.env_name}")
    
    def _create_policy(self):
        if self.env_name == 'cartpole':
            return Policy(input_size=4)
        elif self.env_name == 'pendulum':
            return Policy(input_size=3)
        else:
            raise ValueError(f"Política não suportada para: {self.env_name}")
    
    def collect_fitness_with_params(self, config: Dict) -> List[float]:
        fitness_results = []
        
        for episode in range(config['n_episodes']):
            try:
                obs, _ = self.env.reset()
                
                if 'custom_bounds' in config:
                    theta_range, theta_dot_range = config['custom_bounds']
                    
                    if self.env_name == 'cartpole':
                        initial_theta = np.random.uniform(-theta_range, theta_range)
                        initial_theta_dot = np.random.uniform(-theta_dot_range, theta_dot_range)
                        custom_state = [0.0, 0.0, initial_theta, initial_theta_dot]
                        self.env.unwrapped.state = np.array(custom_state)
                    
                    elif self.env_name == 'pendulum':
                        initial_theta = np.random.uniform(-theta_range, theta_range)
                        initial_theta_dot = np.random.uniform(-theta_dot_range, theta_dot_range)
                        custom_state = [initial_theta, initial_theta_dot]
                        self.env.unwrapped.state = np.array(custom_state)
                        obs = self.env._get_obs()
                
                total_reward = 0
                steps = 0
                
                for step in range(config['max_steps']):
                    action = self.policy.get_action(obs)
                    
                    if self.env_name == 'cartpole':
                        obs, reward, terminated, truncated, _ = self.env.step(action, config['noise_level'])
                    elif self.env_name == 'pendulum':
                        noisy_action = action + np.random.normal(0, config['noise_level'])
                        noisy_action = np.clip(noisy_action, -2.0, 2.0)
                        obs, reward, terminated, truncated, _ = self.env.step(noisy_action)
                    
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                if self.env_name == 'cartpole':
                    fitness = steps
                elif self.env_name == 'pendulum':
                    fitness = total_reward
                
                fitness_results.append(fitness)
                
            except Exception as e:
                print(f"Erro no episódio {episode}: {e}")
                default_fitness = 1 if self.env_name == 'cartpole' else -1000
                fitness_results.append(default_fitness)
        
        return fitness_results
    
    def run_adaptive_training(self, n_adaptation_cycles: int = 15, 
                            save_frequency: int = 5) -> Dict:
        
        print(f"=== TREINAMENTO ADAPTATIVO - {self.env_name.upper()} ===")
        print(f"Ciclos de adaptação: {n_adaptation_cycles}")
        print(f"Diretório de salvamento: {self.save_dir}")
        
        best_fitness = float('-inf') if self.env_name == 'pendulum' else 0
        best_params = None
        stability_window = 3
        
        for cycle in range(n_adaptation_cycles):
            print(f"\n--- Ciclo {cycle + 1}/{n_adaptation_cycles} ---")
            
            config = self.optimizer.get_experiment_config()
            
            print("Parâmetros atuais:")
            for key, value in config.items():
                if key != 'custom_bounds':
                    print(f"  {key}: {value}")
            
            print(f"Executando {config['n_episodes']} episódios...")
            fitness_results = self.collect_fitness_with_params(config)
            
            fitness_mean = np.mean(fitness_results)
            fitness_std = np.std(fitness_results)
            fitness_min = np.min(fitness_results)
            fitness_max = np.max(fitness_results)
            
            print(f"Fitness - Média: {fitness_mean:.2f}, Std: {fitness_std:.2f}")
            print(f"Fitness - Min: {fitness_min:.2f}, Max: {fitness_max:.2f}")
            
            current_performance = fitness_mean
            if ((self.env_name == 'pendulum' and current_performance > best_fitness) or
                (self.env_name == 'cartpole' and current_performance > best_fitness)):
                best_fitness = current_performance
                best_params = config.copy()
                print(f"*** NOVO MELHOR RESULTADO: {best_fitness:.2f} ***")
            
            self.optimizer.update_performance(fitness_results, f'cycle_{cycle}')
            
            param_stability = self._calculate_param_stability(stability_window)
            
            adaptation_success = self._evaluate_adaptation_success(cycle)
            
            self.adaptation_metrics['cycle'].append(cycle)
            self.adaptation_metrics['fitness_mean'].append(fitness_mean)
            self.adaptation_metrics['fitness_std'].append(fitness_std)
            self.adaptation_metrics['param_stability'].append(param_stability)
            self.adaptation_metrics['adaptation_success'].append(adaptation_success)
            
            if cycle < n_adaptation_cycles - 1:
                print("Adaptando parâmetros...")
                self.optimizer.adapt_parameters()
            
            if (cycle + 1) % save_frequency == 0:
                self._save_checkpoint(cycle)
        
        self._save_final_results(best_params, best_fitness)
        
        self._generate_adaptation_report()
        
        print(f"\n=== TREINAMENTO ADAPTATIVO CONCLUÍDO ===")
        print(f"Melhor fitness obtido: {best_fitness:.2f}")
        print(f"Melhores parâmetros: {best_params}")
        
        return {
            'best_fitness': best_fitness,
            'best_params': best_params,
            'adaptation_metrics': self.adaptation_metrics,
            'final_optimizer_state': self.optimizer.current_params
        }
    
    def _calculate_param_stability(self, window_size: int) -> float:
        if len(self.optimizer.param_history) < window_size:
            return 0.0
        
        recent_params = list(self.optimizer.param_history)[-window_size:]
        
        param_variations = []
        for param_name in recent_params[0].keys():
            values = [p[param_name] for p in recent_params]
            if len(set(values)) > 1:
                variation = np.std(values) / (np.mean(values) + 1e-8)
                param_variations.append(variation)
        
        avg_variation = np.mean(param_variations) if param_variations else 0
        stability = 1.0 / (1.0 + avg_variation)
        
        return stability
    
    def _evaluate_adaptation_success(self, current_cycle: int) -> bool:
        if len(self.adaptation_metrics['fitness_mean']) < 3:
            return False
        
        recent_fitness = self.adaptation_metrics['fitness_mean'][-3:]
        
        if self.env_name == 'pendulum':
            trend = recent_fitness[-1] > recent_fitness[0]
        else:
            trend = recent_fitness[-1] > recent_fitness[0]
        
        return trend
    
    def _save_checkpoint(self, cycle: int):
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_cycle_{cycle}.json')
        self.optimizer.save_state(checkpoint_path)
        print(f"Checkpoint salvo: {checkpoint_path}")
    
    def _save_final_results(self, best_params: Dict, best_fitness: float):
        final_state_path = os.path.join(self.save_dir, 'final_optimizer_state.json')
        self.optimizer.save_state(final_state_path)
        
        metrics_path = os.path.join(self.save_dir, 'adaptation_metrics.npy')
        np.save(metrics_path, self.adaptation_metrics)
        
        best_params_path = os.path.join(self.save_dir, 'best_parameters.json')
        import json
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_fitness': float(best_fitness),
                'best_params': best_params,
                'env_name': self.env_name
            }, f, indent=2)
        
        print(f"Resultados finais salvos em: {self.save_dir}")
    
    def _generate_adaptation_report(self):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        cycles = self.adaptation_metrics['cycle']
        
        axes[0,0].plot(cycles, self.adaptation_metrics['fitness_mean'], 'b-', marker='o', label='Média')
        
        fitness_std = self.adaptation_metrics['fitness_std']
        axes[0,0].fill_between(cycles, 
                              np.array(self.adaptation_metrics['fitness_mean']) - np.array(fitness_std),
                              np.array(self.adaptation_metrics['fitness_mean']) + np.array(fitness_std),
                              alpha=0.3)
        
        axes[0,0].set_title('Evolução do Fitness Durante Adaptação')
        axes[0,0].set_xlabel('Ciclo de Adaptação')
        axes[0,0].set_ylabel('Fitness')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        axes[0,1].plot(cycles, self.adaptation_metrics['param_stability'], 'g-', marker='s')
        axes[0,1].set_title('Estabilidade dos Parâmetros')
        axes[0,1].set_xlabel('Ciclo de Adaptação')
        axes[0,1].set_ylabel('Índice de Estabilidade')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].grid(True)
        
        success_rate = np.cumsum(self.adaptation_metrics['adaptation_success']) / (np.arange(len(cycles)) + 1)
        axes[1,0].plot(cycles, success_rate, 'r-', marker='^')
        axes[1,0].set_title('Taxa de Sucesso da Adaptação')
        axes[1,0].set_xlabel('Ciclo de Adaptação')
        axes[1,0].set_ylabel('Taxa de Sucesso Cumulativa')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].grid(True)
        
        best_fitness_per_cycle = []
        running_best = float('-inf') if self.env_name == 'pendulum' else 0
        
        for fitness in self.adaptation_metrics['fitness_mean']:
            if ((self.env_name == 'pendulum' and fitness > running_best) or
                (self.env_name == 'cartpole' and fitness > running_best)):
                running_best = fitness
            best_fitness_per_cycle.append(running_best)
        
        axes[1,1].plot(cycles, best_fitness_per_cycle, 'm-', marker='D', linewidth=2)
        axes[1,1].set_title('Melhor Fitness Acumulativo')
        axes[1,1].set_xlabel('Ciclo de Adaptação')
        axes[1,1].set_ylabel('Melhor Fitness')
        axes[1,1].grid(True)
        
        plt.suptitle(f'Relatório de Adaptação - {self.env_name.upper()}', fontsize=16)
        plt.tight_layout()
        
        report_path = os.path.join(self.save_dir, 'adaptation_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Relatório visual salvo: {report_path}")
    
    def run_comparative_analysis(self, baseline_params: Dict = None) -> Dict:
        
        print(f"\n=== ANÁLISE COMPARATIVA ===")
        
        if baseline_params is None:
            baseline_params = self.optimizer._get_default_params()
        
        print("Testando parâmetros baseline...")
        baseline_fitness = self.collect_fitness_with_params(baseline_params)
        baseline_mean = np.mean(baseline_fitness)
        
        print("Testando parâmetros adaptativos...")
        adaptive_config = self.optimizer.get_experiment_config()
        adaptive_fitness = self.collect_fitness_with_params(adaptive_config)
        adaptive_mean = np.mean(adaptive_fitness)
        
        improvement = adaptive_mean - baseline_mean
        improvement_pct = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
        
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(adaptive_fitness, baseline_fitness)
        
        results = {
            'baseline_mean': baseline_mean,
            'baseline_std': np.std(baseline_fitness),
            'adaptive_mean': adaptive_mean,
            'adaptive_std': np.std(adaptive_fitness),
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
            't_statistic': t_stat
        }
        
        print(f"\nResultados da Análise Comparativa:")
        print(f"Baseline - Média: {baseline_mean:.2f}, Std: {results['baseline_std']:.2f}")
        print(f"Adaptativo - Média: {adaptive_mean:.2f}, Std: {results['adaptive_std']:.2f}")
        print(f"Melhoria: {improvement:.2f} ({improvement_pct:+.1f}%)")
        print(f"Significância estatística: {'Sim' if results['statistical_significance'] else 'Não'} (p={p_value:.4f})")
        
        return results
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

def run_cartpole_adaptive_experiment(n_cycles: int = 15, save_dir: str = None) -> Dict:
    runner = AdaptiveExperimentRunner('cartpole', save_dir)
    
    try:
        results = runner.run_adaptive_training(n_cycles)
        comparative_analysis = runner.run_comparative_analysis()
        results['comparative_analysis'] = comparative_analysis
        
        return results
    finally:
        runner.close()

def run_pendulum_adaptive_experiment(n_cycles: int = 15, save_dir: str = None) -> Dict:
    runner = AdaptiveExperimentRunner('pendulum', save_dir)
    
    try:
        results = runner.run_adaptive_training(n_cycles)
        comparative_analysis = runner.run_comparative_analysis()
        results['comparative_analysis'] = comparative_analysis
        
        return results
    finally:
        runner.close()

def compare_environments_adaptation(n_cycles: int = 12) -> Dict:
    print("=== COMPARAÇÃO ENTRE AMBIENTES ===")
    
    cartpole_results = run_cartpole_adaptive_experiment(n_cycles, "~/adaptive_comparison/cartpole")
    pendulum_results = run_pendulum_adaptive_experiment(n_cycles, "~/adaptive_comparison/pendulum")
    
    comparison = {
        'cartpole': {
            'best_fitness': cartpole_results['best_fitness'],
            'final_params': cartpole_results['best_params'],
            'improvement': cartpole_results['comparative_analysis']['improvement_percentage']
        },
        'pendulum': {
            'best_fitness': pendulum_results['best_fitness'],
            'final_params': pendulum_results['best_params'],
            'improvement': pendulum_results['comparative_analysis']['improvement_percentage']
        }
    }
    
    print(f"\nResumo Comparativo:")
    print(f"CartPole - Melhor fitness: {comparison['cartpole']['best_fitness']:.2f}")
    print(f"CartPole - Melhoria: {comparison['cartpole']['improvement']:+.1f}%")
    print(f"Pendulum - Melhor fitness: {comparison['pendulum']['best_fitness']:.2f}")
    print(f"Pendulum - Melhoria: {comparison['pendulum']['improvement']:+.1f}%")
    
    return comparison

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Executar experimentos adaptativos')
    parser.add_argument('--env', choices=['cartpole', 'pendulum', 'both'], 
                       default='both', help='Ambiente para testar')
    parser.add_argument('--cycles', type=int, default=12, 
                       help='Número de ciclos de adaptação')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    if args.env == 'cartpole':
        results = run_cartpole_adaptive_experiment(args.cycles, args.save_dir)
        print("Experimento CartPole concluído.")
        
    elif args.env == 'pendulum':
        results = run_pendulum_adaptive_experiment(args.cycles, args.save_dir)
        print("Experimento Pendulum concluído.")
        
    elif args.env == 'both':
        results = compare_environments_adaptation(args.cycles)
        
        # return {
        #     'best_fitness': best_fitness,
        #     'best_params': best_params,
        #     'adaptation_metrics': self.adaptation_metrics,
        #     'final_optimizer_state': self.optimizer.current_params
        # }
    
def _calculate_param_stability(self, window_size: int) -> float:
    """Calcula estabilidade dos parâmetros na janela recente"""
    if len(self.optimizer.param_history) < window_size:
        return 0.0
    
    recent_params = list(self.optimizer.param_history)[-window_size:]
    
    # Calcular variação relativa para cada parâmetro
    param_variations = []
    for param_name in recent_params[0].keys():
        values = [p[param_name] for p in recent_params]
        if len(set(values)) > 1:  # Se há variação
            variation = np.std(values) / (np.mean(values) + 1e-8)
            param_variations.append(variation)
    
    # Estabilidade = inverso da variação média
    avg_variation = np.mean(param_variations) if param_variations else 0
    stability = 1.0 / (1.0 + avg_variation)
    
    return stability

def _evaluate_adaptation_success(self, current_cycle: int) -> bool:
    """Avalia se a adaptação está sendo bem-sucedida"""
    if len(self.adaptation_metrics['fitness_mean']) < 3:
        return False
    
    # Verificar tendência de melhoria nos últimos 3 ciclos
    recent_fitness = self.adaptation_metrics['fitness_mean'][-3:]
    
    if self.env_name == 'pendulum':
        # Para pendulum, fitness crescente é melhor
        trend = recent_fitness[-1] > recent_fitness[0]
    else:
        # Para cartpole, fitness crescente é melhor
        trend = recent_fitness[-1] > recent_fitness[0]
    
    return trend

def _save_checkpoint(self, cycle: int):
    """Salva checkpoint do estado atual"""
    checkpoint_path = os.path.join(self.save_dir, f'checkpoint_cycle_{cycle}.json')
    self.optimizer.save_state(checkpoint_path)
    print(f"Checkpoint salvo: {checkpoint_path}")

def _save_final_results(self, best_params: Dict, best_fitness: float):
    """Salva resultados finais e estado do otimizador"""
    # Salvar estado do otimizador
    final_state_path = os.path.join(self.save_dir, 'final_optimizer_state.json')
    self.optimizer.save_state(final_state_path)
    
    # Salvar métricas de adaptação
    metrics_path = os.path.join(self.save_dir, 'adaptation_metrics.npy')
    np.save(metrics_path, self.adaptation_metrics)
    
    # Salvar melhores parâmetros
    best_params_path = os.path.join(self.save_dir, 'best_parameters.json')
    import json
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_fitness': float(best_fitness),
            'best_params': best_params,
            'env_name': self.env_name
        }, f, indent=2)
    
    print(f"Resultados finais salvos em: {self.save_dir}")

def _generate_adaptation_report(self):
    """Gera relatório visual do progresso da adaptação"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    cycles = self.adaptation_metrics['cycle']
    
    # 1. Evolução do fitness
    axes[0,0].plot(cycles, self.adaptation_metrics['fitness_mean'], 'b-', marker='o', label='Média')
    
    # Adicionar barras de erro
    fitness_std = self.adaptation_metrics['fitness_std']
    axes[0,0].fill_between(cycles, 
                            np.array(self.adaptation_metrics['fitness_mean']) - np.array(fitness_std),
                            np.array(self.adaptation_metrics['fitness_mean']) + np.array(fitness_std),
                            alpha=0.3)
    
    axes[0,0].set_title('Evolução do Fitness Durante Adaptação')
    axes[0,0].set_xlabel('Ciclo de Adaptação')
    axes[0,0].set_ylabel('Fitness')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 2. Estabilidade dos parâmetros
    axes[0,1].plot(cycles, self.adaptation_metrics['param_stability'], 'g-', marker='s')
    axes[0,1].set_title('Estabilidade dos Parâmetros')
    axes[0,1].set_xlabel('Ciclo de Adaptação')
    axes[0,1].set_ylabel('Índice de Estabilidade')
    axes[0,1].set_ylim(0, 1)
    axes[0,1].grid(True)
    
    # 3. Taxa de sucesso da adaptação
    success_rate = np.cumsum(self.adaptation_metrics['adaptation_success']) / (np.arange(len(cycles)) + 1)
    axes[1,0].plot(cycles, success_rate, 'r-', marker='^')
    axes[1,0].set_title('Taxa de Sucesso da Adaptação')
    axes[1,0].set_xlabel('Ciclo de Adaptação')
    axes[1,0].set_ylabel('Taxa de Sucesso Cumulativa')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(True)
    
    # 4. Distribuição dos melhores fitness
    best_fitness_per_cycle = []
    running_best = float('-inf') if self.env_name == 'pendulum' else 0
    
    for fitness in self.adaptation_metrics['fitness_mean']:
        if ((self.env_name == 'pendulum' and fitness > running_best) or
            (self.env_name == 'cartpole' and fitness > running_best)):
            running_best = fitness
        best_fitness_per_cycle.append(running_best)
    
    axes[1,1].plot(cycles, best_fitness_per_cycle, 'm-', marker='D', linewidth=2)
    axes[1,1].set_title('Melhor Fitness Acumulativo')
    axes[1,1].set_xlabel('Ciclo de Adaptação')
    axes[1,1].set_ylabel('Melhor Fitness')
    axes[1,1].grid(True)
    
    plt.suptitle(f'Relatório de Adaptação - {self.env_name.upper()}', fontsize=16)
    plt.tight_layout()
    
    # Salvar relatório
    report_path = os.path.join(self.save_dir, 'adaptation_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Relatório visual salvo: {report_path}")

def run_comparative_analysis(self, baseline_params: Dict = None) -> Dict:
    """
    Executa análise comparativa entre parâmetros adaptativos e baseline.
    """
    print(f"\n=== ANÁLISE COMPARATIVA ===")
    
    # Parâmetros baseline (padrão se não especificado)
    if baseline_params is None:
        baseline_params = self.optimizer._get_default_params()
    
    print("Testando parâmetros baseline...")
    baseline_fitness = self.collect_fitness_with_params(baseline_params)
    baseline_mean = np.mean(baseline_fitness)
    
    print("Testando parâmetros adaptativos...")
    adaptive_config = self.optimizer.get_experiment_config()
    adaptive_fitness = self.collect_fitness_with_params(adaptive_config)
    adaptive_mean = np.mean(adaptive_fitness)
    
    # Calcular estatísticas comparativas
    improvement = adaptive_mean - baseline_mean
    improvement_pct = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
    
    # Teste estatístico simples (t-test)
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(adaptive_fitness, baseline_fitness)
    
    results = {
        'baseline_mean': baseline_mean,
        'baseline_std': np.std(baseline_fitness),
        'adaptive_mean': adaptive_mean,
        'adaptive_std': np.std(adaptive_fitness),
        'improvement': improvement,
        'improvement_percentage': improvement_pct,
        'statistical_significance': p_value < 0.05,
        'p_value': p_value,
        't_statistic': t_stat
    }
    
    print(f"\nResultados da Análise Comparativa:")
    print(f"Baseline - Média: {baseline_mean:.2f}, Std: {results['baseline_std']:.2f}")
    print(f"Adaptativo - Média: {adaptive_mean:.2f}, Std: {results['adaptive_std']:.2f}")
    print(f"Melhoria: {improvement:.2f} ({improvement_pct:+.1f}%)")
    print(f"Significância estatística: {'Sim' if results['statistical_significance'] else 'Não'} (p={p_value:.4f})")
    
    return results

def close(self):
    """Limpa recursos"""
    if hasattr(self.env, 'close'):
        self.env.close()

# Funções utilitárias para facilitar o uso

def run_cartpole_adaptive_experiment(n_cycles: int = 15, save_dir: str = None) -> Dict:
    """Função de conveniência para executar experimento adaptativo no CartPole"""
    runner = AdaptiveExperimentRunner('cartpole', save_dir)
    
    try:
        results = runner.run_adaptive_training(n_cycles)
        comparative_analysis = runner.run_comparative_analysis()
        results['comparative_analysis'] = comparative_analysis
        
        return results
    finally:
        runner.close()

def run_pendulum_adaptive_experiment(n_cycles: int = 15, save_dir: str = None) -> Dict:
    """Função de conveniência para executar experimento adaptativo no Pendulum"""
    runner = AdaptiveExperimentRunner('pendulum', save_dir)
    
    try:
        results = runner.run_adaptive_training(n_cycles)
        comparative_analysis = runner.run_comparative_analysis()
        results['comparative_analysis'] = comparative_analysis
        
        return results
    finally:
        runner.close()

def compare_environments_adaptation(n_cycles: int = 12) -> Dict:
    """Compara adaptação entre CartPole e Pendulum"""
    print("=== COMPARAÇÃO ENTRE AMBIENTES ===")
    
    # Executar experimentos
    cartpole_results = run_cartpole_adaptive_experiment(n_cycles, "~/adaptive_comparison/cartpole")
    pendulum_results = run_pendulum_adaptive_experiment(n_cycles, "~/adaptive_comparison/pendulum")
    
    # Análise comparativa
    comparison = {
        'cartpole': {
            'best_fitness': cartpole_results['best_fitness'],
            'final_params': cartpole_results['best_params'],
            'improvement': cartpole_results['comparative_analysis']['improvement_percentage']
        },
        'pendulum': {
            'best_fitness': pendulum_results['best_fitness'],
            'final_params': pendulum_results['best_params'],
            'improvement': pendulum_results['comparative_analysis']['improvement_percentage']
        }
    }
    
    print(f"\nResumo Comparativo:")
    print(f"CartPole - Melhor fitness: {comparison['cartpole']['best_fitness']:.2f}")
    print(f"CartPole - Melhoria: {comparison['cartpole']['improvement']:+.1f}%")
    print(f"Pendulum - Melhor fitness: {comparison['pendulum']['best_fitness']:.2f}")
    print(f"Pendulum - Melhoria: {comparison['pendulum']['improvement']:+.1f}%")
    
    return comparison

# Script principal
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Executar experimentos adaptativos')
    parser.add_argument('--env', choices=['cartpole', 'pendulum', 'both'], 
                       default='both', help='Ambiente para testar')
    parser.add_argument('--cycles', type=int, default=12, 
                       help='Número de ciclos de adaptação')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    if args.env == 'cartpole':
        results = run_cartpole_adaptive_experiment(args.cycles, args.save_dir)
        print("Experimento CartPole concluído.")
        
    elif args.env == 'pendulum':
        results = run_pendulum_adaptive_experiment(args.cycles, args.save_dir)
        print("Experimento Pendulum concluído.")
        
    elif args.env == 'both':
        results = compare_environments_adaptation(args.cycles)
        print("Comparação entre ambientes concluída.")
    
    print("Experimentos adaptativos finalizados!")