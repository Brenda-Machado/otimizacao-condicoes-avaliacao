"""
Exemplo Prático de Uso do Sistema Adaptativo de Otimização de Parâmetros

Este script demonstra como usar o sistema completo para executar
experimentos adaptativos nos ambientes CartPole e Pendulum.
"""

from experiment_integration import AdaptiveExperimentRunner, run_cartpole_adaptive_experiment, run_pendulum_adaptive_experiment
from adaptive_experiment_optimizer import AdaptiveExperimentOptimizer
import numpy as np
import matplotlib.pyplot as plt

def exemplo_basico_cartpole():
    print("=== EXEMPLO BÁSICO - CARTPOLE ===\n")
    
    runner = AdaptiveExperimentRunner('cartpole', save_dir='./resultados_cartpole')
    
    resultados = runner.run_adaptive_training(n_adaptation_cycles=10)
    
    print(f"Melhor fitness obtido: {resultados['best_fitness']:.2f}")
    print(f"Melhores parâmetros encontrados:")
    for param, valor in resultados['best_params'].items():
        if param != 'custom_bounds':
            print(f"  {param}: {valor}")
    
    print("\nExecutando análise comparativa...")
    analise = runner.run_comparative_analysis()
    print(f"Melhoria percentual: {analise['improvement_percentage']:+.1f}%")
    
    runner.close()
    return resultados

def exemplo_basico_pendulum():
    print("=== EXEMPLO BÁSICO - PENDULUM ===\n")
    
    resultados = run_pendulum_adaptive_experiment(n_cycles=10, save_dir='./resultados_pendulum')
    
    print(f"Melhor fitness obtido: {resultados['best_fitness']:.2f}")
    print(f"Melhoria obtida: {resultados['comparative_analysis']['improvement_percentage']:+.1f}%")
    
    return resultados

def exemplo_avancado_personalizacao():
    print("=== EXEMPLO AVANÇADO - PERSONALIZAÇÃO ===\n")
    
    otimizador = AdaptiveExperimentOptimizer('cartpole', history_length=15)
    
    otimizador.param_network.param_ranges.update({
        'n_episodes': (5, 30),
        'max_steps': (100, 400),
        'noise_level': (0.01, 0.5),
    })
    
    runner = AdaptiveExperimentRunner('cartpole', save_dir='./resultados_personalizados')
    runner.optimizer = otimizador
    
    resultados = runner.run_adaptive_training(n_adaptation_cycles=15, save_frequency=3)
    
    print(f"Parâmetros finais otimizados:")
    for param, valor in resultados['final_optimizer_state'].items():
        print(f"  {param}: {valor}")
    
    runner.close()
    return resultados

def exemplo_monitoramento_convergencia():
    print("=== EXEMPLO - MONITORAMENTO DE CONVERGÊNCIA ===\n")
    
    runner = AdaptiveExperimentRunner('pendulum', save_dir='./convergencia_pendulum')
    
    resultados = runner.run_adaptive_training(n_adaptation_cycles=20, save_frequency=2)
    
    fitness_history = runner.adaptation_metrics['fitness_mean']
    
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', marker='o', linewidth=2, markersize=6)
    plt.title('Convergência do Fitness Durante Adaptação')
    plt.xlabel('Ciclo de Adaptação')
    plt.ylabel('Fitness Médio')
    plt.grid(True, alpha=0.3)
    
    z = np.polyfit(range(len(fitness_history)), fitness_history, 1)
    p = np.poly1d(z)
    plt.plot(range(len(fitness_history)), p(range(len(fitness_history))), "r--", alpha=0.8)
    
    plt.legend(['Fitness Real', 'Tendência Linear'])
    plt.tight_layout()
    plt.savefig('./convergencia_analysis.png', dpi=300)
    plt.show()
    
    if len(fitness_history) > 5:
        inicial = np.mean(fitness_history[:3])
        final = np.mean(fitness_history[-3:])
        convergencia = final - inicial
        print(f"Taxa de convergência: {convergencia:.2f}")
        print(f"Melhoria relativa: {(convergencia/abs(inicial)*100):+.1f}%")
    
    runner.close()
    return resultados

def exemplo_comparacao_estrategias():
    print("=== COMPARAÇÃO DE ESTRATÉGIAS ===\n")
    
    estrategias = {
        'conservadora': {'history_length': 20, 'cycles': 15},
        'agressiva': {'history_length': 5, 'cycles': 10},
        'balanceada': {'history_length': 10, 'cycles': 12}
    }
    
    resultados_estrategias = {}
    
    for nome, config in estrategias.items():
        print(f"Testando estratégia: {nome}")
        
        otimizador = AdaptiveExperimentOptimizer('cartpole', 
                                               history_length=config['history_length'])
        
        runner = AdaptiveExperimentRunner('cartpole', 
                                        save_dir=f'./estrategia_{nome}')
        runner.optimizer = otimizador
        
        resultado = runner.run_adaptive_training(n_adaptation_cycles=config['cycles'])
        
        resultados_estrategias[nome] = {
            'best_fitness': resultado['best_fitness'],
            'final_params': resultado['final_optimizer_state']
        }
        
        runner.close()
        print(f"Estratégia {nome}: Fitness = {resultado['best_fitness']:.2f}\n")
    
    melhor_estrategia = max(resultados_estrategias.items(), 
                           key=lambda x: x[1]['best_fitness'])
    
    print(f"Melhor estratégia: {melhor_estrategia[0]} "
          f"(Fitness: {melhor_estrategia[1]['best_fitness']:.2f})")
    
    return resultados_estrategias

def exemplo_transferencia_conhecimento():
    print("=== TRANSFERÊNCIA DE CONHECIMENTO ===\n")
    
    print("Fase 1: Treinamento no CartPole...")
    resultados_cartpole = run_cartpole_adaptive_experiment(
        n_cycles=12, save_dir='./transfer_cartpole'
    )
    
    params_aprendidos = resultados_cartpole['best_params']
    print(f"Parâmetros aprendidos no CartPole: {params_aprendidos}")
    
    print("\nFase 2: Aplicando conhecimento no Pendulum...")
    
    otimizador_pendulum = AdaptiveExperimentOptimizer('pendulum')
    
    params_transferidos = {
        'n_episodes': params_aprendidos['n_episodes'],
        'max_steps': params_aprendidos['max_steps'], 
        'noise_level': params_aprendidos['noise_level'] * 0.8,
        'theta_range': np.pi/2,
        'theta_dot_range': 4.0,
        'learning_rate': params_aprendidos['learning_rate']
    }
    
    otimizador_pendulum.current_params = params_transferidos
    
    runner_pendulum = AdaptiveExperimentRunner('pendulum', save_dir='./transfer_pendulum')
    runner_pendulum.optimizer = otimizador_pendulum
    
    resultados_pendulum = runner_pendulum.run_adaptive_training(n_cycles=8)
    
    print(f"Resultado com transferência: {resultados_pendulum['best_fitness']:.2f}")
    
    print("\nFase 3: Comparação com treinamento do zero...")
    resultados_zero = run_pendulum_adaptive_experiment(n_cycles=8, save_dir='./sem_transfer_pendulum')
    
    melhoria_transfer = resultados_pendulum['best_fitness'] - resultados_zero['best_fitness']
    
    print(f"Sem transferência: {resultados_zero['best_fitness']:.2f}")
    print(f"Com transferência: {resultados_pendulum['best_fitness']:.2f}")
    print(f"Benefício da transferência: {melhoria_transfer:+.2f}")
    
    runner_pendulum.close()
    
    return {
        'cartpole': resultados_cartpole,
        'pendulum_transfer': resultados_pendulum,
        'pendulum_zero': resultados_zero,
        'transfer_benefit': melhoria_transfer
    }

def exemplo_analise_sensibilidade():
    print("=== ANÁLISE DE SENSIBILIDADE ===\n")
    
    noise_levels = [0.05, 0.1, 0.2, 0.5]
    history_lengths = [5, 10, 15, 20]
    
    resultados_sensibilidade = {}
    
    for noise in noise_levels:
        print(f"Testando com noise_level = {noise}")
        
        otimizador = AdaptiveExperimentOptimizer('cartpole')
        otimizador.current_params['noise_level'] = noise
        
        runner = AdaptiveExperimentRunner('cartpole', save_dir=f'./sensitivity_noise_{noise}')
        runner.optimizer = otimizador
        
        resultado = runner.run_adaptive_training(n_adaptation_cycles=8)
        resultados_sensibilidade[f'noise_{noise}'] = resultado['best_fitness']
        
        runner.close()
    
    for history_len in history_lengths:
        print(f"Testando com history_length = {history_len}")
        
        otimizador = AdaptiveExperimentOptimizer('cartpole', history_length=history_len)
        
        runner = AdaptiveExperimentRunner('cartpole', save_dir=f'./sensitivity_history_{history_len}')
        runner.optimizer = otimizador
        
        resultado = runner.run_adaptive_training(n_adaptation_cycles=8)
        resultados_sensibilidade[f'history_{history_len}'] = resultado['best_fitness']
        
        runner.close()
    
    print("\nResultados da Análise de Sensibilidade:")
    for config, fitness in resultados_sensibilidade.items():
        print(f"{config}: {fitness:.2f}")
    
    return resultados_sensibilidade

def main():
    print("Sistema Adaptativo de Otimização de Parâmetros Experimentais")
    print("=" * 60)
    
    opcoes = {
        '1': ('Exemplo Básico - CartPole', exemplo_basico_cartpole),
        '2': ('Exemplo Básico - Pendulum', exemplo_basico_pendulum),
        '3': ('Personalização Avançada', exemplo_avancado_personalizacao),
        '4': ('Monitoramento de Convergência', exemplo_monitoramento_convergencia),
        '5': ('Comparação de Estratégias', exemplo_comparacao_estrategias),
        '6': ('Transferência de Conhecimento', exemplo_transferencia_conhecimento),
        '7': ('Análise de Sensibilidade', exemplo_analise_sensibilidade),
        '8': ('Executar Todos', None)
    }
    
    print("Escolha um exemplo para executar:")
    for key, (desc, _) in opcoes.items():
        print(f"{key}. {desc}")
    
    escolha = input("\nDigite o número da opção: ").strip()
    
    if escolha == '8':
        print("Executando todos os exemplos...\n")
        for key, (desc, func) in opcoes.items():
            if func is not None:
                print(f"\n{'-'*50}")
                print(f"Executando: {desc}")
                print(f"{'-'*50}")
                try:
                    func()
                except Exception as e:
                    print(f"Erro ao executar {desc}: {e}")
    elif escolha in opcoes and opcoes[escolha][1] is not None:
        desc, func = opcoes[escolha]
        print(f"\nExecutando: {desc}")
        try:
            resultado = func()
            print(f"\n{desc} concluído com sucesso!")
        except Exception as e:
            print(f"Erro: {e}")
    else:
        print("Opção inválida!")

if __name__ == "__main__":
    main()