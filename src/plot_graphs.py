"""

Author: Brenda Silva Machado.

plot_graphs.py

"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def quick_plot_evolution(history_file='evolution_history.pkl'):
    plt.style.use('default')
    sns.set_style("whitegrid")
    sns.set_palette("Set1")
    
    try:
        with open(history_file, 'rb') as f:
            data = pickle.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Evolução OpenAI-ES - Pendulum vs CartPole', fontsize=16)
        
        environments = ['pendulum', 'cartpole']
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, (env_name, color) in enumerate(zip(environments, colors)):
            env_data = data[env_name]
            
            generations = [h['generation'] for h in env_data]
            avg_fitness = [h['avg_fitness'] for h in env_data]
            best_fitness = [h['best_fitness'] for h in env_data]
            max_fitness = [h['max_fitness'] for h in env_data]
            
            axes[i].plot(generations, best_fitness, 
                        color=color, linewidth=3, label='Melhor Fitness', alpha=0.9)
            axes[i].plot(generations, avg_fitness, 
                        color=color, linewidth=2, linestyle='--', label='Fitness Médio', alpha=0.7)
            axes[i].fill_between(generations, avg_fitness, best_fitness, 
                                alpha=0.2, color=color)
            
            axes[i].set_title(f'{env_name.capitalize()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Geração')
            axes[i].set_ylabel('Fitness')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            final_best = best_fitness[-1]
            final_avg = avg_fitness[-1]
            axes[i].annotate(f'Final: {final_best:.2f}', 
                           xy=(generations[-1], final_best),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color=color))
        
        plt.tight_layout()
        plt.savefig('quick_evolution_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        for env_name in environments:
            env_data = data[env_name]
            initial_fitness = env_data[0]['best_fitness']
            final_fitness = env_data[-1]['best_fitness']
            max_fitness = max([h['best_fitness'] for h in env_data])
            improvement = final_fitness - initial_fitness
            
            print(f"\n{env_name.capitalize()}:")
            print(f"  Fitness inicial: {initial_fitness:.3f}")
            print(f"  Fitness final:   {final_fitness:.3f}")
            print(f"  Melhor fitness:  {max_fitness:.3f}")
            print(f"  Melhoria:        {improvement:.3f} ({improvement/abs(initial_fitness)*100:+.1f}%)")
        
    except FileNotFoundError:
        print(f"Arquivo {history_file} não encontrado!")
    except Exception as e:
        print(f"Erro: {e}")


def compare_final_performance(history_file='evolution_history.pkl'):
    try:
        with open(history_file, 'rb') as f:
            data = pickle.load(f)
        
        results = {}
        for env_name in ['pendulum', 'cartpole']:
            env_data = data[env_name]
            results[env_name] = {
                'Final Best': env_data[-1]['best_fitness'],
                'Final Avg': env_data[-1]['avg_fitness'],
                'Overall Best': max([h['best_fitness'] for h in env_data]),
                'Overall Avg': np.mean([h['avg_fitness'] for h in env_data])
            }
        
        df = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        df_melted = df.reset_index().melt(id_vars='index', var_name='Métrica', value_name='Fitness')
        df_melted = df_melted.rename(columns={'index': 'Ambiente'})
        
        sns.barplot(data=df_melted, x='Ambiente', y='Fitness', hue='Métrica', ax=axes[0])
        axes[0].set_title('Comparação de Performance Final')
        axes[0].set_ylabel('Fitness')
        
        sns.heatmap(df.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1])
        axes[1].set_title('Heatmap de Performance')
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Erro: {e}")


def plot_learning_curves(history_file='evolution_history.pkl'):
    try:
        with open(history_file, 'rb') as f:
            data = pickle.load(f)
        
        all_data = []
        for env_name in ['pendulum', 'cartpole']:
            for entry in data[env_name]:
                all_data.extend([
                    {
                        'Environment': env_name.capitalize(),
                        'Generation': entry['generation'],
                        'Fitness': entry['best_fitness'],
                        'Type': 'Melhor'
                    },
                    {
                        'Environment': env_name.capitalize(),
                        'Generation': entry['generation'],
                        'Fitness': entry['avg_fitness'],
                        'Type': 'Médio'
                    }
                ])
        
        df = pd.DataFrame(all_data)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        sns.lineplot(data=df, x='Generation', y='Fitness', 
                    hue='Environment', style='Type', linewidth=2.5)
        plt.title('Curvas de Aprendizado - OpenAI-ES', fontsize=14, fontweight='bold')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        best_df = df[df['Type'] == 'Melhor']
        sns.lineplot(data=best_df, x='Generation', y='Fitness', 
                    hue='Environment', linewidth=3, marker='o', markersize=4)
        plt.title('Evolução do Melhor Fitness', fontsize=14, fontweight='bold')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    quick_plot_evolution()
    compare_final_performance()
    plot_learning_curves()
