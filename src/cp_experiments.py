import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata

def load_experiment_data(exp_name, file_pattern):
    """Carrega dados de um experimento específico"""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/cartpole')
    file_path = os.path.join(base_path, exp_name, file_pattern)
    
    if os.path.exists(file_path):
        data = np.load(file_path)
        return data
    else:
        print(f"  ⚠️  Arquivo não encontrado: {file_path}")
        return None

def interpolate_to_grid(data, grid_resolution=50):
    """Interpola dados para uma grade comum"""
    X = data[:, 0]  # ângulo
    Y = data[:, 1]  # velocidade angular
    Z = data[:, 2]  # reward/fitness
    
    # Criar grade regular
    xi = np.linspace(X.min(), X.max(), grid_resolution)
    yi = np.linspace(Y.min(), Y.max(), grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolar valores
    zi_grid = griddata((X, Y), Z, (xi_grid, yi_grid), method='linear')
    
    return xi_grid, yi_grid, zi_grid

def calculate_squared_difference(control_data, exp_data):
    """Calcula a soma das diferenças quadráticas entre experimento e controle"""
    # Interpolar ambos para mesma grade
    xi_c, yi_c, zi_c = interpolate_to_grid(control_data)
    xi_e, yi_e, zi_e = interpolate_to_grid(exp_data)
    
    # Calcular diferença quadrática apenas onde há dados válidos
    mask = ~np.isnan(zi_c) & ~np.isnan(zi_e)
    squared_diff = np.nansum((zi_c[mask] - zi_e[mask])**2)
    
    return squared_diff

def plot_squared_differences():
    """Gera gráfico comparativo das diferenças quadráticas"""
    
    # Carregar experimento controle
    control_data = load_experiment_data('exp_controle', 'fitness_landscape.npy')
    
    if control_data is None:
        print("Erro: Experimento controle não encontrado!")
        return
    
    # Definir experimentos e seus arquivos por categoria
    exp_categories = {
        'Episódios': [],
        'Duração': [],
        'Ruído': [],
        'Cond. Iniciais': []
    }
    
    # Exp 1: número de episódios
    for ep in [2, 5, 10, 15, 20, 50]:
        exp_data = load_experiment_data('exp_1', f'states_rewards_ep_{ep}.npy')
        if exp_data is not None:
            diff = calculate_squared_difference(control_data, exp_data)
            exp_categories['Episódios'].append((ep, diff))
    
    # Exp 2: duração
    for d in [50, 100, 200, 300, 400, 500]:
        exp_data = load_experiment_data('exp_2', f'states_rewards_d_{d}.npy')
        if exp_data is not None:
            diff = calculate_squared_difference(control_data, exp_data)
            exp_categories['Duração'].append((d, diff))
    
    # Exp 3: ruído
    for n in [0.001, 0.01, 0.05, 0.1, 0.5, 1]:
        exp_data = load_experiment_data('exp_3', f'states_rewards_n_{str(n)}.npy')
        if exp_data is not None:
            diff = calculate_squared_difference(control_data, exp_data)
            exp_categories['Ruído'].append((n, diff))
    
    # Exp 4: condições iniciais
    theta_ranges = [3, 2, 1]
    theta_dot_ranges = [0.2, 0.1, 0.05]
    for t in theta_ranges:
        for td in theta_dot_ranges:
            ranges_str = f"{td}_{t}"
            exp_data = load_experiment_data('exp_4', f'states_rewards_r_{ranges_str}.npy')
            if exp_data is not None:
                diff = calculate_squared_difference(control_data, exp_data)
                exp_categories['Cond. Iniciais'].append((f"θ̇={td}, θ={t}", diff))
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CartPole: Soma das Diferenças Quadráticas vs Experimento Controle', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for idx, (category, ax) in enumerate(zip(exp_categories.keys(), axes.flat)):
        data = exp_categories[category]
        
        if not data:
            ax.set_visible(False)
            continue
        
        # Separar valores
        labels = [str(item[0]) for item in data]
        values = [item[1] for item in data]
        
        # Criar gráfico de linha com marcadores
        x_pos = range(len(labels))
        ax.plot(x_pos, values, marker='o', linewidth=2.5, markersize=8,
                color=colors[idx], alpha=0.8, label=category)
        ax.fill_between(x_pos, values, alpha=0.2, color=colors[idx])
        
        # Configurações
        ax.set_title(category, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Σ(Δ²)', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remover bordas
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adicionar valores nos pontos
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', 
                   fontsize=8, color=colors[idx], fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar
    save_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/plots/cartpole/squared_differences.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nGráfico salvo em: {save_path}")
    print("\n" + "="*70)
    print("CARTPOLE - DIFERENÇAS QUADRÁTICAS POR EXPERIMENTO")
    print("="*70)
    
    for category in exp_categories.keys():
        if exp_categories[category]:
            print(f"\n{category}:")
            for label, diff in exp_categories[category]:
                print(f"  {str(label):25s}: {diff:15.2f}")
    
    plt.show()

if __name__ == "__main__":
    plot_squared_differences()