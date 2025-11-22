import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree

def load_experiment_data(exp_name, file_pattern):
    """Carrega dados de um experimento específico"""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    file_path = os.path.join(base_path, exp_name, file_pattern)
    
    if os.path.exists(file_path):
        return np.load(file_path)
    return None

def calculate_squared_difference_aligned(control_data, exp_data):
    """
    Calcula diferenças quadráticas usando nearest neighbor matching.
    Para cada ponto do controle, encontra o ponto mais próximo no experimento.
    """
    # Separar coordenadas e valores
    control_coords = control_data[:, :2]  # (theta, theta_dot)
    control_fitness = control_data[:, 2]
    
    exp_coords = exp_data[:, :2]
    exp_fitness = exp_data[:, 2]
    
    # Construir KD-tree para busca eficiente de vizinhos
    tree = cKDTree(exp_coords)
    
    # Para cada ponto do controle, encontrar o vizinho mais próximo no experimento
    distances, indices = tree.query(control_coords, k=1)
    
    # Calcular diferenças quadráticas apenas para pontos suficientemente próximos
    # (distância máxima = 0.2 para considerar como "mesmo estado")
    threshold = 0.2
    valid_mask = distances < threshold
    
    if np.sum(valid_mask) == 0:
        print("  ⚠️  Nenhum ponto correspondente encontrado!")
        return np.nan
    
    # Diferenças quadráticas dos pontos válidos
    fitness_diff = control_fitness[valid_mask] - exp_fitness[indices[valid_mask]]
    squared_diff = np.sum(fitness_diff ** 2)
    
    # Normalizar pelo número de pontos para comparação justa
    normalized_diff = squared_diff / np.sum(valid_mask)
    
    print(f"    Pontos correspondentes: {np.sum(valid_mask)}/{len(control_coords)}")
    
    return normalized_diff

def plot_squared_differences():
    """Gera gráfico comparativo das diferenças quadráticas"""
    
    # Carregar experimento controle
    control_data = load_experiment_data('exp_controle', 'fitness_landscape.npy')
    
    if control_data is None:
        print("Erro: Experimento controle não encontrado!")
        return
    
    print(f"Controle: {len(control_data)} pontos\n")
    
    # Definir experimentos e seus arquivos por categoria
    exp_categories = {
        'Episódios': [],
        'Duração': [],
        'Ruído': [],
        'Cond. Iniciais': []
    }
    
    # Exp 1: número de episódios
    print("Processando Experimento 1 (Episódios):")
    for ep in [2, 5, 10, 15, 20, 50]:
        exp_data = load_experiment_data('exp_1', f'fitness_landscape_ep_{ep}.npy')
        if exp_data is not None:
            print(f"  Ep={ep}: {len(exp_data)} pontos")
            diff = calculate_squared_difference_aligned(control_data, exp_data)
            if not np.isnan(diff):
                exp_categories['Episódios'].append((ep, diff))
    
    # Exp 2: duração
    print("\nProcessando Experimento 2 (Duração):")
    for d in [50, 100, 200, 300, 400, 500]:
        exp_data = load_experiment_data('exp_2', f'fitness_landscape_d_{d}.npy')
        if exp_data is not None:
            print(f"  Steps={d}: {len(exp_data)} pontos")
            diff = calculate_squared_difference_aligned(control_data, exp_data)
            if not np.isnan(diff):
                exp_categories['Duração'].append((d, diff))
    
    # Exp 3: ruído
    print("\nProcessando Experimento 3 (Ruído):")
    for n in [0.001, 0.01, 0.05, 0.1, 0.5, 1]:
        exp_data = load_experiment_data('exp_3', f'fitness_landscape_n_{str(n)}.npy')
        if exp_data is not None:
            print(f"  Noise={n}: {len(exp_data)} pontos")
            diff = calculate_squared_difference_aligned(control_data, exp_data)
            if not np.isnan(diff):
                exp_categories['Ruído'].append((n, diff))
    
    # Exp 4: condições iniciais
    print("\nProcessando Experimento 4 (Condições Iniciais):")
    theta_ranges = [np.pi/4, np.pi/2, np.pi]
    theta_dot_ranges = [2.0, 4.0, 8.0]
    for t in theta_ranges:
        for td in theta_dot_ranges:
            ranges_str = f"{t:.2f}_{td}"
            exp_data = load_experiment_data('exp_4', f'fitness_landscape_r_{ranges_str}.npy')
            if exp_data is not None:
                # label = f"θ={t:.2f}, θ̇={td:.1f}"
                label = fr"$\theta={t:.2f}, \dot{{\theta}}={td:.1f}$"
                print(f"  {label}: {len(exp_data)} pontos")
                diff = calculate_squared_difference_aligned(control_data, exp_data)
                if not np.isnan(diff):
                    exp_categories['Cond. Iniciais'].append((label, diff))
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pendulum: Diferenças Quadráticas Normalizadas vs Experimento Controle', 
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
        ax.set_ylabel('Σ(Δ²) / n_pontos', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remover bordas
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adicionar valores nos pontos (formato científico se necessário)
        for i, v in enumerate(values):
            if v > 1000:
                ax.text(i, v, f'{v:.1e}', ha='center', va='bottom', 
                       fontsize=8, color=colors[idx], fontweight='bold')
            else:
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', 
                       fontsize=8, color=colors[idx], fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar
    save_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/plots/pendulum/squared_differences_corrected.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n{'='*70}")
    print("DIFERENÇAS QUADRÁTICAS NORMALIZADAS POR EXPERIMENTO")
    print(f"{'='*70}")
    
    for category in exp_categories.keys():
        if exp_categories[category]:
            print(f"\n{category}:")
            for label, diff in exp_categories[category]:
                print(f"  {str(label):25s}: {diff:15.6f}")
    
    print(f"\nGráfico salvo em: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_squared_differences()