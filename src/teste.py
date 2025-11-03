"""
Visualizações adicionais para análise de fitness landscapes.
Sem alterar as funções existentes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_controle_multiangulo(results_path):
    """Plota o experimento controle em múltiplos ângulos."""
    results = np.load(results_path)
    X, Y, Z = results[:, 0], results[:, 1], results[:, 2]
    
    angulos = [
        (30, 45, "padrao"),
        (0, 0, "frontal"),
        (0, 90, "lateral"),
        (90, 0, "topo"),
        (45, 135, "diagonal1"),
        (60, 225, "diagonal2")
    ]
    
    fig = plt.figure(figsize=(18, 12))
    
    for idx, (elev, azim, nome) in enumerate(angulos, 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        ax.plot_trisurf(X, Y, Z, cmap='Blues', alpha=0.8)
        ax.set_xlabel('Theta')
        ax.set_ylabel('Theta_dot')
        ax.set_zlabel('Fitness')
        ax.set_title(f'Visão {nome} (elev={elev}, azim={azim})')
        ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    save_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/plots/pendulum/exp_controle/multiangulo.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_diff(ref_results, var_results):
    """Calcula diferença entre landscapes."""
    ref_dict = {(round(r[0], 6), round(r[1], 6)): r[2] for r in ref_results}
    
    diffs = []
    for var_r in var_results:
        key = (round(var_r[0], 6), round(var_r[1], 6))
        if key in ref_dict:
            diff = var_r[2] - ref_dict[key]
            diffs.append([var_r[0], var_r[1], diff])
    
    if len(diffs) == 0:
        return None
    
    return np.array(diffs)

def plot_diff_3d(diff_results, exp, name):
    """Plota diferença em 3D."""
    if diff_results is None or len(diff_results) == 0:
        print(f"Aviso: Sem dados para plotar diff 3D em {exp}/{name}")
        return
    
    X, Y, Z = diff_results[:, 0], diff_results[:, 1], diff_results[:, 2]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_trisurf(X, Y, Z, cmap='RdBu_r', alpha=0.8)
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_zlabel('Diff Fitness')
    ax.set_title(f'Diferença: {name} vs Controle')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    save_path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}/diff_3d_{name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_diff_heatmap(diff_results, exp, name):
    """Plota diferença como heatmap 2D."""
    if diff_results is None or len(diff_results) == 0:
        print(f"Aviso: Sem dados para plotar heatmap em {exp}/{name}")
        return
    
    X, Y, Z = diff_results[:, 0], diff_results[:, 1], diff_results[:, 2]
    
    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    
    Z_grid = np.full((len(y_unique), len(x_unique)), np.nan)
    
    for i, x_val in enumerate(x_unique):
        for j, y_val in enumerate(y_unique):
            mask = (X == x_val) & (Y == y_val)
            if mask.any():
                Z_grid[j, i] = Z[mask][0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(Z_grid, cmap='RdBu_r', aspect='auto', 
                   extent=[x_unique.min(), x_unique.max(), 
                          y_unique.min(), y_unique.max()],
                   origin='lower')
    
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_title(f'Heatmap Diferença: {name} vs Controle')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Diff Fitness')
    
    save_path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/{exp}/diff_heatmap_{name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_experiment_1():
    """Compara variações de episódios com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    episodios = [2, 5, 10, 15, 20, 50]
    
    for ep in episodios:
        var = np.load(f'{base_path}/exp_1/fitness_landscape_ep_{ep}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_1', f'ep_{ep}')
        plot_diff_heatmap(diff, 'exp_1', f'ep_{ep}')

def compare_experiment_2():
    """Compara variações de duração com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    duracoes = [50, 100, 200, 300, 400, 500]
    
    for d in duracoes:
        var = np.load(f'{base_path}/exp_2/fitness_landscape_d_{d}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_2', f'mstep_{d}')
        plot_diff_heatmap(diff, 'exp_2', f'mstep_{d}')

def compare_experiment_3():
    """Compara variações de ruído com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    noises = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    
    for n in noises:
        var = np.load(f'{base_path}/exp_3/fitness_landscape_n_{str(n)}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_3', f'noise_{str(n)}')
        plot_diff_heatmap(diff, 'exp_3', f'noise_{str(n)}')

def compare_experiment_4():
    """Compara variações de condições iniciais com controle."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    interval_ranges_theta = [np.pi/4, np.pi/2, np.pi]
    interval_ranges_theta_dot = [2.0, 4.0, 8.0]
    
    for t in interval_ranges_theta:
        for td in interval_ranges_theta_dot:
            ranges = f"{t:.2f}_{td}"
            var = np.load(f'{base_path}/exp_4/fitness_landscape_r_{ranges}.npy')
            diff = compute_diff(ref, var)
            
            plot_diff_3d(diff, 'exp_4', f'ranges_{ranges.replace(".", "_")}')
            plot_diff_heatmap(diff, 'exp_4', f'ranges_{ranges.replace(".", "_")}')

def compare_experiment_5():
    """Compara diferentes métricas de fitness."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_5/fitness_landscape_mean.npy')
    
    methods = ['min', 'max', 'median', 'std']
    
    for method in methods:
        var = np.load(f'{base_path}/exp_5/fitness_landscape_{method}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_5', f'fitness_{method}')
        plot_diff_heatmap(diff, 'exp_5', f'fitness_{method}')

def compare_experiment_6():
    """Compara diferentes pesos."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_6/fitness_landscape_w_010000.npy')
    
    pesos = ['100000', '001000', '030403', '050302', '010207']
    
    for p in pesos:
        var = np.load(f'{base_path}/exp_6/fitness_landscape_w_{p}.npy')
        diff = compute_diff(ref, var)
        
        plot_diff_3d(diff, 'exp_6', f'pesos_{p}')
        plot_diff_heatmap(diff, 'exp_6', f'pesos_{p}')

def create_comparison_grid(exp_num, variations):
    """Cria grade comparativa para um experimento."""
    base_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum')
    ref = np.load(f'{base_path}/exp_controle/fitness_landscape.npy')
    
    n_vars = len(variations)
    rows = (n_vars + 2) // 3
    
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
    axes = axes.flatten()
    
    for idx, (path, label) in enumerate(variations):
        try:
            var = np.load(f'{base_path}/{path}')
            diff = compute_diff(ref, var)
            
            if diff is None or len(diff) == 0:
                axes[idx].text(0.5, 0.5, 'Sem dados', ha='center', va='center')
                axes[idx].set_title(label)
                continue
            
            X, Y, Z = diff[:, 0], diff[:, 1], diff[:, 2]
            x_unique = np.unique(X)
            y_unique = np.unique(Y)
            
            Z_grid = np.full((len(y_unique), len(x_unique)), np.nan)
            for i, x_val in enumerate(x_unique):
                for j, y_val in enumerate(y_unique):
                    mask = (X == x_val) & (Y == y_val)
                    if mask.any():
                        Z_grid[j, i] = Z[mask][0]
            
            im = axes[idx].imshow(Z_grid, cmap='RdBu_r', aspect='auto',
                                 extent=[x_unique.min(), x_unique.max(),
                                        y_unique.min(), y_unique.max()],
                                 origin='lower')
            axes[idx].set_title(label)
            axes[idx].set_xlabel('Theta')
            axes[idx].set_ylabel('Theta_dot')
            plt.colorbar(im, ax=axes[idx])
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Erro: {str(e)[:30]}', ha='center', va='center')
            axes[idx].set_title(label)
    
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.expanduser(f'~/otimizacao-condicoes-avaliacao/plots/pendulum/exp_{exp_num}/comparison_grid.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_all_visualizations():
    # """Executa todas as visualizações adicionais."""
    
    # # Multi-ângulo controle
    # print("Gerando visualizações multi-ângulo do controle...")
    # controle_path = os.path.expanduser('~/otimizacao-condicoes-avaliacao/data/pendulum/exp_controle/fitness_landscape.npy')
    # plot_controle_multiangulo(controle_path)
    
    # # Comparações exp 1
    # print("Comparando experimento 1 (episódios)...")
    # compare_experiment_1()
    # variations_1 = [(f'exp_1/fitness_landscape_ep_{ep}.npy', f'{ep} eps') 
    #                 for ep in [2, 5, 10, 15, 20, 50]]
    # create_comparison_grid(1, variations_1)
    
    # # Comparações exp 2
    # print("Comparando experimento 2 (duração)...")
    # compare_experiment_2()
    # variations_2 = [(f'exp_2/fitness_landscape_d_{d}.npy', f'{d} steps') 
    #                 for d in [50, 100, 200, 300, 400, 500]]
    # create_comparison_grid(2, variations_2)
    
    # # Comparações exp 3
    # print("Comparando experimento 3 (ruído)...")
    # compare_experiment_3()
    # variations_3 = [(f'exp_3/fitness_landscape_n_{str(n)}.npy', f'noise={n}') 
    #                 for n in [0.001, 0.01, 0.05, 0.1, 0.5, 1]]
    # create_comparison_grid(3, variations_3)
    
    # Comparações exp 4
    print("Comparando experimento 4 (condições iniciais)...")
    compare_experiment_4()
    variations_4 = []
    for t in [np.pi/4, np.pi/2, np.pi]:
        for td in [2.0, 4.0, 8.0]:
            ranges = f"{t:.2f}_{td}"
            variations_4.append((f'exp_4/fitness_landscape_r_{ranges}.npy', 
                               f'θ±{t:.2f}, θ_dot±{td}'))
    create_comparison_grid(4, variations_4)
    
    print("Visualizações concluídas!")

if __name__ == "__main__":
    
    run_all_visualizations()