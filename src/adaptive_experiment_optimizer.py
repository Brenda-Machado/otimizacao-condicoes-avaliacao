"""
Algoritmo Adaptativo para Otimização Online de Parâmetros Experimentais

Este módulo implementa uma rede neural simples que ajusta dinamicamente parâmetros
experimentais (número de episódios, duração, ruído, condições iniciais) durante o
treinamento online de agentes autônomos.

Compatível com CartPole e Pendulum.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import json
import os

class ExperimentParameterNetwork(nn.Module):
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 6):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size),
            nn.Sigmoid()
        )
        
        self.param_mapping = {
            0: 'n_episodes',
            1: 'max_steps',
            2: 'noise_level',
            3: 'theta_range',
            4: 'theta_dot_range',
            5: 'learning_rate'
        }
        
        self.param_ranges = {
            'n_episodes': (2, 50),
            'max_steps': (50, 500), 
            'noise_level': (0.001, 1.0),
            'theta_range': (0.1, 3.14),
            'theta_dot_range': (0.5, 8.0),
            'learning_rate': (0.0001, 0.1)
        }
    
    def forward(self, x):
        return self.network(x)
    
    def denormalize_params(self, normalized_output):
        params = {}
        for i, (param_name, (min_val, max_val)) in enumerate(zip(self.param_mapping.values(), self.param_ranges.values())):
            normalized_val = normalized_output[i].item()
            if param_name in ['n_episodes', 'max_steps']:
                params[param_name] = int(min_val + normalized_val * (max_val - min_val))
            else:
                params[param_name] = min_val + normalized_val * (max_val - min_val)
        
        return params

class AdaptiveExperimentOptimizer:
    
    def __init__(self, env_name: str = 'cartpole', history_length: int = 10):
        self.env_name = env_name.lower()
        self.history_length = history_length
        
        self.param_network = ExperimentParameterNetwork()
        self.optimizer = optim.Adam(self.param_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.performance_history = deque(maxlen=history_length)
        self.param_history = deque(maxlen=history_length)
        self.fitness_history = deque(maxlen=history_length)
        
        self.current_params = self._get_default_params()
        
        self.training_metrics = {
            'episodes': [],
            'avg_fitness': [],
            'param_changes': [],
            'prediction_errors': []
        }
    
    def _get_default_params(self) -> Dict:
        return {
            'n_episodes': 10,
            'max_steps': 500,
            'noise_level': 0.1,
            'theta_range': 0.15 if self.env_name == 'cartpole' else np.pi/2,
            'theta_dot_range': 2.0 if self.env_name == 'cartpole' else 4.0,
            'learning_rate': 0.01
        }
    
    def _create_feature_vector(self) -> torch.Tensor:
        if len(self.performance_history) < 3:
            features = np.array([0.0] * self.history_length)
        else:
            recent_fitness = list(self.fitness_history)[-self.history_length:]
            
            while len(recent_fitness) < self.history_length:
                recent_fitness.insert(0, 0.0)
            
            features = np.array(recent_fitness)
        
        if np.std(features) > 0:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def predict_optimal_params(self) -> Dict:
        self.param_network.eval()
        
        with torch.no_grad():
            features = self._create_feature_vector()
            normalized_output = self.param_network(features)[0]
            optimal_params = self.param_network.denormalize_params(normalized_output)
        
        return optimal_params
    
    def update_performance(self, fitness_results: List[float], experiment_type: str = 'training'):
        avg_fitness = np.mean(fitness_results)
        std_fitness = np.std(fitness_results)
        
        performance_metrics = {
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'min_fitness': np.min(fitness_results),
            'max_fitness': np.max(fitness_results),
            'experiment_type': experiment_type,
            'n_samples': len(fitness_results)
        }
        
        self.performance_history.append(performance_metrics)
        self.fitness_history.append(avg_fitness)
        self.param_history.append(self.current_params.copy())
        
        self.training_metrics['episodes'].append(len(self.performance_history))
        self.training_metrics['avg_fitness'].append(avg_fitness)
    
    def train_parameter_predictor(self, n_iterations: int = 5):
        if len(self.performance_history) < 5:
            print("Histórico insuficiente para treinamento. Aguardando mais dados...")
            return
        
        self.param_network.train()
        
        for iteration in range(n_iterations):
            total_loss = 0.0
            
            for i in range(2, len(self.performance_history)):
                past_fitness = [self.fitness_history[j] for j in range(max(0, i-self.history_length), i)]
                while len(past_fitness) < self.history_length:
                    past_fitness.insert(0, 0.0)
                
                features = torch.FloatTensor(past_fitness).unsqueeze(0)
                
                future_fitness = [self.fitness_history[j] for j in range(i, min(len(self.fitness_history), i+3))]
                if future_fitness:
                    best_future_idx = i + np.argmax(future_fitness)
                    target_params = self.param_history[best_future_idx]
                    
                    normalized_targets = []
                    for param_name, value in target_params.items():
                        if param_name in self.param_network.param_ranges:
                            min_val, max_val = self.param_network.param_ranges[param_name]
                            normalized_val = (value - min_val) / (max_val - min_val)
                            normalized_targets.append(np.clip(normalized_val, 0, 1))
                    
                    target = torch.FloatTensor(normalized_targets).unsqueeze(0)
                    
                    predicted = self.param_network(features)
                    loss = self.criterion(predicted, target)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(self.performance_history) - 2)
            self.training_metrics['prediction_errors'].append(avg_loss)
            
            if iteration % 2 == 0:
                print(f"Iteração {iteration+1}/{n_iterations}, Loss: {avg_loss:.4f}")
    
    def adapt_parameters(self) -> Dict:
        if len(self.performance_history) < 3:
            print("Usando parâmetros padrão (histórico insuficiente)")
            return self.current_params
        
        if len(self.performance_history) % 5 == 0:
            print("Retreinando modelo de predição...")
            self.train_parameter_predictor()
        
        predicted_params = self.predict_optimal_params()
        
        alpha = 0.3
        adapted_params = {}
        
        for param_name, new_value in predicted_params.items():
            old_value = self.current_params[param_name]
            adapted_params[param_name] = alpha * new_value + (1 - alpha) * old_value
            
            if param_name in ['n_episodes', 'max_steps']:
                adapted_params[param_name] = int(adapted_params[param_name])
        
        param_changes = {k: abs(adapted_params[k] - self.current_params[k]) 
                        for k in adapted_params.keys()}
        self.training_metrics['param_changes'].append(param_changes)
        
        self.current_params = adapted_params
        
        print(f"Parâmetros adaptados: {adapted_params}")
        return adapted_params
    
    def get_experiment_config(self) -> Dict:
        config = self.current_params.copy()
        
        if self.env_name == 'cartpole':
            config['theta_range'] = min(config['theta_range'], 0.2)
            config['custom_bounds'] = (config['theta_range'], config['theta_dot_range'])
        else:
            config['custom_bounds'] = (config['theta_range'], config['theta_dot_range'])
        
        return config
    
    def save_state(self, filepath: str):
        state = {
            'current_params': self.current_params,
            'performance_history': list(self.performance_history),
            'param_history': list(self.param_history),
            'fitness_history': list(self.fitness_history),
            'training_metrics': self.training_metrics,
            'env_name': self.env_name
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        model_path = filepath.replace('.json', '_model.pth')
        torch.save(self.param_network.state_dict(), model_path)
        
        print(f"Estado salvo em: {filepath}")
    
    def load_state(self, filepath: str):
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_params = state['current_params']
        self.performance_history = deque(state['performance_history'], maxlen=self.history_length)
        self.param_history = deque(state['param_history'], maxlen=self.history_length)
        self.fitness_history = deque(state['fitness_history'], maxlen=self.history_length)
        self.training_metrics = state['training_metrics']
        
        model_path = filepath.replace('.json', '_model.pth')
        if os.path.exists(model_path):
            self.param_network.load_state_dict(torch.load(model_path))
        
        print(f"Estado carregado de: {filepath}")
    
    def plot_adaptation_progress(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = self.training_metrics['episodes']
        
        axes[0,0].plot(episodes, self.training_metrics['avg_fitness'], 'b-', marker='o')
        axes[0,0].set_title('Evolução do Fitness Médio')
        axes[0,0].set_xlabel('Episódios de Treinamento')
        axes[0,0].set_ylabel('Fitness Médio')
        axes[0,0].grid(True)
        
        if self.param_history:
            param_names = ['n_episodes', 'max_steps', 'noise_level']
            for param in param_names:
                values = [p.get(param, 0) for p in self.param_history]
                if values:
                    axes[0,1].plot(range(len(values)), values, marker='o', label=param)
            
            axes[0,1].set_title('Evolução dos Parâmetros')
            axes[0,1].set_xlabel('Iterações')
            axes[0,1].set_ylabel('Valor do Parâmetro')
            axes[0,1].legend()
            axes[0,1].grid(True)
        
        if self.training_metrics['prediction_errors']:
            axes[1,0].plot(self.training_metrics['prediction_errors'], 'r-', marker='s')
            axes[1,0].set_title('Erro de Predição da Rede Neural')
            axes[1,0].set_xlabel('Iterações de Treinamento')
            axes[1,0].set_ylabel('MSE Loss')
            axes[1,0].grid(True)
        
        if self.training_metrics['param_changes']:
            total_changes = [sum(change_dict.values()) for change_dict in self.training_metrics['param_changes']]
            axes[1,1].plot(total_changes, 'g-', marker='^')
            axes[1,1].set_title('Magnitude das Mudanças de Parâmetros')
            axes[1,1].set_xlabel('Adaptações')
            axes[1,1].set_ylabel('Soma das Mudanças')
            axes[1,1].grid(True)
        
        plt.suptitle(f'Progresso da Adaptação - {self.env_name.upper()}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def run_adaptive_experiment(env, policy, optimizer: AdaptiveExperimentOptimizer, 
                          n_adaptation_cycles: int = 10):
    
    print(f"=== EXPERIMENTO ADAPTATIVO - {optimizer.env_name.upper()} ===")
    
    for cycle in range(n_adaptation_cycles):
        print(f"\n--- Ciclo de Adaptação {cycle + 1}/{n_adaptation_cycles} ---")
        
        config = optimizer.get_experiment_config()
        print(f"Configuração atual: {config}")
        
        fitness_results = []
        
        for episode in range(config['n_episodes']):
            
            if optimizer.env_name == 'cartpole':
                base_fitness = np.random.uniform(10, 500)
                noise_penalty = config['noise_level'] * 50
                fitness = max(1, base_fitness - noise_penalty)
            else:
                base_fitness = np.random.uniform(-1500, -200)
                noise_penalty = config['noise_level'] * 200
                fitness = base_fitness - noise_penalty
            
            fitness_results.append(fitness)
        
        optimizer.update_performance(fitness_results, f'adaptive_cycle_{cycle}')
        
        if cycle < n_adaptation_cycles - 1:
            optimizer.adapt_parameters()
        
        print(f"Fitness médio: {np.mean(fitness_results):.2f}")
    
    save_path = f"~/adaptive_optimizer_{optimizer.env_name}_final.json"
    save_path = os.path.expanduser(save_path)
    optimizer.save_state(save_path)
    
    plot_path = save_path.replace('.json', '_progress.png')
    optimizer.plot_adaptation_progress(plot_path)
    
    return optimizer

if __name__ == "__main__":
    print("Testando otimizador adaptativo com CartPole...")
    cartpole_optimizer = AdaptiveExperimentOptimizer('cartpole')
    
    run_adaptive_experiment(None, None, cartpole_optimizer, n_adaptation_cycles=8)
    
    print("\n" + "="*60 + "\n")
    
    print("Testando otimizador adaptativo com Pendulum...")
    pendulum_optimizer = AdaptiveExperimentOptimizer('pendulum')
    
    run_adaptive_experiment(None, None, pendulum_optimizer, n_adaptation_cycles=8)
    