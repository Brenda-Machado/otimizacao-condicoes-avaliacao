"""

This file is based on the one belonging to https://github.com/snolfi/evorobotpy
   which was written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it
   salimans.py include an implementation of the OpenAI-ES algorithm described in
   Salimans T., Ho J., Chen X., Sidor S & Sutskever I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864v2

---

Author: Brenda Silva Machado.

OpenAI-ES

"""

import numpy as np

class OpenAIES:
    def __init__(self, param_count, population_size=20, learning_rate=0.01, noise_std=0.02, weight_decay=0.0, seed=42):
        self.param_count = param_count
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.weight_decay = weight_decay
        self.seed = seed
        
        np.random.seed(seed)
        self.center = np.random.randn(param_count) * 0.1
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = np.zeros(param_count)
        self.v = np.zeros(param_count)
        
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_params = self.center.copy()
        
    def ask(self):
        rs = np.random.RandomState(self.seed + self.generation)
        noise = rs.randn(self.population_size, self.param_count)
        
        samples = []

        for i in range(self.population_size):
            samples.append(self.center + noise[i] * self.noise_std)
            samples.append(self.center - noise[i] * self.noise_std)
            
        return samples, noise
    
    def tell(self, fitness_list, noise):
        fitness_list = np.array(fitness_list)
        
        indices = np.argsort(fitness_list)
        utilities = np.zeros(len(fitness_list))

        for i, idx in enumerate(indices):
            utilities[idx] = i / (len(fitness_list) - 1) - 0.5
            
        weights = np.zeros(self.population_size)

        for i in range(self.population_size):
            pos_idx = 2 * i
            neg_idx = 2 * i + 1
            weights[i] = utilities[pos_idx] - utilities[neg_idx]
        
        gradient = np.dot(weights, noise) / len(fitness_list)
        
        if self.weight_decay > 0:
            gradient -= self.weight_decay * self.center
        
        self.generation += 1
        a = self.learning_rate * np.sqrt(1 - self.beta2 ** self.generation) / (1 - self.beta1 ** self.generation)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        step = a * self.m / (np.sqrt(self.v) + self.epsilon)
        self.center += step
        
        best_idx = np.argmax(fitness_list)
        if fitness_list[best_idx] > self.best_fitness:
            self.best_fitness = fitness_list[best_idx]
            sample_idx = best_idx // 2
            if best_idx % 2 == 0:
                self.best_params = self.center + noise[sample_idx] * self.noise_std
            else:
                self.best_params = self.center - noise[sample_idx] * self.noise_std
    
    def get_best_params(self):
        return self.best_params.copy()
    
    def get_current_params(self):
        return self.center.copy()

