"""
Advanced Policy with Gated Recurrent Unit (GRU) Architecture

Based on: Pagliuca, P., Milano, N., & Nolfi, S. (2020). "Efficacy of modern 
neuroevolutionary strategies for continuous control optimization". 
Frontiers in Robotics and AI, 7, 98.


Author: Brenda Silva Machado

gru_policy.py
"""

import numpy as np
import time

class GRUCell:

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ir = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hr = np.random.randn(hidden_size, hidden_size) * 0.05
        self.b_r = np.random.randn(hidden_size) * 0.1
        self.W_iz = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hz = np.random.randn(hidden_size, hidden_size) * 0.05
        self.b_z = np.random.randn(hidden_size) * 0.1
        self.W_ih = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.05
        self.b_h = np.random.randn(hidden_size) * 0.1
        self.h = np.zeros(hidden_size)
    
    def forward(self, x):
        r = 1.0 / (1.0 + np.exp(-(np.dot(x, self.W_ir) + np.dot(self.h, self.W_hr) + self.b_r)))
        z = 1.0 / (1.0 + np.exp(-(np.dot(x, self.W_iz) + np.dot(self.h, self.W_hz) + self.b_z)))
        h_tilde = np.tanh(np.dot(x, self.W_ih) + np.dot(r * self.h, self.W_hh) + self.b_h)
        self.h = (1 - z) * h_tilde + z * self.h
        
        return self.h
    
    def reset_state(self):
        self.h = np.zeros(self.hidden_size)
    
    def get_weights(self):
        weights = []
        weights.extend(self.W_ir.flatten())
        weights.extend(self.W_hr.flatten())
        weights.extend(self.b_r)
        weights.extend(self.W_iz.flatten())
        weights.extend(self.W_hz.flatten())
        weights.extend(self.b_z)
        weights.extend(self.W_ih.flatten())
        weights.extend(self.W_hh.flatten())
        weights.extend(self.b_h)

        return np.array(weights)
    
    def set_weights(self, weights):
        idx = 0
        size = self.input_size * self.hidden_size
        self.W_ir = weights[idx:idx+size].reshape(self.input_size, self.hidden_size)
        idx += size
        
        size = self.hidden_size * self.hidden_size
        self.W_hr = weights[idx:idx+size].reshape(self.hidden_size, self.hidden_size)
        idx += size
        
        self.b_r = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        size = self.input_size * self.hidden_size
        self.W_iz = weights[idx:idx+size].reshape(self.input_size, self.hidden_size)
        idx += size
        
        size = self.hidden_size * self.hidden_size
        self.W_hz = weights[idx:idx+size].reshape(self.hidden_size, self.hidden_size)
        idx += size
        
        self.b_z = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        size = self.input_size * self.hidden_size
        self.W_ih = weights[idx:idx+size].reshape(self.input_size, self.hidden_size)
        idx += size
        
        size = self.hidden_size * self.hidden_size
        self.W_hh = weights[idx:idx+size].reshape(self.hidden_size, self.hidden_size)
        idx += size
        
        self.b_h = weights[idx:idx+self.hidden_size]
    
    def get_param_count(self):
        return (3 * self.input_size * self.hidden_size +
                3 * self.hidden_size * self.hidden_size +
                3 * self.hidden_size)


class GRUController:
    
    def __init__(self, input_size=3, gru_size=8, dense_size=6, output_size=1):
        self.input_size = input_size
        self.gru_size = gru_size
        self.dense_size = dense_size
        self.output_size = output_size
        self.gru = GRUCell(input_size, gru_size)
        
        self.W_dense = np.random.randn(gru_size, dense_size) * 0.1
        self.b_dense = np.random.randn(dense_size) * 0.1
        self.W_output = np.random.randn(dense_size, output_size) * 0.1
        self.b_output = np.random.randn(output_size) * 0.1
        
        self.total_params = (self.gru.get_param_count() +
                           gru_size * dense_size +
                           dense_size +
                           dense_size * output_size +
                           output_size)
    
    def get_param_count(self):
        return self.total_params
    
    def set_params(self, params):
        assert len(params) == self.total_params, \
            f"Expected {self.total_params} params, got {len(params)}"
        
        idx = 0
        gru_params = self.gru.get_param_count()
        self.gru.set_weights(params[idx:idx+gru_params])
        idx += gru_params
        
        size = self.gru_size * self.dense_size
        self.W_dense = params[idx:idx+size].reshape(self.gru_size, self.dense_size)
        idx += size
        
        self.b_dense = params[idx:idx+self.dense_size]
        idx += self.dense_size
        
        size = self.dense_size * self.output_size
        self.W_output = params[idx:idx+size].reshape(self.dense_size, self.output_size)
        idx += size
        
        self.b_output = params[idx:idx+self.output_size]
    
    def get_params(self):
        params = []
        params.extend(self.gru.get_weights())
        params.extend(self.W_dense.flatten())
        params.extend(self.b_dense)
        params.extend(self.W_output.flatten())
        params.extend(self.b_output)

        return np.array(params)
    
    def forward(self, obs):
        obs = np.array(obs).reshape(-1)
        h_gru = self.gru.forward(obs)
        h_dense = np.maximum(0, np.dot(h_gru, self.W_dense) + self.b_dense)
        output = np.dot(h_dense, self.W_output) + self.b_output
        
        return output[0]
    
    def reset_hidden_state(self):
        self.gru.reset_state()


class GRUPolicy:
    
    def __init__(self, input_size=4, output_size=1, maxsteps=500, action_type='discrete'):
        self.input_size = input_size
        self.output_size = output_size
        self.maxsteps = maxsteps
        self.action_type = action_type
        self.state_reward = []
        self.noise = 0.1
        self.controller = GRUController(
            input_size=input_size,
            gru_size=8,
            dense_size=6,
            output_size=output_size
        )
        
        self.total_params = self.controller.get_param_count()
    
    def get_param_count(self):
        return self.total_params
    
    def set_params(self, params):
        assert len(params) == self.total_params, \
            f"Expected {self.total_params} params, got {len(params)}"
        self.controller.set_params(params)
    
    def get_params(self):
        return self.controller.get_params()
    
    def get_action(self, obs):

        output = self.controller.forward(obs)
        
        if self.action_type == 'discrete':
            return 1 if output > 0 else 0
        else:
            return np.clip(np.tanh(output) * 2.0, -2.0, 2.0)
    
    def rollout(self, env, ntrials=1, render=False, seed=None, custom_maxsteps=None, 
                custom_bounds=None, custom_state=None, custom_noise=None, weights=[0,1,0]):

        total_rew = 0.0
        total_steps = 0
        self.state_reward = []
        
        maxsteps = custom_maxsteps if custom_maxsteps is not None else self.maxsteps
        
        if seed is not None:
            np.random.seed(seed)
        
        if custom_noise is not None:
            self.noise = custom_noise
        
        for trial in range(ntrials):
            self.controller.reset_hidden_state()
            
            if hasattr(env, 'reset_custom') and custom_state is not None:
                obs, *_ = env.reset_custom(custom_state=custom_state)
            elif hasattr(env, 'reset_custom') and custom_bounds is not None:
                obs, *_ = env.reset_custom(custom_bounds=custom_bounds)
            else:
                if seed is not None:
                    obs, _ = env.reset(seed=seed)
                else:
                    obs, _ = env.reset()
            
            rew = 0.0
            t = 0
            
            while t < maxsteps:
                action = self.get_action(obs)
                
                if hasattr(env.step, '__code__') and len(env.step.__code__.co_varnames) > 2:
                    obs, r, terminated, truncated, info = env.step(action, self.noise)
                else:
                    obs, r, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated
                
                if len(obs) >= 2:
                    x, y = obs[0], obs[1]
                    self.state_reward.append((round(float(x), 2), round(float(y), 2), round(float(r), 2)))
                else:
                    self.state_reward.append((*[round(float(o), 2) for o in obs[:2]], round(float(r), 2)))
                
                rew += r
                t += 1
                
                if render:
                    env.render()
                    time.sleep(0.05)
                
                if done:
                    break
            
            total_rew += rew
            total_steps += t
        
        avg_rew = total_rew / ntrials

        return avg_rew, total_steps, self.state_reward


class PendulumPolicy(GRUPolicy):
    
    def __init__(self, maxsteps=200):
        super().__init__(
            input_size=3,
            output_size=1,
            maxsteps=maxsteps,
            action_type='continuous'
        )


class CartPolePolicy(GRUPolicy):
    
    def __init__(self, maxsteps=500):
        super().__init__(
            input_size=4,
            output_size=1,
            maxsteps=maxsteps,
            action_type='discrete'
        )