"""
Fixed GRU Policy with Proper Weight Initialization

Problem identified: GRU has 349 parameters vs. 13 in simple perceptron.
Large networks require careful initialization and tuning.

Solution: Add proper Xavier/He initialization and simplify architecture option.

Author: Brenda Silva Machado
"""

import numpy as np
import time


class SimplifiedGRUController:
    def __init__(self, input_size=3, gru_size=4, output_size=1):
        self.input_size = input_size
        self.gru_size = gru_size
        self.output_size = output_size
        
        scale_input = np.sqrt(2.0 / (input_size + gru_size))
        scale_hidden = np.sqrt(2.0 / (gru_size + gru_size))
        scale_output = np.sqrt(2.0 / (gru_size + output_size))
        

        self.W_ir = np.random.randn(input_size, gru_size) * scale_input
        self.W_hr = np.random.randn(gru_size, gru_size) * scale_hidden
        self.b_r = np.zeros(gru_size)  
        
        self.W_iz = np.random.randn(input_size, gru_size) * scale_input
        self.W_hz = np.random.randn(gru_size, gru_size) * scale_hidden
        self.b_z = np.ones(gru_size) * 0.5  # Start with gates half-open
        
        self.W_ih = np.random.randn(input_size, gru_size) * scale_input
        self.W_hh = np.random.randn(gru_size, gru_size) * scale_hidden
        self.b_h = np.zeros(gru_size)
        
        self.W_out = np.random.randn(gru_size, output_size) * scale_output
        self.b_out = np.zeros(output_size)
        
        self.h = np.zeros(gru_size)
        
        self.total_params = (
            3 * input_size * gru_size +
            3 * gru_size * gru_size +
            3 * gru_size +
            gru_size * output_size +
            output_size
        )
    
    def forward(self, x):
        x = np.array(x).reshape(-1)
        r = 1.0 / (1.0 + np.exp(-(np.dot(x, self.W_ir) + np.dot(self.h, self.W_hr) + self.b_r)))
        z = 1.0 / (1.0 + np.exp(-(np.dot(x, self.W_iz) + np.dot(self.h, self.W_hz) + self.b_z)))
        h_tilde = np.tanh(np.dot(x, self.W_ih) + np.dot(r * self.h, self.W_hh) + self.b_h)
        self.h = (1 - z) * h_tilde + z * self.h
        output = np.dot(self.h, self.W_out) + self.b_out

        return output[0]
    
    def reset_state(self):
        self.h = np.zeros(self.gru_size)
    
    def get_param_count(self):
        return self.total_params
    
    def get_params(self):
        params = []
        params.extend(self.W_ir.flatten())
        params.extend(self.W_hr.flatten())
        params.extend(self.b_r)
        params.extend(self.W_iz.flatten())
        params.extend(self.W_hz.flatten())
        params.extend(self.b_z)
        params.extend(self.W_ih.flatten())
        params.extend(self.W_hh.flatten())
        params.extend(self.b_h)
        params.extend(self.W_out.flatten())
        params.extend(self.b_out)
        return np.array(params)
    
    def set_params(self, params):
        idx = 0
        
        size = self.input_size * self.gru_size
        self.W_ir = params[idx:idx+size].reshape(self.input_size, self.gru_size)
        idx += size
        
        size = self.gru_size * self.gru_size
        self.W_hr = params[idx:idx+size].reshape(self.gru_size, self.gru_size)
        idx += size
        
        self.b_r = params[idx:idx+self.gru_size]
        idx += self.gru_size
        
        size = self.input_size * self.gru_size
        self.W_iz = params[idx:idx+size].reshape(self.input_size, self.gru_size)
        idx += size
        
        size = self.gru_size * self.gru_size
        self.W_hz = params[idx:idx+size].reshape(self.gru_size, self.gru_size)
        idx += size
        
        self.b_z = params[idx:idx+self.gru_size]
        idx += self.gru_size
        
        size = self.input_size * self.gru_size
        self.W_ih = params[idx:idx+size].reshape(self.input_size, self.gru_size)
        idx += size
        
        size = self.gru_size * self.gru_size
        self.W_hh = params[idx:idx+size].reshape(self.gru_size, self.gru_size)
        idx += size
        
        self.b_h = params[idx:idx+self.gru_size]
        idx += self.gru_size
        
        size = self.gru_size * self.output_size
        self.W_out = params[idx:idx+size].reshape(self.gru_size, self.output_size)
        idx += size
        
        self.b_out = params[idx:idx+self.output_size]


class SimplePerceptronController:
    def __init__(self, input_size=3, output_size=1):
        self.input_size = input_size
        self.output_size = output_size
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros(output_size)
        
        self.total_params = input_size * output_size + output_size
    
    def forward(self, x):
        x = np.array(x).reshape(-1)
        output = np.dot(x, self.W) + self.b
        return output[0]
    
    def reset_state(self):
        pass  
    
    def get_param_count(self):
        return self.total_params
    
    def get_params(self):
        return np.concatenate([self.W.flatten(), self.b])
    
    def set_params(self, params):
        size = self.input_size * self.output_size
        self.W = params[:size].reshape(self.input_size, self.output_size)
        self.b = params[size:]


class Policy:    
    def __init__(self, input_size=4, output_size=1, maxsteps=500, 
                 action_type='discrete', controller_type='simple'):
        self.input_size = input_size
        self.output_size = output_size
        self.maxsteps = maxsteps
        self.action_type = action_type
        self.state_reward = []
        self.noise = 0.1
        self.controller_type = controller_type
        
        # Select controller
        if controller_type == 'simple':
            self.controller = SimplePerceptronController(input_size, output_size)
        elif controller_type == 'gru_small':
            self.controller = SimplifiedGRUController(input_size, gru_size=4, 
                                                     output_size=output_size)
        else:  # gru_full
            from advanced_policy import GRUController
            self.controller = GRUController(input_size, gru_size=8, dense_size=6,
                                           output_size=output_size)
        
        self.total_params = self.controller.get_param_count()
        
        print(f"Policy initialized: {controller_type}, {self.total_params} parameters")
    
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
            self.controller.reset_state()
            
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


class PendulumPolicy(Policy):
    def __init__(self, maxsteps=200, controller_type='simple'):
        super().__init__(
            input_size=3,
            output_size=1,
            maxsteps=maxsteps,
            action_type='continuous',
            controller_type=controller_type
        )


class CartPolePolicy(Policy):
    def __init__(self, maxsteps=500, controller_type='simple'):
        super().__init__(
            input_size=4,
            output_size=1,
            maxsteps=maxsteps,
            action_type='discrete',
            controller_type=controller_type
        )