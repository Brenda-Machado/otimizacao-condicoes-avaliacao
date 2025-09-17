"""

Author: Brenda Silva Machado.

Policy

"""

import numpy as np
import time

class Policy:
    def __init__(self, input_size=4, output_size=1, maxsteps=500, action_type='discrete'):
        self.input_size = input_size
        self.output_size = output_size
        self.maxsteps = maxsteps
        self.action_type = action_type
        self.state_reward = []
        self.noise = 0.1
        
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.1
        self.bias = np.random.randn(self.output_size) * 0.1
        
        self.total_params = self.input_size * self.output_size + self.output_size
    
    def get_param_count(self):
        return self.total_params
    
    def set_params(self, params):
        assert len(params) == self.total_params, f"Expected {self.total_params} params, got {len(params)}"
        
        weights_size = self.input_size * self.output_size
        self.weights = params[:weights_size].reshape(self.input_size, self.output_size)
        self.bias = params[weights_size:]
    
    def get_params(self):
        return np.concatenate([self.weights.flatten(), self.bias])
    
    def perceptron(self, obs):
        obs = np.array(obs).reshape(-1)
        output = np.dot(obs, self.weights) + self.bias
        
        if self.action_type == 'discrete':
            return 1 if output[0] > 0 else 0
        else:
            return np.clip(np.tanh(output[0]) * 2.0, -2.0, 2.0)
    
    def get_action(self, obs):
        return self.perceptron(obs)
    
    def rollout(self, env, ntrials=1, render=False, seed=None, custom_maxsteps=None, 
                custom_bounds=None, custom_state=None, custom_noise=None, weights=[0,1,0]):
        total_rew = 0.0
        total_steps = 0
        self.state_reward = []
        
        if custom_maxsteps is not None:
            maxsteps = custom_maxsteps
        else:
            maxsteps = self.maxsteps
            
        if seed is not None:
            np.random.seed(seed)
            
        if custom_noise is not None:
            self.noise = custom_noise
        
        for trial in range(ntrials):
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
    def __init__(self, maxsteps=200):
        super().__init__(
            input_size=3,
            output_size=1,
            maxsteps=maxsteps,
            action_type='continuous'
        )


class CartPolePolicy(Policy):
    def __init__(self, maxsteps=500):
        super().__init__(
            input_size=4,
            output_size=1,
            maxsteps=maxsteps,
            action_type='discrete'
        )

