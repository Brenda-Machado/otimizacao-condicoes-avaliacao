"""
Brenda Machado

policy.py
"""

import numpy as np
import time

class Policy:
    def __init__(self, input_size=1, maxsteps=500):
        self.input_size = input_size
        self.weights = np.random.randn(self.input_size)
        self.bias = np.random.randn()
        self.maxsteps = maxsteps
        self.state_reward = []

    def perceptron(self, obs):
        output = np.dot(self.weights, obs) + self.bias
        return 1 if output > 0 else 0

    def get_action(self, obs):
        return self.perceptron(obs)

    def rollout(self, env, ntrials=1, render=False, seed=None, custom_maxsteps=None, custom_bounds= None, custom_state=None, wheights=[0,1,0]):
        total_rew = 0.0
        total_steps = 0
        self.state_reward = []

        if custom_maxsteps is not None:
            self.maxsteps = custom_maxsteps

        if seed is not None:
            np.random.seed(seed)
            env.reset(seed=seed)

        for trial in range(ntrials):
            if custom_state is not None:
                obs, _ = env.reset_custom(custom_state=custom_state)
            if custom_bounds is not None:
                obs, _ = env.reset_custom(custom_bounds=custom_bounds)
            else:
                obs, _ = env.reset()
            
            rew = 0.0
            t = 0

            while t < self.maxsteps:
                action = self.get_action(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                x, y = obs[0], obs[1]
                # self.state_reward.append((float(x), float(y), float(r)))
                self.state_reward.append((round(float(x), 2), round(float(y), 2), round(float(r), 2)))
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
