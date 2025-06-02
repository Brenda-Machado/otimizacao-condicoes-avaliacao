
import numpy as np
import time

class Policy:
    def __init__(self, param1=0.25, param2=0.25, maxsteps=500):
        self.param1 = param1
        self.param2 = param2
        self.maxsteps = maxsteps
        self.state_reward = []

    def get_action(self, obs):
        val = self.param1 * obs[2] + self.param2 * obs[3]
        return 1 if val > 0 else 0

    def rollout(self, env, ntrials=1, render=False, seed=None):
        total_rew = 0.0
        total_steps = 0
        best_rew = -999.9
        worst_rew = 0.0
        components_wheights = [0, 1, 0]

        if seed is not None:
            np.random.seed(seed)
            env.reset(seed=seed)

        for trial in range(ntrials):
            obs, _ = env.reset()
            rew = 0.0
            t = 0

            while t < self.maxsteps:
                action = self.get_action(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                x, y = obs[0], obs[1]
                self.state_reward.append((float(x), float(y), float(r)))
                rew += r
                t += 1

                if render:
                    env.render()
                    time.sleep(0.05)

                if done:
                    break

            print(f"Trial {trial} Fit {rew:.2f} Steps {t}")
            total_rew += rew
            total_steps += t

        avg_rew = total_rew / ntrials
        return avg_rew, total_steps, self.state_reward

# Exemplo de uso:
# env = gym.make("CartPole-v1")
# policy = SimplePolicy(param1=1.0, param2=0.5)
# avg_reward, steps, traj = policy.rollout(env, ntrials=10)
