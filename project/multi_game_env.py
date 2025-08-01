import gym
import gym_gvgai as gvgai
import random
import numpy as np

class MultiGameEnv(gym.Env):
    def __init__(self, game_list, levels):
        self.game_list = game_list
        self.levels = levels
        self.current_env = None
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(250, 690, 4), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(5)
        self._make_env()

    def _pad_obs(self, obs):
        padded_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        padded_obs[:obs.shape[0], :obs.shape[1], :] = obs
        return padded_obs

    def _make_env(self):
        if self.current_env:
            self.current_env.close()
        
        while True:
            game_name = random.choice(self.game_list)
            level = random.choice(self.levels)
            env_name = f'gvgai-{game_name}-lvl{level}-v0'
            try:
                self.current_env = gvgai.make(env_name)
                self.game_name = game_name
                break
            except Exception as e:
                print(f"Error creating environment {env_name}: {e}. Trying another.")

    def step(self, action):
        obs, reward, done, info = self.current_env.step(action)
        return self._pad_obs(obs), reward, done, info

    def reset(self):
        self._make_env()
        obs = self.current_env.reset()
        return self._pad_obs(obs)

    def render(self, mode='human'):
        return self.current_env.render(mode=mode)

    def close(self):
        if self.current_env:
            self.current_env.close()

    @property
    def unwrapped(self):
        return self.current_env.unwrapped
