import gym_gvgai as gvgai
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from gym.wrappers import ResizeObservation

# Create the same environment setup as during training
log_dir = "dqn_gvgai_logs/"
env_name = 'gvgai-golddigger-lvl0-v0'
env = DummyVecEnv([lambda: ResizeObservation(gvgai.make(env_name), (84, 84))])
env = VecMonitor(env, log_dir)
env = VecTransposeImage(env)

# Method 1: Load the trained model directly
model_path = "dqn_gvgai_logs/best_model_gold_digger_lvl0.zip"
model = DQN.load(model_path, env=env)

print("Model loaded successfully!")
print(f"Model policy: {model.policy}")
print(f"Model buffer size: {model.buffer_size}")
print(f"Model exploration final eps: {model.exploration_final_eps}")

# Method 2: If you want to create a new model with same params and load weights
# model = DQN("CnnPolicy", env, verbose=1, buffer_size=1000000, learning_starts=1000, exploration_final_eps=0.1, train_freq=4)
# model = DQN.load(model_path, env=env)

# Now you can use the model for inference
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        print(f"Episode finished with reward: {rewards}")
        obs = env.reset()
