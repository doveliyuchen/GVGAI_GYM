import sys
sys.path.insert(0, '.')
import gym_gvgai as gvgai
import random

print('Creating aliens-lvl0-v0 environment...')
env = gvgai.make('gvgai-aliens-lvl0-v0')
print('Resetting environment...')
obs = env.reset()
print(f'Initial obs: {type(obs)}')

print('Running 10 random steps...')
for i in range(10):
    action = random.randint(0, env.action_space.n - 1)
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    winner = info.get('winner', 'N/A')
    ascii_state = info.get('ascii', '')
    print(f'Step {i+1}: action={action}, reward={reward:.2f}, done={done}, winner={winner}, ascii_len={len(ascii_state)}')
    if done:
        print('Game ended early.')
        break

env.close()
print('Test completed successfully!')
