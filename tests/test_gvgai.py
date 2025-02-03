def test_gvgaiENV():
    try: 
        import gym
        import gym_gvgai
        env = gym.make('gvgai-aliens-lvl0-v0')
        env.reset()
        action_id = env.action_space.sample()
        env.step(action_id)
        return True
    except Exception as er:
        assert False, er


if __name__ == "__main__":
    print(test_gvgaiENV())
