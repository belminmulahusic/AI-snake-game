from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from snake_env import SnakeEnv

env = SnakeEnv()
check_env(env)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
model.save("dqn_snake_model")
env.close()