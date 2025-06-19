from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from snake_env import SnakeEnv

env = SnakeEnv()
check_env(env)
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=32,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
)

model.learn(total_timesteps=500000)
model.save("dqn_snake_model")
env.close()