from stable_baselines3 import DQN
from snake_env import SnakeEnv

def test_model(model_path):
    env = SnakeEnv(render_mode="human")
    model = DQN.load(model_path)
    
    obs, info = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    print(f"Spiel beendet. End-Score: {env.score}")
    env.close()

if __name__ == "__main__":
    test_model("dqn_snake_model")