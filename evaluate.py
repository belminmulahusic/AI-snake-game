import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from snake_env import SnakeEnv

import torch
ACTION_NAMES = {
    0: "Gerade",
    1: "Links",
    2: "Rechts"
}

def get_q_values(model, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    return q_values.cpu().numpy()[0]


def evaluate_model(model_path, num_episodes=50, render=False, show_q_values=False):
    model = DQN.load(model_path)
    scores = []
    steps_per_game = []
    valid_episodes = 0
    loop_count = 0
    for episode in range(num_episodes):
        env = SnakeEnv(render_mode="human")
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0

        print(f"\n--- Episode {episode + 1} ---")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            q_values = get_q_values(model, obs)

            obs, reward, done, truncated, info = env.step(action_int)
            steps += 1
            if show_q_values:
                print("---------------------------------------------------------------------------")
                print(f"Q-Werte: Geradeaus = {q_values[0]:.2f}, Links = {q_values[1]:.2f}, Rechts = {q_values[2]:.2f}")
                print(" ")
                print(f"Step {steps}: Action = {ACTION_NAMES.get(action_int, action_int)}, Reward = {reward:.3f}, Score = {env.score}")
                print("---------------------------------------------------------------------------")

            if render:
                env.render(game_mode="AI")

        if truncated:
            print(f"Episode {episode + 1} abgebrochen aufgrund einer Dauerschleife")
            loop_count += 1
            continue
        valid_episodes += 1


        scores.append(env.score)
        steps_per_game.append(steps)
        print(f"Episode {episode + 1} beendet: Final Score = {env.score}, Schritte = {steps}")
        env.close()

    scores_np = np.array(scores)

    print("\n--- Auswertung ---")
    print(f"Durchschnittlicher Score: {np.mean(scores_np):.2f}")
    print(f"Median Score: {np.median(scores_np):.2f}")
    print(f"Standardabweichung: {np.std(scores_np):.2f}")
    print(f"Höchster Score: {np.max(scores_np)}")
    print(f"Niedrigster Score: {np.min(scores_np)}")
    print(f"Durchschnittliche Schritte: {np.mean(steps_per_game):.2f}")
    print(f"Anzahl valider Episoden: {valid_episodes:.2f}")
    print(f"Anzahl von Schleifen: {loop_count:.2f}")


    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.hist(scores_np, bins=15, color='skyblue', edgecolor='black')
    plt.title('Score-Verteilung')
    plt.xlabel('Score')
    plt.ylabel('Anzahl der Episoden')

    plt.subplot(1,2,2)
    plt.plot(range(1, valid_episodes+1), scores_np, marker='o')
    plt.title('Score-Verlauf pro Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.savefig("auswertung_score_histogramm.png")

    return scores, steps_per_game

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DQN Snake model")
    parser.add_argument("model_path", type=str, help="Pfad zum gespeicherten Modell")
    parser.add_argument("--episodes", type=int, default=50, help="Anzahl der Test-Episoden")
    parser.add_argument("-r", action="store_true", help="Spiel während der Evaluation anzeigen")
    parser.add_argument("-q", action="store_true", help="Q-Werte für jede Situation anzeigen")
    args = parser.parse_args()

    evaluate_model(args.model_path, num_episodes=args.episodes, render=args.r, show_q_values=args.q)