##################################################################
#                                                                #
#   Die evaluate.py übernimmt die Evaluation des DQN-Agenten.    #
#   Die Ergebnisse können in der Konsole ausgelesen werden       #
#                                                                #
##################################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from snake_env import SnakeEnv
import torch
import argparse

ACTION_NAMES = {
    0: "Gerade",
    1: "Links",
    2: "Rechts"
}


# Gibt die Q-Werte für eine gegebene Beobachtung zurück
# Parameter:
#   model (DQN): Das geladene DQN-Modell
#   obs (np.array): Beobachtungsraum
# Return:
#   np.array: Q-Werte für alle Aktionen
def get_q_values(model, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    return q_values.cpu().numpy()[0]


# Analysiert Q-Werte: Berechnet Maximalwert und Standardabweichung
# Parameter:
#   q_values (np.array): Array der Q-Werte
# Return:
#   Tuple(float, float): max Q-Wert, Standardabweichung der Q-Werte
def analyze_q_values(q_values):
    max_q = np.max(q_values)
    std_q = np.std(q_values)
    return max_q, std_q


# Füllt kürzere Listen mit NaN auf (Damit man die Daten leichter vergleichen kann)
# Parameter:
#   lists (List[List[float]]): Liste von Q-Wert-Listen verschiedener Episoden
#   max_len (int): Länge der längsten Liste
# Return:
#   np.array: 2D-Array mit aufgefüllten Werten
def pad_and_stack(lists, max_len):
    arr = np.full((len(lists), max_len), np.nan)
    for i, lst in enumerate(lists):
        arr[i, :len(lst)] = lst
    return arr


# Führt die Evaluation des DQN-Modells durch
# Parameter:
#   model_path (str): Pfad zum gespeicherten Modell
#   num_episodes (int): Anzahl der Episoden
#   render (bool): Ob das Spiel gerendert werden soll
#   show_q_values (bool): Ob Q-Werte bei jedem Schritt ausgegeben werden sollen
#   show_dqn (bool): Ob die DQN-Netzwerkarchitektur ausgegeben werdn soll
def evaluate_model(model_path, num_episodes=50, render=False, show_q_values=False, show_dqn=False):
    model = DQN.load(model_path)
    scores = []
    steps_per_game = []
    max_qs_all = []
    std_qs_all = []
    valid_episodes = 0
    loop_count = 0

    for episode in range(num_episodes):
        env = SnakeEnv(render_mode="human" if render else None)
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        max_qs = []
        std_qs = []

        print(f"\n--- Episode {episode + 1} ---")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            q_values = get_q_values(model, obs)
            max_q, std_q = analyze_q_values(q_values)
            max_qs.append(max_q)
            std_qs.append(std_q)
            
            obs, reward, done, truncated, info = env.step(action_int)
            steps += 1
            
            if show_q_values:
                print("--------------------------------------------------------------------")
                print(f"Q-Werte: Geradeaus = {q_values[0]:.2f}, Links = {q_values[1]:.2f}, Rechts = {q_values[2]:.2f}")
                print(f"Step {steps}: Action = {ACTION_NAMES.get(action_int, action_int)}, Reward = {reward:.3f}, Score = {env.score}")
                print("--------------------------------------------------------------------")

            if render:
                env.render()

        # Abbruch durch Loop erkennen (Siehe Ausarbeitung -> Evaluation)
        if truncated:
            print(f"Episode {episode + 1} abgebrochen aufgrund einer Dauerschleife")
            loop_count += 1
            env.close()
            continue
            
        valid_episodes += 1
        scores.append(env.score)
        steps_per_game.append(steps)
        max_qs_all.append(max_qs)
        std_qs_all.append(std_qs)
        
        print(f"Final Score = {env.score}, Schritte = {steps}")
        env.close()

    if valid_episodes == 0:
        print("\nKeine validen Episoden zum Auswerten.")
        return

    scores_np = np.array(scores)
    steps_np = np.array(steps_per_game)

    # Statistische Auswertung der Ergebnisse
    print("\n--- Auswertung ---")
    print(f"Durchschnittlicher Score: {np.mean(scores_np):.2f}")
    print(f"Median Score: {np.median(scores_np):.2f}")
    print(f"Standardabweichung: {np.std(scores_np):.2f}")
    print(f"Höchster Score: {np.max(scores_np)}")
    print(f"Niedrigster Score: {np.min(scores_np)}")
    print(f"Durchschnittliche Schritte: {np.mean(steps_np):.2f}")
    print(f"Anzahl valider Episoden: {valid_episodes}")
    print(f"Anzahl von Schleifen: {loop_count}")
    
    # Ergebnisse in Textdatei speichern
    with open("evaluation.txt", "w", encoding="utf-8") as f:
        f.write("--- Auswertung ---\n")
        f.write(f"Durchschnittlicher Score: {np.mean(scores_np):.2f}\n")
        f.write(f"Median Score: {np.median(scores_np):.2f}\n")
        f.write(f"Standardabweichung: {np.std(scores_np):.2f}\n")
        f.write(f"Höchster Score: {np.max(scores_np)}\n")
        f.write(f"Niedrigster Score: {np.min(scores_np)}\n")
        f.write(f"Durchschnittliche Schritte: {np.mean(steps_np):.2f}\n")
        f.write(f"Anzahl valider Episoden: {valid_episodes}\n")
        f.write(f"Anzahl von Schleifen: {loop_count}\n")
    
    # Architektur des DQN-Netzes ausgeben
    if show_dqn:
        print(model.q_net)
        

    # Plots erstellen und speichern
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(scores_np, bins=15, color='pink', edgecolor='black')
    plt.title('Score-Verteilung')
    plt.xlabel('Score')
    plt.ylabel('Anzahl der Episoden')

    max_len = max(len(x) for x in max_qs_all) if max_qs_all else 0
    if max_len > 0:
        max_qs_arr = pad_and_stack(max_qs_all, max_len)
        std_qs_arr = pad_and_stack(std_qs_all, max_len)

        avg_max_qs = np.nanmean(max_qs_arr, axis=0)
        avg_std_qs = np.nanmean(std_qs_arr, axis=0)

        plt.subplot(1, 2, 2)
        plt.plot(avg_max_qs, label='Durchschnittlicher Max Q-Wert')
        plt.plot(avg_std_qs, label='Durchschnittliche Std. Abw.')
        plt.title('Q-Wert Metriken im Verlauf')
        plt.xlabel('Schritt')
        plt.ylabel('Q-Wert')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("evaluation_combined.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Snake model")
    parser.add_argument("model_path", type=str, help="Pfad zum gespeicherten Modell")
    parser.add_argument("--episodes", type=int, default=50, help="Anzahl der Test-Episoden")
    parser.add_argument("-r", "--render", action="store_true", help="Spiel während der Evaluation anzeigen")
    parser.add_argument("-q", "--q_values", action="store_true", help="Q-Werte für jede Situation anzeigen")
    parser.add_argument("-dqn", "--show_dqn", action="store_true", help="DQN-Netzwerk anzeigen lassen")
    args = parser.parse_args()

    evaluate_model(args.model_path, num_episodes=args.episodes, render=args.render, show_q_values=args.q_values, show_dqn=args.show_dqn)