##################################################################
#                                                                #
#   Die train_env.py wird benutzt um den Agenten mit der         #
#   Stable-baselines3 Bibliothek zu trainieren.                  #
#                                                                #
##################################################################

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from snake_env import SnakeEnv

# Hauptfunktion zum Trainieren eines DQN-Agenten auf der Snake-Umgebung.
# Erstellt die Umgebung, überprüft ihre Kompatibilität, initialisiert das Modell
# mit spezifischen Hyperparametern, führt das Training durch und speichert das Ergebnis.
if __name__ == "__main__":
    
    env = SnakeEnv()
    # Überprüft, ob die Umgebung den Anforderungen von Gymnasium entspricht
    check_env(env)
    
    # Erstellt ein DQN-Modell mit den angegebenen Hyperparametern
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=1_000_000,
        learning_starts=200_000,
        batch_size=64,
        target_update_interval=20_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="./dqn_snake_tensorboard_182_10m/"
    )


    model.learn(total_timesteps=10_000_000, log_interval=4) 
    
    print("Training für 10 Millionen Schritte abgeschlossen. Speichere das Modell...")
    model.save("dqn_snake_model_182_10m")
    env.close()