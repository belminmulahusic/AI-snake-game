# Snake Reinforcement Learning

Dieses Projekt implementiert das klassische Snake-Spiel in Python und integriert es in eine eigene Reinforcement-Learning-Umgebung basierend auf [Gymnasium](https://github.com/Farama-Foundation/Gymnasium). Mittels [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) kann ein Agent mit Deep Q-Learning (DQN) trainiert werden. Die Dokumentation des Projekts ist nicht direkt im Wiki, sondern in der X.pdf zu finden.

## Voraussetzungen & Installation

Zur Nutzung des Projekts sind einige Programme erforderlich. Stelle sicher, dass **Python 3.12.3** installiert ist.

Empfohlen wird die Nutzung einer virtuellen Umgebung:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Folgende Paketversionen werden verwendet:

```
gymnasium==1.1.1
numpy==2.1.3
pygame==2.5.2
stable_baselines3==2.6.0
```

### Spiel starten

```bash
python game.py
```

- Starte das  Spielmenü
- Im Menü:
  - **Play**: Manuell spielbares Snake-Spiel
  - **Run AI Model**: DQN-Modell spielt automatisch eine Runde

### Training

```bash
python train_env.py
```

- Trainiert ein neues DQN-Modell
- Speichert:
  - Modell unter `dqn_snake_model.zip`
  - TensorBoard-Logs unter `dqn_snake_tensorboard/`



### Modell evaluieren

```bash
python evaluate.py dqn_snake_model.zip --episodes 50 --render -q --dqn
```

- Bewertet das Modell über eine definierte Anzahl an Episoden
- Optionen:
  - `--episodes`: Anzahl an Spielwiederholungen
  - `--render`: Visuelle Darstellung aktivieren
  - `-q`: Q-Werte anzeigen
  - `-dqn`: DQN-Modellausgabe aktivieren

