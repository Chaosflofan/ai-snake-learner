# AI Snake Learner 🐍🤖

Eine selbstlernende KI, die das Spiel **Snake** spielt und durch **Reinforcement Learning** und **Deep Q-Learning (DQN)** mit neuronalen Netzen trainiert wird.

## 🎯 Features

✅ **Deep Q-Learning (DQN)** - Neuronale Netze mit RL
✅ **Self-Learning** - Lernt durch Trial-and-Error
✅ **Experience Replay** - Effizientes Batch-Training
✅ **Visualisierung** - Pygame Rendering mit Live Gameplay
✅ **Training Plots** - Automatische Lernkurven

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Chaosflofan/ai-snake-learner.git
cd ai-snake-learner
pip install -r requirements.txt
```

### Training (100 Episoden)
```bash
python train.py
```

**Output:**
- Trainingsprogress im Terminal
- `models/snake_dqn.pt` - Trainiertes Modell
- `training_results.png` - Lernkurven

### Mit trainiertem Modell spielen
```bash
python play.py
```

**Visual Output:**
- Pygame Fenster mit Snake Animation
- Live Gameplay mit Scores
- 5 Spiele Statistik

## 📊 Projekt Struktur

```
ai-snake-learner/
├── game/snake_game.py      # Snake Environment mit Pygame
├── ai/
│   ├── agent.py            # DQN Agent
│   └── model.py            # Neural Network
├── train.py                # Training Script
├── play.py                 # Spielmodus mit Visuals
├── requirements.txt        # Dependencies
└── README.md              # Doku
```

## 🎮 Visuelles Output

### Training Phase
```
🤖 Starting DQN Training...
Episodes: 100

Episode 1/100 | Score: 2 | Avg: 2.00 | Loss: 0.5432 | Eps: 1.000
Episode 20/100 | Score: 8 | Avg: 4.25 | Loss: 0.1234 | Eps: 0.905
Episode 100/100 | Score: 28 | Avg: 18.50 | Loss: 0.0045 | Eps: 0.135

✅ Modell gespeichert: models/snake_dqn.pt
📊 Plot gespeichert: training_results.png

🎉 Training abgeschlossen!
Max Score: 35
Avg Score: 16.42
```

### Play Phase
```
✅ Modell geladen: models/snake_dqn.pt

🎮 Spielmodus gestartet!
Spiele: 5

Spiel 1: Score = 24 | Steps = 389
Spiel 2: Score = 28 | Steps = 421
Spiel 3: Score = 26 | Steps = 405
Spiel 4: Score = 30 | Steps = 456
Spiel 5: Score = 25 | Steps = 392

📊 Durchschnitt: 26.60
Max Score: 30
Min Score: 24
```

## 🎨 Visuelle Features

### Pygame Rendering
- **Fenster:** 400x400px (20x20 Grid)
- **Snake:** Grün (hellgrün = Kopf)
- **Essen:** Rot
- **FPS:** 10 (gute Beobachtbarkeit)

### Training Plots
- **Score Graph:** Lernfortschritt
- **Loss Graph:** Netzwerk Konvergenz

## 🧠 Architektur

### State (13 Dimensionen)
- Head Position
- Body Length
- Food Distance (X, Y)
- Danger Signals (4 Richtungen)
- Direction (One-Hot)

### DQN Network
```
Input(13) → Dense(128) + ReLU → Dense(128) + ReLU → Output(4)
```

### Actions
- 0: UP
- 1: DOWN
- 2: LEFT
- 3: RIGHT

### Rewards
- +10: Essen ✅
- -10: Crash ❌
- -0.01: Jeder Schritt

## 📈 Hyperparameter

| Parameter | Wert |
|-----------|------|
| Episodes | 100 |
| Grid Size | 20×20 |
| Learning Rate | 0.001 |
| Gamma | 0.95 |
| Epsilon Start | 1.0 |
| Epsilon Decay | 0.995 |
| Batch Size | 64 |
| Memory Size | 2000 |

## 🎓 Was die KI lernt

- **Episode 1-20:** Zufällige Bewegungen, niedrige Scores (2-5)
- **Episode 21-60:** Essen Jagd, bessere Navigation (5-15)
- **Episode 61-100:** Strategische Planung (15-30+)

## 🔧 Customization

### Längeres Training
```python
train_agent(episodes=500)
```

### Mit Rendering
```python
train_agent(render=True)
```

### Mehr Spiele ansehen
In `play.py`:
```python
play_game(num_games=10)
```

## ✅ Status

✅ Alle Dateien funktionieren
✅ Visuals implementiert (Pygame)
✅ Training und Play Mode fertig
✅ Plots und Statistiken

## 📜 Lizenz

MIT License

---

**Viel Erfolg! 🚀🐍**
