# AI Snake Learner 🐍🤖

Eine selbstlernende KI, die das Spiel **Snake** spielt und durch **Reinforcement Learning** und **Deep Q-Learning (DQN)** mit neuronalen Netzen trainiert wird.

## 🎯 Features

- **Deep Q-Learning (DQN)**: Kombination von RL + neuronalen Netzen
- **Self-Learning**: KI lernt durch Trial-and-Error
- **Experience Replay**: Effizientes Batch-Training
- **Epsilon-Greedy Strategy**: Balance zwischen Exploration und Exploitation
- **Visualisierung**: Training Plots und Echtzeit Gameplay
- **Modell Speicherung**: Trainierte Modelle speichern und laden

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Chaosflofan/ai-snake-learner.git
cd ai-snake-learner
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

Das trainiert die KI für 500 Episoden. Nach dem Training:
- Modell wird in `models/snake_dqn.pt` gespeichert
- `training_results.png` zeigt die Lernkurven

### Mit trainiertem Modell spielen

```bash
python play.py
```

Die KI spielt 5 Spiele mit dem trainierten Modell.

## 📁 Projekt Struktur

```
ai-snake-learner/
├── game/
│   └── snake_game.py           # Snake Game Environment
├── ai/
│   ├── agent.py                # DQN Agent
│   └── model.py                # Neural Network
├── train.py                    # Training Script
├── play.py                     # Play Script
├── requirements.txt            # Dependencies
└── README.md                   # Dokumentation
```

## 🧠 Architektur

### State (13 Dimensionen)
- Head Position (normalized)
- Body Length
- Food Distance (X, Y)
- Danger Signals (4 Richtungen)
- Direction (One-Hot)

### DQN Network
```
Input (13) → Dense(128) + ReLU → Dense(128) + ReLU → Output(4)
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

## 📊 Hyperparameter

| Parameter | Wert |
|-----------|------|
| Episodes | 500 |
| Grid Size | 20×20 |
| Learning Rate | 0.001 |
| Gamma | 0.95 |
| Epsilon Start | 1.0 |
| Epsilon Decay | 0.995 |
| Batch Size | 64 |
| Memory Size | 2000 |

## 📈 Was die KI lernt

- **Nach 50 Episodes**: Grundlegende Bewegung
- **Nach 200 Episodes**: Aktive Essen-Jagd  
- **Nach 500 Episodes**: Strategische Planung, hohe Scores (25-35+)

## 🔧 Customization

### Längeres Training
```python
train_agent(episodes=1000)
```

### Mit Rendering
```python
train_agent(render=True)
```

### Hyperparameter
In `ai/agent.py`:
```python
self.gamma = 0.99        # Längere Sicht
self.epsilon_decay = 0.99  # Mehr Exploration
```

## 🐛 Troubleshooting

- **CUDA nicht vorhanden**: Nutzt automatisch CPU
- **Memory Error**: Reduziere batch_size
- **Pygame Error**: `pip install pygame --upgrade`
- **Modell nicht gefunden**: Trainiere erst mit `python train.py`

## 📝 Beta Status

Geplante Features:
- [ ] Double DQN
- [ ] Dueling Architecture
- [ ] Priority Experience Replay
- [ ] Tensorboard Integration
- [ ] Video Recording

## 📜 Lizenz

MIT License

---

**Viel Erfolg! 🚀🐍**
