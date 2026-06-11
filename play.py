from game.snake_game import SnakeGame
from ai.agent import DQNAgent
import os

def play_game(num_games=5, render=True, model_path="models/snake_dqn.pt"):
    """Trainiertes Modell spielen"""
    
    env = SnakeGame(grid_size=20, render=render)
    agent = DQNAgent(state_size=13, action_size=4)
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ Modell geladen: {model_path}\n")
    else:
        print(f"❌ Modell nicht gefunden: {model_path}")
        print("Bitte trainiere zuerst mit: python train.py")
        return
    
    print("🎮 Spielmodus gestartet!")
    print(f"Spiele: {num_games}\n")
    
    scores = []
    
    for game in range(1, num_games + 1):
        state = env.reset()
        
        while True:
            action = agent.act(state, training=False)
            next_state, reward, done = env.step(action)
            state = next_state
            
            if done:
                break
        
        scores.append(env.score)
        print(f"Spiel {game}: Score = {env.score} | Steps = {env.steps}")
    
    print(f"\n📊 Durchschnitt: {sum(scores)/len(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")

if __name__ == "__main__":
    play_game(
        num_games=5,
        render=True,
        model_path="models/snake_dqn.pt"
    )
