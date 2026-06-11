import numpy as np
import matplotlib.pyplot as plt
from game.snake_game import SnakeGame
from ai.agent import DQNAgent
import os

def train_agent(episodes=100, render=False, save_model=True):
    """Agent trainieren"""
    
    env = SnakeGame(grid_size=20, render=render)
    agent = DQNAgent(state_size=13, action_size=4, learning_rate=0.001)
    
    scores = []
    losses = []
    
    print("🤖 Starting DQN Training...")
    print(f"Episodes: {episodes}")
    print(f"Render: {render}\n")
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_loss = 0
        steps = 0
        
        while True:
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay_optimized(batch_size=64)
            episode_loss += loss
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        scores.append(env.score)
        losses.append(episode_loss)
        
        if episode % 20 == 0 or episode == 1:
            avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            print(f"Episode {episode}/{episodes} | Score: {env.score} | Avg: {avg_score:.2f} | Loss: {episode_loss:.4f} | Eps: {agent.epsilon:.3f}")
    
    if save_model:
        os.makedirs("models", exist_ok=True)
        model_path = "models/snake_dqn.pt"
        agent.save(model_path)
        print(f"\n✅ Modell gespeichert: {model_path}")
    
    plot_training_results(scores, losses)
    
    return agent, scores, losses

def plot_training_results(scores, losses):
    """Trainingsergebnisse plotten"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(scores, alpha=0.6, label='Score', color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('🐍 Snake Scores während Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(losses, alpha=0.6, color='red', label='Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('🧠 Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("📊 Plot gespeichert: training_results.png")
    plt.close()

if __name__ == "__main__":
    agent, scores, losses = train_agent(
        episodes=100,
        render=False,
        save_model=True
    )
    
    print("\n🎉 Training abgeschlossen!")
    print(f"Max Score: {max(scores)}")
    print(f"Avg Score: {np.mean(scores):.2f}")
