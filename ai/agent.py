import numpy as np
import torch
from collections import deque
import random
from ai.model import DQNNetwork

class DQNAgent:
    """
    DQN Agent mit Experience Replay
    
    Kombiniert Reinforcement Learning mit Deep Q-Learning
    """
    
    def __init__(self, state_size=13, action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameter
        self.gamma = 0.95  # Discount Faktor
        self.epsilon = 1.0  # Exploration Rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Memory für Experience Replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        
        # Neural Network
        self.model = DQNNetwork(state_size, action_size, learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Experience speichern für späteres Training
        
        Args:
            state: Aktueller State
            action: Gewählte Aktion
            reward: Erhaltene Belohnung
            next_state: Nächster State
            done: Spiel beendet?
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Aktion wählen (Epsilon-Greedy)
        
        Args:
            state: Aktueller State
            training: Trainingsmodus (mit Exploration)?
        
        Returns:
            Aktion (0-3)
        """
        # Exploration vs Exploitation
        if training and np.random.random() < self.epsilon:
            # Zufällige Aktion
            return random.randrange(self.action_size)
        
        # Beste Aktion wählen
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return np.argmax(q_values.cpu().numpy()[0])
    
    def replay(self, batch_size=None):
        """
        Experience Replay Training
        
        Args:
            batch_size: Größe des Trainings Batches
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return 0
        
        # Random Batch aus Memory
        batch = random.sample(self.memory, batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Tensoren vorbereiten
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Current Q-Werte
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-Werte berechnen
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Training Step
        loss = self.model.train_step(
            torch.nn.functional.one_hot(actions, num_classes=self.action_size).float() * q_values.unsqueeze(1),
            torch.nn.functional.one_hot(actions, num_classes=self.action_size).float() * target_q.unsqueeze(1)
        )
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def replay_optimized(self, batch_size=None):
        """
        Optimierte Experience Replay Implementierung
        
        Args:
            batch_size: Größe des Trainings Batches
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Predict Q(s,a)
        q_predict = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute Q(s',a) Target
        with torch.no_grad():
            q_target_next = self.model(next_states).max(1)[0]
            q_target = rewards + (self.gamma * q_target_next * (1 - dones))
        
        # MSE Loss
        loss = self.model.loss_fn(q_predict, q_target)
        
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        """Modell speichern"""
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath):
        """Modell laden"""
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
