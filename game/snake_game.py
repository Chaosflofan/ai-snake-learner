import numpy as np
import pygame
from enum import Enum
from collections import deque

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class SnakeGame:
    """Snake Game Environment für Training"""
    
    def __init__(self, grid_size=20, render=False):
        self.grid_size = grid_size
        self.render_mode = render
        
        if self.render_mode:
            pygame.init()
            self.cell_size = 20
            self.window = pygame.display.set_mode(
                (grid_size * self.cell_size, grid_size * self.cell_size)
            )
            pygame.display.set_caption("Snake AI Learning")
            self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        """Spiel zurücksetzen"""
        start_pos = self.grid_size // 2
        self.snake = deque([(start_pos, start_pos)])
        self.direction = Direction.RIGHT
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.max_steps = 100 * self.grid_size
        
        return self._get_state()
    
    def _spawn_food(self):
        """Essen spawnen"""
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.snake:
                return (x, y)
    
    def step(self, action):
        """Action ausführen"""
        self.steps += 1
        self._set_direction(action)
        
        head_x, head_y = self.snake[0]
        
        if self.direction == Direction.UP:
            head_y -= 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        elif self.direction == Direction.RIGHT:
            head_x += 1
        
        # Collision Check
        if (head_x < 0 or head_x >= self.grid_size or 
            head_y < 0 or head_y >= self.grid_size):
            return self._get_state(), -10.0, True
        
        if (head_x, head_y) in self.snake:
            return self._get_state(), -10.0, True
        
        self.snake.appendleft((head_x, head_y))
        
        reward = 0.0
        if (head_x, head_y) == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._spawn_food()
        else:
            self.snake.pop()
            reward = -0.01
        
        done = self.steps >= self.max_steps
        
        if self.render_mode:
            self.render()
        
        return self._get_state(), reward, done
    
    def _set_direction(self, action):
        """Richtung setzen"""
        new_direction = Direction(action)
        
        if (self.direction == Direction.UP and new_direction == Direction.DOWN):
            return
        if (self.direction == Direction.DOWN and new_direction == Direction.UP):
            return
        if (self.direction == Direction.LEFT and new_direction == Direction.RIGHT):
            return
        if (self.direction == Direction.RIGHT and new_direction == Direction.LEFT):
            return
        
        self.direction = new_direction
    
    def _get_state(self):
        """State als Array"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        state = [
            head_x / self.grid_size,
            head_y / self.grid_size,
            len(self.snake) / (self.grid_size * self.grid_size),
            (food_x - head_x) / self.grid_size,
            (food_y - head_y) / self.grid_size,
        ]
        
        up_danger = self._is_collision(head_x, head_y - 1)
        down_danger = self._is_collision(head_x, head_y + 1)
        left_danger = self._is_collision(head_x - 1, head_y)
        right_danger = self._is_collision(head_x + 1, head_y)
        
        state.extend([float(up_danger), float(down_danger), 
                      float(left_danger), float(right_danger)])
        
        direction_encoding = [0, 0, 0, 0]
        direction_encoding[self.direction.value] = 1
        state.extend(direction_encoding)
        
        return np.array(state, dtype=np.float32)
    
    def _is_collision(self, x, y):
        """Collision Check"""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if (x, y) in self.snake:
            return True
        return False
    
    def render(self):
        """Spiel rendern"""
        self.window.fill((0, 0, 0))
        
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(
                self.window,
                color,
                (x * self.cell_size, y * self.cell_size, 
                 self.cell_size - 1, self.cell_size - 1)
            )
        
        x, y = self.food
        pygame.draw.rect(
            self.window,
            (255, 0, 0),
            (x * self.cell_size, y * self.cell_size,
             self.cell_size - 1, self.cell_size - 1)
        )
        
        pygame.display.flip()
        self.clock.tick(10)
    
    def get_state_size(self):
        return 13
    
    def get_action_size(self):
        return 4
