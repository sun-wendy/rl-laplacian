from typing import Dict, Tuple, List, Optional
from option import Option
import numpy as np


class QLearningAgent:
    """Q-learning agent"""
    def __init__(self, n_actions: int, learning_rate=0.1,
                 epsilon=0.1,
                 q_initialisation=0.0, discount=0.99):
        """Create a Q-learning agent

        Args:
        n_actions: number of valid actions
        alpha: agent learning rate
        epsilon: agent exploration rate
        q_initialisation: float, used to initialise the value function
        discount: discount factor for rewards
        action_names: names of the actions
        """
        self.q_table = {}
        self.n_actions = n_actions
        self.available_actions = [a for a in range(n_actions)]
        self.q_initialisation = q_initialisation

        # Hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # exploration rate
        self.discount = discount

    def get_q_vals(self, state: Tuple[int]) -> np.ndarray:
        if state not in self.q_table.keys():
            self.q_table[state] = self.q_initialisation*np.ones(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state: Tuple[int]) -> int:
        """Selects actions from a list of available actions according to an
        epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.available_actions)
        else:
            return np.argmax(self.get_q_vals(state)[self.available_actions])

    def update(self, state: Tuple[int], action: int, reward: float, next_state: Tuple[int], done: bool):
        """Update the q-table"""
        current_q = self.get_q_vals(state)[action]
        max_next_q = np.max(self.get_q_vals(next_state))
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q * (not done) - current_q)
        self.get_q_vals(state)[action] = new_q
