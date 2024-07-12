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

class QLearningFixedOptions(object):
    """Q learning agent that learns from fixed options."""
    def __init__(self, options: Dict[str, Option], learning_rate=0.1, epsilon=0.1,
                 q_initialisation=0.0, discount=0.99):
        """Create a Q-learning agent
        Args:
        options: list of fixed options. IMPORTANT: options must be executable
        everywhere
        alpha: agent learning rate
        epsilon: agent exploration rate
        q_initialisation: float, used to initialise the value function
        discount: discount factor for rewards
        action_names: names of the actions
        """
        self.q_table = {}
        self.q_initialisation = q_initialisation
        self.options = list(options.values())
        self.option_names = list(options.keys())
        self.n_options = len(self.options)
        self.current_option = None
        self.current_option_idx = None
        self.current_option_start_state = None
        self.current_option_steps = 0

        # Hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount

    def get_q_vals(self, state: Tuple[int]) -> Tuple[int]:
        if state not in self.q_table.keys():
            self.q_table[state] = self.q_initialisation*np.ones(self.n_options)
        return self.q_table[state]

    def _choose_option_idx(self, state: Tuple[int]) -> int:
        """Selects an option index according to epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_options)
        else:
            return np.argmax(self.get_q_vals(state))

    def choose_action(self, state: Tuple[int]) -> int:
        """Select a base action from the option. Records the start state of the
        option and the number of steps taken during the option"""

        # Select an option the current option is not defined, or has terminated
        if (self.current_option is None or
                self.current_option.termination_condition(state)):
            self.current_option_idx = self._choose_option_idx(state)
            self.current_option = self.options[self.current_option_idx]

            self.current_option_start_state = state
            self.current_option_steps = 1

        else:
            self.current_option_steps += 1

        return self.current_option.policy_selection(state)

    def update(self, state: Tuple[int], action: int, reward: float,
               next_state: Tuple[int], done: bool):
        """Define an update rule for the agent"""
        pass


class SMDPQLearningAgent(QLearningFixedOptions):
    """Q learning agent that learns from fixed options. Learns the values of fixed
    according to the SMDP learning update rule. Ref:
     https://web.eecs.umich.edu/~baveja/Papers/ICML98_SPS.pdf"""
    def __init__(self, options: Dict[str, Option], learning_rate=0.1, epsilon=0.1,
                 q_initialisation=0.0, discount=0.99, discount_option_steps=False):
        super().__init__(options, learning_rate, epsilon, q_initialisation, discount)

        self.discount_option_steps = discount_option_steps

    def update(self, state: Tuple[int], action: int, reward: float,
               next_state: Tuple[int], done: bool):
        """Apply the SMDP Q learning update. Note that :state: is not used,
        since update() is called after every base action, but SMDP Q learning
        only applies an update from the state in which the option began.
        # TODO: assure that current option always gets updated!
        """
        assert (self.current_option is not None and
                self.current_option_idx is not None and
                self.current_option_start_state is not None, "No option to update")

        # Only update the q values once the option has terminated
        if self.current_option.termination_condition(next_state):
            current_q = self.get_q_vals(self.current_option_start_state)[self.current_option_idx]
            max_next_q = np.max(self.get_q_vals(next_state))

            if self.discount_option_steps:
                discount = np.power(self.discount, self.current_option_steps)
            else:
                discount = self.discount

            new_q = current_q + self.learning_rate * (reward + discount * max_next_q
                                                      * (not done) - current_q)
            self.get_q_vals(self.current_option_start_state)[self.current_option_idx] = new_q

