"""File for adding distance impact penalties"""
import numpy as np
from typing import List
from collections import defaultdict


class DistanceImpactPenalty:
    """Template class for a distance impact penalty. In the FourRooms
    environment, the counterfactual state is always just the state"""
    def __init__(self, env, discount=1.):
        self.env = env
        self.discount = discount

    def distance(self, state_1: int, state_2: int):
        """Calculates a (metric) distance between two states"""
        raise NotImplementedError

    def calculate(self, state: int, next_state: int):
        """Calculates distance impact penalty. Since the state is equal to the
        counterfactual state in deterministic environments, the distance between
        the state and counterfactual state is always zero"""
        return self.discount * self.distance(next_state, state)
    
    def get_ideal_state(self, start_state: int, term_states_idx: list):
        """Summarize what the ideal (intermediate) terminal state should be for the current option"""
        raise NotImplementedError


class BrokenVaseDistance(DistanceImpactPenalty):
    """Counts the difference in the number of broken vases between two states"""
    def __init__(self, env):
        super().__init__(env)

    def distance(self, state_1: int, state_2: int):
        """Calculates the difference in number of broken vases between two states"""
        return abs(sum(self.env.idx_to_state[state_2][-1]) -
                   sum(self.env.idx_to_state[state_1][-1]))


class HardcodedDistance(DistanceImpactPenalty):
    """Masks the state diff vector w/ an importance mask"""
    def __init__(self, env):
        super().__init__(env)
    
    def distance(self, state_1: int, state_2: int):
        importance_mask = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        state_1, state_2 = self.env.idx_to_state[state_1], self.env.idx_to_state[state_2]
        state_1_arr = np.array([state_1[0], state_1[1]] + list(state_1[2]))
        state_2_arr = np.array([state_2[0], state_2[1]] + list(state_2[2]))
        state_diff = state_2_arr - state_1_arr
        assert len(state_diff) == len(importance_mask)
        return abs(np.dot(state_diff, importance_mask))


class ImportanceDistance(DistanceImpactPenalty):
    """Counts the difference in the number of broken vases between two states"""
    def __init__(self, env, start_state_idx: int, term_states_idx: List[int],
                 discount=0.9, learning_rate=0.1, init_reach_prob=1.):
        super().__init__(env)
        self.start_state_idx = start_state_idx
        self.term_states_idx = term_states_idx
        print("Start state idx: ", self.start_state_idx)
        print("End state idx: ", self.term_states_idx)

        # Find the closest terminal state to the start state
        self.ideal_state_idx = self.get_ideal_state(start_state_idx,
                                                    term_states_idx)

        print("Ideal state idx: ", self.ideal_state_idx)

        # Estimate how reachable the ideal state is
        self.reach_estimator = ReachabilityEstimator(n_actions=env.action_space.n,
                                                     goal_state_idxs=[self.ideal_state_idx],
                                                     init_reach_prob=init_reach_prob,
                                                     discount=discount,
                                                     learning_rate=learning_rate)

    def get_ideal_state(self, start_state: int, term_states_idx: list):
        """Summarize what the ideal (intermediate) terminal state should be for the current option"""

        def flatten_tuple(t):
            flattened_list = []

            def recursive_flatten(x):
                if isinstance(x, tuple):
                    for y in x:
                        recursive_flatten(y)
                else:
                    flattened_list.append(x)

            recursive_flatten(t)
            return flattened_list

        ideal_state = []
        start_state = self.env.idx_to_state[start_state]
        start_state = ((start_state[0], start_state[1]), start_state[2])
        term_states = [self.env.idx_to_state[idx] for idx in term_states_idx]
        term_states = [((term_state[0], term_state[1]), term_state[2]) for term_state in term_states]

        for i, start_state_var in enumerate(start_state):
            print("Start var: ", start_state_var)
            var_ever_match = False
            for term_state in term_states:
                if start_state_var == term_state[i]:
                    var_ever_match = True
                    break
            if var_ever_match:
                ideal_state.append(start_state_var)
            else:
                ideal_state.append(term_states[0][i])  # TODO: Assuming a single ideal state, need to relax this later

        ideal_state = tuple(ideal_state)
        ideal_state_idx = self.env.state_to_idx[(ideal_state[0][0], ideal_state[0][1], ideal_state[1])]

        assert ideal_state_idx in term_states_idx, "Ideal state not in term states"

        return ideal_state_idx

    def distance(self, state_1_idx: int, state_2_idx: int):
        """Calculates the difference in reachability to the ideal state between
        two states. Higher positive value means that the ideal state is less
        reachable.

        For now we ensure that the return value is always non-negative"""
        return 1 - np.max(self.reach_estimator.predict(state_2_idx))


class ReachabilityEstimator:
    """Estimates how reachable a set of goal states is"""
    def __init__(self, n_actions: int, goal_state_idxs: List[int],
                 init_reach_prob=1., discount=0.9, learning_rate=0.1):
        self.n_actions = n_actions
        self.goal_state_idxs = goal_state_idxs

        # Hyperparameters
        self.discount = discount
        self.learning_rate = learning_rate
        self.init_reach_prob = init_reach_prob

        # Initialize the Q table
        self.q_table = {}
        for goal_state in goal_state_idxs:
            self.q_table[goal_state] = np.ones(self.n_actions)

    def get_q_vals(self, state: int) -> np.ndarray:
        if state not in self.q_table.keys():
            self.q_table[state] = self.init_reach_prob * np.ones(self.n_actions)

        return self.q_table[state]

    def update(self, state_idx, action, next_state_idx):
        """Updates the reachability probabilities"""
        reward = 1 if next_state_idx in self.goal_state_idxs else 0

        current_q = self.get_q_vals(state_idx)[action]
        max_next_q = np.max(self.get_q_vals(next_state_idx))
        new_q = current_q + self.learning_rate * (reward + (
                1-reward) * self.discount * max_next_q - current_q)

        self.get_q_vals(state_idx)[action] = new_q

    def predict(self, state_idx: int) -> np.ndarray:
        """Given a state, return the probability of reaching the goal states for
        each action"""
        return self.get_q_vals(state_idx)