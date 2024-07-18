"""File for adding distance impact penalties"""
import numpy as np


class DistanceImpactPenalty:
    """Template class for a distance impact penalty. In the FourRooms
    environment, the counterfactual state is always just the state"""
    def __init__(self, env, discount=1.):
        self.env = env
        self.discount = discount

    def distance(self, state_1: int, state_2: int):
        """Calculates a (metric) distance between two states"""
        raise NotImplementedError

    def calculate(self, state: int, next_state: int, term_states_abstracted: list, term_states: list):
        """Calculates distance impact penalty. Since the state is equal to the
        counterfactual state in deterministic environments, the distance between
        the state and counterfactual state is always zero"""
        return self.discount * self.distance(next_state, state, term_states_abstracted, term_states)


class BrokenVaseDistance(DistanceImpactPenalty):
    """Counts the difference in the number of broken vases between two states"""
    def __init__(self, env):
        super().__init__(env)

    def distance(self, state_2: int, state_1: int, term_states_abstracted: list, term_states: list):
        """Calculates the difference in number of broken vases between two states"""

        return abs(sum(self.env.idx_to_state[state_2][-1]) -
                   sum(self.env.idx_to_state[state_1][-1]))


class HardcodedDistance(DistanceImpactPenalty):
    """Masks the state diff vector w/ an importance mask"""
    def __init__(self, env):
        super().__init__(env)
    
    def distance(self, state_2: int, state_1: int, term_states_abstracted: list, term_states: list):
        importance_mask = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        state_1, state_2 = self.env.idx_to_state[state_1], self.env.idx_to_state[state_2]
        state_1_arr = np.array([state_1[0], state_1[1]] + list(state_1[2]))
        state_2_arr = np.array([state_2[0], state_2[1]] + list(state_2[2]))
        state_diff = state_2_arr - state_1_arr
        assert len(state_diff) == len(importance_mask)
        return abs(np.dot(state_diff, importance_mask))


class ImportanceDistance(DistanceImpactPenalty):
    """Counts the difference in the number of broken vases between two states"""
    def __init__(self, env):
        super().__init__(env)

    def distance(self, state_2: int, state_1: int, term_states_abstracted: list, term_states: list):
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

        term_state_abs, term_state = term_states_abstracted[0], term_states[0]
        flattened_term_state_abs, flattened_term_state = flatten_tuple(term_state_abs), flatten_tuple(term_state)
        importance_mask = np.ones(len(flattened_term_state))
        importance_mask[:len(flattened_term_state_abs)] = 0
        state_1, state_2 = self.env.idx_to_state[state_1], self.env.idx_to_state[state_2]
        state_1_arr = np.array([state_1[0], state_1[1]] + list(state_1[2]))
        state_2_arr = np.array([state_2[0], state_2[1]] + list(state_2[2]))
        state_diff = state_2_arr - state_1_arr
        assert len(state_diff) == len(importance_mask)
        return abs(np.dot(state_diff, importance_mask))
