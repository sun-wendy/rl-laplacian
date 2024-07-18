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

    def calculate(self, state: int, next_state: int, start_state_idx: int, ideal_state_arr: list):
        """Calculates distance impact penalty. Since the state is equal to the
        counterfactual state in deterministic environments, the distance between
        the state and counterfactual state is always zero"""
        return self.discount * self.distance(next_state, state, start_state_idx, ideal_state_arr)
    
    def get_ideal_state(self, start_state: int, term_states_idx: list):
        """Summarize what the ideal (intermediate) terminal state should be for the current option"""
        raise NotImplementedError


class BrokenVaseDistance(DistanceImpactPenalty):
    """Counts the difference in the number of broken vases between two states"""
    def __init__(self, env):
        super().__init__(env)

    def distance(self, state_1: int, state_2: int, start_state_idx: int, ideal_state_arr: list):
        """Calculates the difference in number of broken vases between two states"""
        return abs(sum(self.env.idx_to_state[state_2][-1]) -
                   sum(self.env.idx_to_state[state_1][-1]))


class HardcodedDistance(DistanceImpactPenalty):
    """Masks the state diff vector w/ an importance mask"""
    def __init__(self, env):
        super().__init__(env)
    
    def distance(self, state_1: int, state_2: int, start_state_idx: int, ideal_state_arr: list):
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
        start_state_flattened = flatten_tuple(self.env.idx_to_state[start_state])
        term_states = [self.env.idx_to_state[idx] for idx in term_states_idx]

        for i, start_state_var in enumerate(start_state_flattened):
            var_ever_match = False
            for term_state in term_states:
                term_state_flattened = flatten_tuple(term_state)
                if start_state_var == term_state_flattened[i]:
                    var_ever_match = True
                    break
            if var_ever_match:
                ideal_state.append(start_state_var)
            else:
                ideal_state.append(flatten_tuple(term_states[0])[i])  # TODO: Assuming a single ideal state, need to relax this later
        
        return ideal_state


    def distance(self, state_1: int, state_2: int, start_state_idx: int, ideal_state_arr: list):
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

        start_state = self.env.idx_to_state[start_state_idx]
        start_state_flattened = flatten_tuple(start_state)
        assert len(ideal_state_arr) == len(start_state_flattened)
        importance_mask = np.array([int(start_state_flattened[i] == ideal_state_arr[i]) for i in range(len(start_state_flattened))])
        state_1, state_2 = self.env.idx_to_state[state_1], self.env.idx_to_state[state_2]
        state_1_arr = np.array([state_1[0], state_1[1]] + list(state_1[2]))
        state_2_arr = np.array([state_2[0], state_2[1]] + list(state_2[2]))
        state_diff = state_2_arr - state_1_arr
        assert len(state_diff) == len(importance_mask)
        return abs(np.dot(state_diff, importance_mask))
