from typing import Optional, List, Dict, Tuple, Callable
from enum import IntEnum


class Option:
    def __init__(self, init_set: List[Tuple[int]], policy: Dict[Tuple[int], int],
                 term_set: List[Tuple[int]],
                 policy_selection: Callable,
                 term_condition: Optional[Callable] = None):
        """
        Base class to create an option. Adapted from
        https://github.com/s-mawjee/q-learning-with-options/blob/master/option.py

        :param init_set: Set of states in which the option can start
        :param policy: Policy of the option
        :param term_set: Set of states in which the option can end
        :param policy_selection: Function to select an action from the policy
        :param term_condition: Function which determines if the option ends
        """
        self.initialisation_set = init_set
        self.policy = policy
        self.termination_set = term_set
        self._policy_selection = policy_selection
        if term_condition is not None:
            self._termination_condition = term_condition
        else:
            self._termination_condition = self.check_membership

    def policy_selection(self, state: Tuple[int]) -> int:
        return self._policy_selection(self.policy, state)

    def termination_condition(self, state: Tuple[int]) -> bool:
        return self._termination_condition(self.termination_set, state)

    def check_membership(self, term_set: List[Tuple[int]], state: Tuple[int]) -> (
            bool):
        return True if state in term_set else False

    def step(self, state):
        return self.policy_selection(state), self.termination_condition(state)


def create_primitive_options(states: List[Tuple[int]], actions: List[IntEnum]) -> Dict[str, Option]:
    """Create primitive options from a list of actions"""
    primitive_options = {}

    def _select_action(policy: Dict[Tuple[int], int], state: Tuple[int]) -> int:
        """Simple function to select an action from a deterministic policy."""
        assert state in policy.keys(), f"Invalid state {state}"
        return policy[state]

    # Create an option for each action
    for action in actions:
        primitive_options[str(action).split(".")[-1]] = \
            Option(init_set=states,
                   term_set=states,
                   policy={state: action for state in states},
                   policy_selection=_select_action)

    return primitive_options

def get_options(option_sets, env):
    pass

