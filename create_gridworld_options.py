from typing import List, Tuple, Dict, Optional, Union
from enum import IntEnum
from option import Option
# from safe_option import SafeOption
from env.gridworld import GridWorld, Actions
import numpy as np
import matplotlib.pyplot as plt


def plot_option(env: GridWorld, option: Option, suffix=''):
    """ Renders the environment """
    # Plot the environment
    plt.imshow(env._grid, cmap='magma')
    for i in range(env._grid.shape[0]):
        plt.hlines(i + 0.5, -0.5, env._grid.shape[1] - 0.5, alpha=0.2)
    for j in range(env._grid.shape[1]):
        plt.vlines(j + 0.5, -0.5, env._grid.shape[0] - 0.5, alpha=0.2)
    plt.xticks([])
    plt.yticks([])

    for idx, state in env.idx_to_state.items():
        if idx in option.termination_set:
            plt.annotate(r'$\mathbf{T}$', state[::-1], va='center', ha='center',
                         c='r')
        else:
            plt.annotate('⇩⇨⇧⇦'[option.policy[idx]], state[::-1], va='center',ha='center')

    plt.savefig(f'{suffix}.png')
    plt.close()


def create_primitive_options(env: GridWorld, actions: Optional[List[IntEnum]] =
None) -> Dict[str, Option]:
    """Create primitive options from a list of actions"""
    primitive_options = {}

    state_idxs = list(env.idx_to_state.keys())

    if actions is None:
        actions = [Actions.down, Actions.right, Actions.up, Actions.left]

    # Create an option for each action
    for action in actions:
        primitive_options[str(action).split(".")[-1]] = \
            Option(init_set=state_idxs,
                   term_set=state_idxs,
                   policy={state_idx: action for state_idx in state_idxs})

    return primitive_options


def _create_eigenoption(env: GridWorld, k: int, discount: float) -> (
        Option):
    """:k: is the index of the eigenpurpose to create the option from"""

    # returns value function (V) and set of terminal states (T)
    V, T = env.value_iterate(k, p=1, gamma=discount)

    policy = {}
    init_set = []
    actions = [Actions.down, Actions.right, Actions.up, Actions.left]

    for idx, state in env.idx_to_state.items():
        values = [env.r(k, state, next_state) + discount * V[next_state] for
                  next_state in env.find_adjacent(state)]
        if len(np.unique(values)) > 1:
            policy[idx] = actions[np.argmax(values)]
            init_set.append(idx)

    return Option(init_set=init_set,
                  policy=policy,
                  term_set=[env.state_to_idx[term_state] for term_state in T])


def create_eigenoptions(env: GridWorld, n_eigenoptions: int, discount: float) \
        -> (
        Dict)[str, Option]:
    """Creates eigenoptions from the environment"""
    eigenoptions = {}
    for k in range(n_eigenoptions):
        eigenoptions[f"pvf_{k}"] = _create_eigenoption(env, k, discount)

    return eigenoptions