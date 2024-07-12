from typing import List, Tuple, Dict
from enum import IntEnum
from option import Option
from gridworld import GridWorld, Actions
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

    for idx, cell in env.idx_to_cell.items():
        if idx in option.termination_set:
            plt.annotate(r'$\mathbf{T}$', cell[::-1], va='center', ha='center',
                         c='r')
        else:
            plt.annotate('⇩⇨⇧⇦'[option.policy[idx]], cell[::-1], va='center',ha='center')

    plt.savefig(f'{suffix}.png')
    plt.close()

def create_primitive_options(env: GridWorld, actions: List[IntEnum]) -> Dict[
    str, Option]:
    """Create primitive options from a list of actions"""
    primitive_options = {}

    states = list(env.idx_to_cell.keys())

    # Create an option for each action
    for action in actions:
        primitive_options[str(action).split(".")[-1]] = \
            Option(init_set=states,
                   term_set=states,
                   policy={state: action for state in states})

    return primitive_options


def _create_eigenoption(env: GridWorld, k: int, discount: float) -> (
        Option):
    """:k: is the index of the eigenpurpose to create the option from"""

    # returns value function (V) and set of terminal states (T)
    V, T = env.value_iterate(k, p=1, gamma=discount)

    policy = {}
    init_set = []
    actions = [Actions.down, Actions.right, Actions.up, Actions.left]

    for cell in zip(*np.where(env.grid)):
        values = [env.r(k, cell, next_cell) + discount * V[next_cell] for
                  next_cell in env.find_adjacent(cell)]
        if len(np.unique(values)) > 1:
            policy[env.cell_to_idx[cell]] = actions[np.argmax(values)]
            init_set.append(env.cell_to_idx[cell])

    return Option(init_set=init_set,
                  policy=policy,
                  term_set=[env.cell_to_idx[term_cell] for term_cell in T])


def create_eigenoptions(env: GridWorld, n_eigenoptions: int, discount: float) \
        -> (
        Dict)[str, Option]:
    """Creates eigenoptions from the environment"""
    eigenoptions = {}
    for k in range(n_eigenoptions):
        eigenoptions[f"pvf_{k}"] = _create_eigenoption(env, k, discount)

    return eigenoptions