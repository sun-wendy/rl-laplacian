# Below defines the gridworlds where space denotes a wall and a non-space
# character denotes an available state. Each has a comment with a number
# reference as to where it comes from.
import os
import itertools
import numpy as np
from enum import IntEnum
from typing import Optional, List, Tuple
from matplotlib import animation, pyplot as plt
from gymnasium import Env, spaces, core
plt.rcParams['animation.html'] = 'jshtml'

# 1 (original name 10 x 10)
one_room = """
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
"""

# 1
i_maze = """
O             0
OOOOOOOOOOOOOOO
O             O
"""

# 1
four_rooms = """
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
OOOOO OOOOO
 O    OOOOO
OOOOO   O  
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
"""

four_rooms_alt = """
OOOOO0OOOOO
OOOOO OOOOO
OOOOO OOOOO
OOOOO OOOOO
OOOOO OOOOO
0     OOOOO
OOOOO     0
OOOOO OOOOO
OOOOO OOOOO
OOOOO OOOOO
OOOOO0OOOOO
"""

# 2
two_rooms = """
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
        OO      
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
"""

# 2
hard_maze = """
OOOOOOOOOOOOOO
OOOOOOOOOOOOOO
      OO
OOOOO OOOOOOOO
OOOOO OOOOOOOO
OO     OO
OOOOOOOOOOOOOO
OOOOOOOOOOOOOO
OO OO OO
OO OO OO OOOOO
OO OO OO OOOOO
OO    OO    OO
OOOOO OOOOOOOO
OOOOO OOOOOOOO
"""

# 3
three_rooms = """
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
OOOOO OOOOO
OOOOO 
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
OOOOO OOOOO
"""


# Actions: '⇩⇨⇧⇦'
class Actions(IntEnum):
    down = 0
    right = 1
    up = 2
    left = 3


class GridWorldWithVases(Env):
    """Simple gridworld with vases"""

    def __init__(self, grid, goal_coords=(11, 11), agent_start_coords=(1, 1), 
                _max_steps=100, vase_coords: Optional[List[Tuple[int]]] = []):
        self.name = grid
        self.vase_coords = vase_coords
        self.broken_vase_coords = []
        self._grid = self.get_grid(grid)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self._grid == 1))
        self.directions = [np.array((1, 0)), np.array((0, 1)), np.array((-1, 0)),
                           np.array((0, -1))]  # down,right,up,left

        # Create a mapping between cell coordinates and state indices
        # First coord of the state is the row and second coord is the column
        self.state_to_idx = {}
        state_num = 0
        for coords in zip(*np.where(self._grid)):
            if self._grid[coords] == 1:
                for broken_vases in itertools.product([0, 1], repeat=len(vase_coords)):
                    # broken_vases is a tuple of length :vase_coords:
                    # broken_vases[i] is 0 if (and only if) the vase at vase_coords[i] is not broken
                    self.state_to_idx[(*coords, broken_vases)] = state_num
                    state_num += 1

        self.idx_to_state = {v: k for k, v in self.state_to_idx.items()}
        self.update_freq = np.zeros(2 + len(self.idx_to_state[0][2]))

        assert agent_start_coords != goal_coords, ('agent_start_state and goal_statere the same')
        self.agent_start_coords = agent_start_coords
        self.agent_state = None
        self.goal_coords = goal_coords
        self.term_states = [k for k in self.state_to_idx.keys() if k[0] == goal_coords[0] and k[1] == goal_coords[1]]

        self.init_states = list(range(self.observation_space.n))
        #self.init_states.remove(self.goal)
        self.first = True

        self.episode_steps = 0
        self._max_steps = _max_steps
        self.info = {}

    def reset(self):
        self.episode_steps = 0
        self.broken_vase_coords = []
        state_idx = self.state_to_idx[(*self.agent_start_coords, tuple([0]*len(self.vase_coords)))]
        self.agent_state = self.idx_to_state[state_idx]
        self.update_freq = np.zeros(2 + len(self.idx_to_state[0][2]))
        return state_idx, self.info

    def set_goal(self, goal_coords: Tuple[int]):
        self.goal_coords = goal_coords

    def step(self, action: int):
        """
        Take one step in the environment
        """
        next_state_coords = tuple(np.array((self.agent_state[0],
                                   self.agent_state[1])) + self.directions[action])
        
        if next_state_coords[0] != self.agent_state[0]:
            self.update_freq[0] += 1
        if next_state_coords[1] != self.agent_state[1]:
            self.update_freq[1] += 1

        # Check if we can move to the next cell
        try:
            self.info['hit_vase'] = False

            if self._grid[next_state_coords] == 1:
                # Check if an agent has stepped in a square with a vase
                if (next_state_coords in self.vase_coords and
                        next_state_coords not in self.broken_vase_coords):

                    # Update the broken vases list
                    broken_vases = list(self.agent_state[-1])
                    new_broken_vase_idx = self.vase_coords.index(next_state_coords)
                    broken_vases[new_broken_vase_idx] = 1
                    self.broken_vase_coords.append(next_state_coords)

                    # Update the agent state
                    self.agent_state = (*next_state_coords, tuple(broken_vases))
                    self.info['hit_vase'] = True
                    self.update_freq[new_broken_vase_idx+2] += 1
                else:
                    self.agent_state = (*next_state_coords, self.agent_state[-1])

        except ValueError:
            print("Next State Coords: ", next_state_coords)
            print(self._grid[next_state_coords])
            print("Error!")
            exit(1)

        next_state_idx = self.state_to_idx[self.agent_state]
        reward = float((self.agent_state[0], self.agent_state[1]) ==
                       self.goal_coords)

        truncation = False #TODO: fix for option steps self.episode_steps >= self._max_steps
        done = reward > 0 or truncation

        #self.episode_steps += 1

        # next_state, reward, done, truncation, info
        return next_state_idx, reward, done, truncation, self.info

    def get_grid(self, name):
        if name not in globals():
            raise Exception(f'"{name}" not recognised!')
        string = globals().get(name)
        split = string.split('\n')[1:-1]
        n = len(split)
        m = max(map(len, split))
        grid = np.zeros((n, m))
        for i, row in enumerate(split):
            grid[i, np.where(np.array([*row]) != ' ')[0]] = 1.
        return np.pad(grid, 1)

    def render_frame(self) -> np.ndarray:
        """ Renders the environment and returns a frame as a numpy array,
        with dimensions (width, height, channel)"""
        frame = np.zeros((self._grid.shape[0], self._grid.shape[1], 3),
                         dtype=np.uint8)

        frame[self._grid == 1] = [255, 255, 255]
        frame[self.goal_coords] = [0, 255, 0]
        frame[(self.agent_state[0], self.agent_state[1])] = [50, 0, 255]
        for vase_coords in self.vase_coords:
            if vase_coords in self.broken_vase_coords:
                continue
            frame[vase_coords] = [255, 255, 0]

        return frame