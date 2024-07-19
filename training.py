"""Training loop"""
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent
from env.gridworld import Actions, GridWorld
from env.gridworld_with_vases import GridWorldWithVases
from create_gridworld_options import create_primitive_options, create_eigenoptions
import distance_impact_penalties as dip


def get_env(env_name: str, _max_steps: int, diffusion='normalised'):
    """Get a copy of the environment"""
    assert env_name in ['one_room', 'two_rooms', 'four_rooms', 'i_maze',
                        'hard_maze', 'four_rooms_alt','four_rooms_alt_with_vases'],\
                       f"Invalid environment name: {env_name}"

    if "vases" not in env_name:
        env = GridWorld(grid=env_name, diffusion=diffusion, _max_steps=_max_steps)
    else:
        grid_name = env_name.split("_with_vases")[0]
        if grid_name == 'four_rooms_alt':
            vase_coords = [(1, 3), (1, 9), (3, 1), (9, 1), (11, 3), (11, 9), (9, 11), (3, 11)]
        else:
            raise NotImplementedError
        env = GridWorldWithVases(grid=grid_name, _max_steps=_max_steps, vase_coords=vase_coords)

    return env


def run_loop_to_term_state(agent, env, n_episodes, anneal, term_states_idx):
    """Training an agent to select fixed options."""
    stats = {'return': np.zeros(n_episodes),
             'exploration_rate': np.zeros(n_episodes),
             'total_steps': np.zeros(n_episodes),
             'n_broken_vases': np.zeros(n_episodes),
             'penalty': np.zeros(n_episodes)}

    di_penalty = dip.ImportanceDistance(env)

    if anneal:
        agent.epsilon = 1.0
        eps_unit = 1.0 / n_episodes

    for episode in tqdm(range(n_episodes)):
        return_ = 0
        total_steps = 0
        n_broken_vases = 0
        state_idx, info = env.reset()
        done = False
        total_penalty = 0.0
        start_state_idx = state_idx
        ideal_state_arr = di_penalty.get_ideal_state(start_state_idx, term_states_idx)
        term_coords = [(env.idx_to_state[idx][0], env.idx_to_state[idx][1]) for idx in term_states_idx]

        while not done and total_steps < env._max_steps:
            action_idx = agent.choose_action(state_idx)
            next_state_idx, reward, done, truncated, info = env.step(action_idx)
            n_broken_vases += int(info["hit_vase"])
            total_steps += 1
            state_coords = env.idx_to_state[next_state_idx][:2]

            if state_coords in term_coords:
                reward += 1
                done = True

            penalty = di_penalty.calculate(state_idx, next_state_idx, start_state_idx, ideal_state_arr)
            total_penalty += penalty
            reward -= penalty
            agent.update(state_idx, action_idx, reward, next_state_idx, done)
            state_idx = next_state_idx
            return_ += np.power(agent.discount, total_steps) * reward

        stats['return'][episode] = return_
        stats['total_steps'][episode] = total_steps
        stats['n_broken_vases'][episode] = n_broken_vases
        stats['exploration_rate'][episode] = agent.epsilon
        stats['penalty'][episode] = total_penalty

        if anneal:
            agent.epsilon = max(0, agent.epsilon - eps_unit)

    return stats


def run_loop_fixed_options(agent, env, options, n_episodes, anneal):
    """Training an agent to select fixed options."""
    stats = {'return': np.zeros(n_episodes),
             'option_return': np.zeros(n_episodes),
             'option_steps': np.zeros(n_episodes),
             'exploration_rate': np.zeros(n_episodes),
             'total_steps': np.zeros(n_episodes),
             'n_broken_vases': np.zeros(n_episodes)}

    _options = list(options.values())
    option_names = list(options.keys())

    assert agent.n_actions == len(_options), ("Number of agent actions must match number of options.")

    if anneal:
        agent.epsilon = 1.0
        eps_unit = 1.0 / n_episodes

    for episode in tqdm(range(n_episodes)):
        return_ = 0
        option_return_ = 0
        total_steps = 0
        option_steps = 0  # Number of times a new option is selected
        n_broken_vases = 0
        state_idx, info = env.reset()
        done = False

        while not done and total_steps < env._max_steps:
            option_idx = agent.choose_action(state_idx)
            option = _options[option_idx]
            action = option.policy_selection(state_idx)
            next_state_idx, reward, done, truncated, info = env.step(action)
            n_broken_vases += int(info["hit_vase"])
            total_steps += 1
            option_steps += 1
            current_option_steps = 1

            # Run the option until it terminates
            while (not option.termination_condition(next_state_idx) and not done and total_steps <= env._max_steps):
                action = option.policy_selection(next_state_idx)
                next_state_idx, next_reward, done, truncated, info = env.step(action)
                n_broken_vases += int(info["hit_vase"])
                reward += next_reward
                total_steps += 1

            agent.update(state_idx, option_idx, reward, next_state_idx, done)
            state_idx = next_state_idx
            return_ += np.power(agent.discount, total_steps) * reward
            option_return_ += np.power(agent.discount, option_steps) * reward

        stats['return'][episode] = return_
        stats['option_return'][episode] = option_return_
        stats['total_steps'][episode] = total_steps
        stats['option_steps'][episode] = option_steps
        stats['n_broken_vases'][episode] = n_broken_vases
        stats['exploration_rate'][episode] = agent.epsilon
        #stats['n_broken_vases'][episode] = n_broken_vases

        if anneal:
            agent.epsilon = max(0, agent.epsilon - eps_unit)

    return stats


def run_agent(learning_rate, discount, anneal, n_episodes, seed, env_name,
              diffusion, max_steps, agent_class, n_eigenoptions, penalty_strength):
    """Run agent

    Create an agent with the given parameters for the side effects penalty.
    Run the agent for `n_episodes` episodes with an exploration rate that is
    either annealed from 1 to 0 (`anneal=True`) or constant (`anneal=False`).

    Args:
        learning_rate: learning_rate
        discount: discount factor
        anneal: whether to anneal the exploration rate from 1 to 0 or use a constant
                exploration rate
        n_episodes: number of episodes
        seed: random seed
        env_name: environment name
        max_steps: maximum number of steps per episode
        agent_class: Q-learning agent class

    Returns:
        stats: training statistics
        agent: trained agent
    """
    np.random.seed(seed)
    env = get_env(env_name=env_name, _max_steps=max_steps)

    if agent_class == QLearningAgent:
        # Create four primitive options for each of the base actions
        # options = create_primitive_options(env)

        if n_eigenoptions > 0:
            # eigenoptions = create_eigenoptions(env, n_eigenoptions, discount)
            # options.update(eigenoptions)
            base_env = get_env(env.name.split("_with_vases")[0], _max_steps=max_steps, diffusion=diffusion)
            eigenoptions = create_eigenoptions(base_env, n_eigenoptions, discount)
            term_states_idx = list(eigenoptions.values())[0].termination_set
        else:
            term_states = [(2, 5, (0, 0, 0, 0, 0, 0, 0, 0)), (5, 2, (0, 0, 0, 0, 0, 0, 0, 0))]
            term_states_idx = [env.state_to_idx[state] for state in term_states]

        agent = QLearningAgent(n_actions=env.action_space.n, learning_rate=learning_rate, discount=discount)
        # stats = run_loop_fixed_options(agent, env, options=options, n_episodes=n_episodes, anneal=anneal)
        stats = run_loop_to_term_state(agent, env, n_episodes, anneal, term_states_idx)

    else:
        raise ValueError(f'Invalid agent class {agent_class}')

    return stats, agent



if __name__ == "__main__":
    env_name = 'four_rooms_alt_with_vases'
    diffusion = "normalised"
    n_eigenoptions = 0
    discount = 0.9
    env = get_env(env_name, _max_steps=100, diffusion=diffusion)

    # options = create_primitive_options(env)

    # if n_eigenoptions > 0:
    #     eigenoptions = create_eigenoptions(env, n_eigenoptions, discount=discount)
    #     options.update(eigenoptions)
    #     from create_gridworld_options import plot_option

    #     for name, option in options.items():
    #         if 'pvf' not in name:
    #             continue
    #         plot_option(env, option, f'figures/option_plots/{env_name}/diffusion_'
    #                                  f'{diffusion}/{env_name}_diffusion_{diffusion}_{name}')

    state_idx, info = env.reset()

    for i in range(20):
        action = env.action_space.sample()
        next_state_idx, reward, done, truncated, info = env.step(action)
        state_idx = next_state_idx
        frame = env.render_frame()
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(f'figures/step_{i}.png', dpi=400)
        plt.close()
