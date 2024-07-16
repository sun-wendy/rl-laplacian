"""Training loop"""
from typing import Optional
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from tqdm import tqdm
import numpy as np
import itertools
from agent import QLearningAgent
from gridworld import Actions, GridWorld
from gridworld_with_vases import GridWorldWithVases
from create_gridworld_options import create_primitive_options, create_eigenoptions


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
            vase_coords = [(1, 3), (1, 9), (3, 1), (9, 1), (11, 3), (11, 9), (9, 11),
                           (3, 11)]
        else:
            raise NotImplementedError

        env = GridWorldWithVases(grid=grid_name, diffusion=diffusion,
                                 _max_steps=_max_steps, vase_coords=vase_coords)

    return env


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

    assert agent.n_actions == len(_options), ("Number of agent actions must match "
                                              "the number of options.")

    if anneal:
        agent.epsilon = 1.0
        eps_unit = 1.0 / n_episodes

    for episode in tqdm(range(n_episodes)):
        return_ = 0
        option_return_ = 0
        total_steps = 0
        option_steps = 0    # number of times a new option is selected
        n_broken_vases = 0
        state_idx, info = env.reset()
        #print(f'Start state: {env.idx_to_state[state_idx]}, index {state_idx}')
        done = False

        while not done and total_steps < env._max_steps:

            option_idx = agent.choose_action(state_idx)
            option = _options[option_idx]
            #print("Selecting option ", option_names[option_idx])
            action = option.policy_selection(state_idx)
            #print('Action: ', action)

            next_state_idx, reward, done, truncated, info = env.step(action)
            n_broken_vases += int(info["hit_vase"])
            total_steps += 1
            option_steps += 1

            current_option_steps = 1
            #print('Option terminates at: ', [env.idx_to_state[t] for t in
            #                                 option.termination_set])
            # Run the option until it terminates
            while (not option.termination_condition(next_state_idx) and not done and
                   total_steps <= env._max_steps):
                #print(f'state: {env.idx_to_state[next_state_idx]}, index'
                #      f' {next_state_idx}')
                action = option.policy_selection(next_state_idx)
                #print('action: ', action)
                next_state_idx, next_reward, done, truncated, info = env.step(
                    action)
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
              diffusion, max_steps, agent_class, n_eigenoptions):
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
    env = get_env(env_name=env_name, _max_steps=max_steps, diffusion=diffusion)

    if agent_class == QLearningAgent:

        # Create four primitive options for each of the base actions
        options = create_primitive_options(env)

        if n_eigenoptions > 0:
            eigenoptions = create_eigenoptions(env, n_eigenoptions, discount)
            options.update(eigenoptions)

            #print("Plotting options...")
            #from create_gridworld_options import plot_option

            #for name, eig_option in eigenoptions.items():

            #    plot_option(env, eig_option, f'figures/option_plots'
            #                              f'/{env_name}_diffusion_{diffusion}_{name}')

            #print("Done.")

        agent = QLearningAgent(n_actions=len(list(options.keys())),
                               learning_rate=learning_rate,
                               discount=discount)

        stats = run_loop_fixed_options(agent, env, options=options,
                                       n_episodes=n_episodes, anneal=anneal)

    else:
        raise ValueError(f'Invalid agent class {agent_class}')

    return stats, agent



if __name__ == "__main__":
    env_name = 'four_rooms_alt_with_vases'
    diffusion = "normalised"
    n_eigenoptions = 0
    discount = 0.9
    env = get_env(env_name, _max_steps=100, diffusion=diffusion)

    options = create_primitive_options(env)

    if n_eigenoptions > 0:
        eigenoptions = create_eigenoptions(env, n_eigenoptions, discount=discount)
        options.update(eigenoptions)
        from create_gridworld_options import plot_option

        for name, option in options.items():
            if 'pvf' not in name:
                continue

            plot_option(env, option, f'figures/option_plots/{env_name}/diffusion_'
                                     f'{diffusion}/{env_name}_diffusion_{diffusion}_{name}')


    import matplotlib.pyplot as plt

    state_idx, info = env.reset()

    for i in range(20):
        action = env.action_space.sample()
        next_state_idx, reward, done, truncated, info = env.step(action)
        print(env.idx_to_state[next_state_idx])
        state_idx = next_state_idx

        frame = env.render_frame()
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(f'figures/step_{i}.png', dpi=400)
        plt.close()

