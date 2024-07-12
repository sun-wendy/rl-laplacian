"""Training loop"""
from typing import Optional
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from tqdm import tqdm
import numpy as np
import itertools
from agent import SMDPQLearningAgent, QLearningAgent
from gridworld import Actions, GridWorld
from create_gridworld_options import create_primitive_options, create_eigenoptions


def get_env(env_name: str, max_steps: int, diffusion='normalised'):
    """Get a copy of the environment"""
    assert env_name in ['one_room', 'two_rooms', 'four_rooms', 'i_maze',
                        'hard_maze'], f"Invalid environment name: {env_name}"

    env = GridWorld(grid=env_name, diffusion='normalised', max_steps=max_steps)

    return env


def run_loop(agent, env, n_episodes, anneal):
    """Training agent"""
    stats = {'return': np.zeros(n_episodes),
             'total_steps': np.zeros(n_episodes)}
             #'n_broken_vases': np.zeros(n_episodes)}

    if anneal:
        agent.epsilon = 1.0
        eps_unit = 1.0 / n_episodes

    for episode in tqdm(range(n_episodes)):
        return_ = 0
        total_steps = 0
        n_broken_vases = 0
        state, info = env.reset()
        done = False

        while not done and total_steps < env.max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            has_updated = agent.update(state, action, reward, next_state, done)
            state = next_state
            return_ += reward * np.power(agent.discount, total_steps)
            if has_updated:
                total_steps += 1
            #n_broken_vases += int(info['hit_vase'])

        stats['return'][episode] = return_
        stats['total_steps'][episode] = total_steps
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
    env = get_env(env_name=env_name, max_steps=max_steps, diffusion=diffusion)

    if agent_class == SMDPQLearningAgent:

        options = create_primitive_options(env, actions=[Actions.down, Actions.right, Actions.up, Actions.left])

        if n_eigenoptions > 0:
            eigenoptions = create_eigenoptions(env, n_eigenoptions, discount)
            options.update(eigenoptions)
        print(options.keys())

        agent = agent_class(options=options,
                            learning_rate=learning_rate,
                            discount=discount)

    elif agent_class == QLearningAgent:

        agent = agent_class(n_actions=env.action_space.n,
                            learning_rate=learning_rate,
                            discount=discount)
    else:
        raise ValueError(f'Invalid agent class {agent_class}')

    stats = run_loop(agent, env, n_episodes=n_episodes, anneal=anneal)

    return stats, agent


if __name__ == "__main__":
    env_name = 'four_rooms'
    diffusion = None
    n_eigenoptions = 10
    env = get_env(env_name, max_steps=100, diffusion=diffusion)
    env.render()


    options = create_primitive_options(env, actions=[Actions.down, Actions.right, Actions.up, Actions.left])

    eigenoptions = create_eigenoptions(env, n_eigenoptions, discount=0.99)

    options.update(eigenoptions)
    from create_gridworld_options import plot_option

    for name, option in options.items():
        if 'pvf' not in name:
            continue

        plot_option(env, option, f'figures/option_plots/{env_name}/diffusion_'
                                 f'{diffusion}/{env_name}_diffusion_{diffusion}_{name}')


    print("Done.")


    #for _ in range(20):
    #    action = env.action_space.sample()
    #    next_state, reward, done, truncated, info = env.step(action)
    #    print(f'Cell: {env.idx_to_cell[state]}, '
    #          f'Action: {action_names[action]}, '
    #          f'Next Cell: {env.idx_to_cell[next_state]}')
    #    state = next_state

