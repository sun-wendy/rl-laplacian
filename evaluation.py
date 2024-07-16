import numpy as np
from tqdm import tqdm
from training import get_env
from create_gridworld_options import create_primitive_options, create_eigenoptions


def eval_loop_fixed_options(agent, env_name, n_eigenoptions, max_steps, seed) -> (
        np.ndarray):
    """Training an agent to select fixed options."""

    np.random.seed(seed)
    env = get_env(env_name=env_name, _max_steps=max_steps)

    # Create four primitive options for each of the base actions
    options = create_primitive_options(env)

    if n_eigenoptions > 0:
        eigenoptions = create_eigenoptions(env, n_eigenoptions, agent.discount)
        options.update(eigenoptions)

    _options = list(options.values())
    option_names = list(options.keys())
    assert agent.n_actions == len(_options), ("Number of agent actions must match "
                                              "the number of options.")

    frames = []

    state_idx, info = env.reset()
    done = False
    total_steps = 0

    while not done and total_steps < env._max_steps:

        frames.append(env.render_frame())

        option_idx = agent.choose_action(state_idx)
        option = _options[option_idx]
        action = option.policy_selection(state_idx)

        next_state_idx, _, done, _, _ = env.step(action)
        total_steps += 1

        # Run the option until it terminates
        while (not option.termination_condition(next_state_idx) and not done and
               total_steps <= env._max_steps):
            frames.append(env.render_frame())
            action = option.policy_selection(next_state_idx)
            next_state_idx, _, done, _, _ = env.step(
                action)
            total_steps += 1

        state_idx = next_state_idx

    return np.array(frames).astype(np.uint8)
