"""Run a Q-learning agent with a side effects penalty."""

import argparse
import training
from agent import QLearningAgent, QLearningFixedOptionsAgent


class Args:
    """Handles the command-line arguments"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Command Line Arguments")
        # Agent settings
        self.parser.add_argument('--agent_class', type=str2agentclass,
                                 default='QLearningAgent', choices= [QLearningFixedOptionsAgent, QLearningAgent]),
        self.parser.add_argument('--learning_rate', type=float,
                                 default=0.1, help='Learning rate for training')
        self.parser.add_argument('--discount', type=float, default=0.9,
                                 help='Discount factor for rewards.')
        self.parser.add_argument('--n_episodes', type=int, default=200,
                                 help='Number of episodes.')
        self.parser.add_argument('--seed', type=int, default=1,
                                 help='Random seed.')
        self.parser.add_argument('--anneal', type=str2bool, default='True',
                                 choices=[True, False], help='Whether to anneal exploration')
        self.parser.add_argument('--n_eigenoptions', type=int, default=0,
                                 help='Number of eigenoptions to use')
        # Environment settings
        self.parser.add_argument('--env_name', type=str,
                                 default='one_room',
                                 choices=['one_room', 'four_rooms', 'i_maze',
                                          'two_rooms', 'three_rooms', 'hard_maze'],
                                 help='Environment name.')
        self.parser.add_argument('--diffusion', type=str, default='normalised',
                                 choices=['None', 'normalised', 'random_walk'],
                                 help='Diffusion type for environment.')
        self.parser.add_argument('--max_steps', type=int, default=5_000,
                                 help='Maximum number of steps per episode.')
        # Settings for outputting results
        self.parser.add_argument('--mode', type=str, default='none',
                                 choices=['print', 'save', 'none'],
                                 help='Print results or save to file.')
        self.parser.add_argument('--log_dir', type=str, default='',
                                 help='Logging directory.')
        self.parser.add_argument('--suffix', type=str, default='',
                                 help='Filename suffix.')
        self.parser.add_argument('--wandb', type=str2bool, default='False',
                                 choices=[True, False],
                                 help='Save stats to wandb.')
        self.parser.add_argument('--wandb_project', type=str, default='',
                                 help='wandb project name.')
        self.parser.add_argument('--eval_video', type=str2bool, default='False',
                                 choices=[True, False], help='Save video after training')

    def parse(self):
        """Parse command line arguments and assign them as attributes."""
        args = self.parser.parse_args()

        for key, value in vars(args).items():
            setattr(self, key, value)

        if args.diffusion == 'None':
            args.diffusion = None

        if args.mode == 'save':
            assert args.log_dir != '', 'Must provide log_dir when using save'

        if args.wandb:
            assert args.wandb_project != '', 'Must provide wandb project name'


def str2bool(v):
    """Turns command line arguments into boolean values"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2agentclass(v):
    """Turns command line arguments into boolean values"""
    if v == "QLearningAgent":
        return QLearningAgent
    elif v == "QLearningFixedOptionsAgent":
        return QLearningFixedOptionsAgent
    else:
        raise argparse.ArgumentTypeError(f'Unrecognized agent class: {v}.')


def run_experiment(args):
    """Run agent and save or print the results."""

    stats, _agent = training.run_agent(
        agent_class=args.agent_class,
        learning_rate=args.learning_rate,
        discount=args.discount,
        n_episodes=args.n_episodes,
        anneal=args.anneal,
        seed=args.seed,
        env_name=args.env_name,
        diffusion=args.diffusion,
        max_steps=args.max_steps,
        n_eigenoptions=args.n_eigenoptions)

    # Save a video of the agent in the environment
    frames = []
    if args.eval_video:
        raise NotImplementedError

    # This python script is run :n: times, with the exact same arguments except for
    # the seeds. To visualize the mean and variance across runs, we use the same
    # group for each run, and distinguish the runs within a group by their seeds
    if args.agent_class == QLearningAgent:
        group_name = f'{args.env_name}_{args.suffix}_no_options'
    elif args.agent_class == QLearningFixedOptionsAgent:
        group_name = (f'{args.env_name}_{args.suffix}_{args.n_eigenoptions}_eigenoptions')


    # Print stats
    if args.mode == 'print':
        print('Stats in the last 10 steps:')
        for k, v in stats.items():
            print(f'{k}: {v[-10:]}')

    # Save stats, agent and eval_video to log file
    elif args.mode == 'save':
        import os
        import pickle

        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        stats_file = os.path.join(args.log_dir, f'{group_name}_seed_'
                                                f'{args.seed}_stats.pkl')
        with open(stats_file, "wb") as f:
            pickle.dump(stats, f)

        agent_file = os.path.join(args.log_dir, f'{group_name}_seed_'
                                                f'{args.seed}_agent.pkl')
        with open(agent_file, "wb") as f:
            pickle.dump(_agent, f)

        if args.eval_video:
            import cv2

            fps = 4
            width = frames.shape[1]
            height = frames.shape[2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_file = os.path.join(args.log_dir, f'{group_name}_seed'
                                                    f'_{args.seed}_eval_video.mp4')
            video = cv2.VideoWriter(video_file, fourcc, float(fps), (width, height))

            for i in range(frames.shape[0]):
                video.write(frames[i])

            video.release()

    # Upload stats to wandb
    if args.wandb:
        import wandb
        import numpy as np

        run = wandb.init(project=args.wandb_project,
                         name=f'{group_name}_seed_{args.seed}',
                         group=group_name)

        stats['max_steps'] = np.array([args.max_steps] * args.n_episodes)
        stats['seed'] = np.array([args.seed] * args.n_episodes)
        stats['episode'] = np.arange(1, args.n_episodes + 1)

        # Note: this only works when 'stats' are 1-dimensional numpy arrays
        for episode in range(args.n_episodes):
            run.log({stat_name: value[episode] for stat_name, value in
                     stats.items()})

        # Save the eval_video to wandb
        if args.eval_video:
            # put RGB channels first
            frames = frames.transpose(0, 3, 2, 1)
            wandb.log({"eval_video": wandb.Video(frames, fps=4)})


if __name__ == '__main__':
    args = Args()
    args.parse()
    run_experiment(args)