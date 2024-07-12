#!/bin/bash

source .venv/bin/activate

set -x  # for printing commands

current_datetime=$(date '+%Y-%m-%d-%H-%M')

for seed in {1..10}
do
  # Save a video of the agent on the first run
  python3 run_experiment.py \
      --agent_class="QLearningAgent" \
      --env_name="four_rooms" \
      --n_episodes=500 \
      --max_steps=100 \
      --seed=$seed \
      --mode=save \
      --log_dir=./logs/$current_datetime \
      --suffix=no_options \
      --wandb=True \
      --wandb_project=side-effects-debug\
      --eval_video=False
done
