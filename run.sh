#!/bin/bash

source .venv/bin/activate

set -x  # for printing commands

current_datetime=$(date '+%Y-%m-%d-%H-%M')

for seed in {1..5}
do
  # Save a video of the agent on the first run
  python3 run_experiment.py \
      --env_name="four_rooms_alt_with_vases" \
      --n_episodes=500 \
      --max_steps=100 \
      --seed=$seed \
      --agent_class="QLearningAgent" \
      --n_eigenoptions=3 \
      --diffusion="None" \
      --discount=0.9 \
      --mode=save \
      --log_dir=./logs/$current_datetime \
      --suffix="" \
      --wandb=True \
      --wandb_project=rl-laplacian\
      --eval_video=True
done
