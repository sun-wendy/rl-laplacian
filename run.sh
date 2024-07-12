#!/bin/bash

source .venv/bin/activate

set -x  # for printing commands

current_datetime=$(date '+%Y-%m-%d-%H-%M')

for seed in {1..1}
do
  # Save a video of the agent on the first run
  python3 run_experiment.py \
      --env_name="four_rooms" \
      --n_episodes=500 \
      --max_steps=100 \
      --seed=$seed \
      --agent_class="QLearningFixedOptionsAgent" \
      --n_eigenoptions=8 \
      --diffusion="normalised" \
      --discount=0.9 \
      --mode=save \
      --log_dir=./logs/$current_datetime \
      --suffix= \
      --wandb=False \
      --wandb_project=side-effects-debug\
      --eval_video=False
done
