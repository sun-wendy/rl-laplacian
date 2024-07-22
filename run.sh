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
      --n_eigenoptions=0 \
      --diffusion="None" \
      --discount=0.9 \
      --mode=none \
      --log_dir=./logs/$current_datetime \
      --suffix="reach_regul_1,6_penalty_strength_10_init_prob_0" \
      --penalty_strength=1 \
      --wandb=True \
      --wandb_project=learn-safe-options\
      --eval_video=True
done
