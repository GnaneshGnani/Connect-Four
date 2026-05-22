# Connect Four RL Agent

Self-play reinforcement learning agent for Connect Four, built with PyTorch and a Gymnasium-compatible environment.

The project focuses on the mechanics that make a board-game RL system usable for experimentation: legal-action masking, replay memory, target networks, Double DQN updates, checkpointing, ClearML logging, metric plotting, and an interactive play mode.

## What It Does

- Defines a `ConnectFourEnv` environment with a 6 x 7 board, discrete actions, valid-move checks, win/draw detection, heuristic rewards, and text/Pygame rendering.
- Trains a DQN or Double DQN agent through self-play.
- Supports CNN and feed-forward Q-network architectures.
- Uses replay memory, epsilon-greedy exploration, target-network updates, legal-action masking, TD-error tracking, and model checkpoints.
- Logs training metrics locally through plots and optionally to ClearML.
- Lets a human play against a saved agent through a terminal or Pygame UI.
- Provides a Docker workflow for reproducible local training/play sessions.

## Repository Layout

```text
main.py          Training and play entry point
dqn.py           DQNAgent plus CNN/FFN model definitions
environment.py   Gymnasium-style Connect Four environment
utils.py         Plotting and helper utilities
Dockerfile       Containerized training/play workflow
requirements.txt Python dependencies
```

## Setup

Create an environment with Python 3.10+ and install dependencies:

```bash
pip install -r requirements.txt
```

If your environment does not have ClearML configured, use `--no-clearml` in the commands below.

## Train

Train a CNN Double DQN agent for a small local smoke run:

```bash
python main.py \
  --episodes 100 \
  --save-frequency 50 \
  --agent-path models/agent.pth \
  --no-play \
  --no-clearml
```

Train the default longer run:

```bash
python main.py \
  --episodes 50000 \
  --save-frequency 1000 \
  --agent-path models/agent.pth \
  --no-play
```

Double DQN is enabled by default. Add `--no-double-dqn` to use the standard DQN target update.

## Play Against a Saved Agent

```bash
python main.py \
  --no-train \
  --agent-path models/agent.pth \
  --no-clearml
```

The script first tries to use the Pygame UI. If a GUI is unavailable, it falls back to text mode.

## Docker

Build the image:

```bash
docker build -t connect-four-rl .
```

Run a training session while preserving models and plots on the host:

```bash
mkdir -p models plots

docker run --rm \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/plots":/app/plots \
  connect-four-rl \
  python main.py --episodes 1000 --save-frequency 100 --no-play --no-clearml
```

On Linux, a GUI play session can be run by sharing the X11 socket:

```bash
xhost +local:docker

docker run --rm -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY="$DISPLAY" \
  -v "$(pwd)/models":/app/models \
  connect-four-rl \
  python main.py --no-train --agent-path models/agent.pth --no-clearml
```

## Key Arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `--episodes` | `50000` | Number of self-play training episodes |
| `--save-frequency` | `1000` | Checkpoint interval |
| `--agent-path` | `models/agent.pth` | Model save/load path |
| `--network-model` | `CNN` | Q-network architecture: `CNN` or `FFN` |
| `--learning-rate` | `1e-4` | Optimizer learning rate |
| `--memory-size` | `50000` | Replay-buffer capacity |
| `--batch-size` | `512` | Replay batch size |
| `--epsilon` | `1.0` | Initial exploration rate |
| `--epsilon-decay` | `0.99997` | Per-episode epsilon decay |
| `--epsilon-min` | `0.1` | Minimum exploration rate |
| `--gamma` | `0.9` | Discount factor |
| `--target-update-freq` | `500` | Target-network update interval |
| `--no-double-dqn` | off | Disable Double DQN |
| `--no-train` | off | Load a saved agent and skip training |
| `--no-play` | off | Skip interactive play after training |
| `--no-clearml` | off | Disable ClearML logging |

## Evidence Status

This repository contains the environment, agent implementations, training loop, checkpointing, plotting hooks, and Docker workflow. It does not currently include a checked-in benchmark report or win-rate evaluation artifact, so README/resume claims should describe implementation scope rather than model strength.
