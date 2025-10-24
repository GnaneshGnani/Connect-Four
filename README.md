# MSML642 Homework 4: Connect Four with Reinforcement Learning

This project implements a Reinforcement Learning agent to master the game of Connect Four through self-play, as required for MSML642 Homework 4.

## Overview

The goal is to train an agent that can learn to play Connect Four effectively. The agent is trained by playing against itself repeatedly, learning from its successes and failures.

The project uses a **Deep Q-Network (DQN)** architecture. Specifically, it implements **Double DQN** by default to stabilize training and improve performance. The agent's "brain" is a Convolutional Neural Network (CNN) that processes the game board as an image, or a simple Feed-Forward Network (FFN).

## File Structure

* `main.py`: The main executable script. Handles agent training, model saving/loading, and provides an interface to play against the trained agent.
* `agent.py`: Contains the `DQNAgent` class, which manages the Q-learning logic, replay memory, and target networks. It also defines the `CNN` and `FFN` model architectures using PyTorch.
* `environment.py`: Defines the `ConnectFourEnv` class, a Gymnasium-compatible environment that manages game state, actions, and rewards. It includes logic for win-checking and reward shaping.
* `utils.py`: A helper module for generating `matplotlib` plots of training metrics like reward, loss, and episode length.
* `Dockerfile`: A file to build and run the project within a Docker container, as required by the assignment.
* `README.md`: This file.

## Docker Workflow

### Step 1: Build the Docker Image

Before building, **you must have a trained `agent.pth` file** (and its `agent_config.json`) in a local `models` directory. The `Dockerfile` will copy these files directly into the image.

From the project's root directory, run:
```bash
docker build -t connect-four-rl .
```

### Step 2: Play Against the Agent (GUI on Linux)

The container runs in GUI mode by default. To make this work, we need to give the container permission to send drawing instructions to your host's monitor (your X server).

**1. Give Permission (On Your Host):**
Run this command in your host terminal. It allows local Docker containers to connect to your display.
```bash
xhost +local:docker
```

**2. Run the Container (On Your Host):**
Now, run the Docker container, connecting it to your host's display system:
```bash
docker run --rm -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  connect-four-rl
```

The Pygame window should now appear on your desktop!

#### How This Works
You are right, we are using your host's monitor directly!
* Your Linux desktop is already running an "X server," which is the program that controls your monitor and draws all your windows.
* The Pygame app inside Docker is an "X client." It needs to send instructions (like "draw a red circle") to your X server.
* The `xhost` command gives **permission**.
* The `-v /tmp/.X11-unix...` command shares the **connection file** (the "socket") so the app can talk to the server.
* The `-e DISPLAY=$DISPLAY` command tells the app the **address** of that connection.

### Step 3: Train a New Agent (Saving Your Work)

To train a new model and **ensure your models and plots are not lost**, you must use **volumes**. Volumes connect directories on your host machine to directories inside the container.

**1. Create Directories (On Your Host):**
Make sure you have local directories to store the results.
```bash
mkdir -p models plots
```

**2. Run Training with Volumes:**
This command overrides the container's default "play" command and runs training instead.
```bash
docker run --rm \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/plots":/app/plots \
  connect-four-rl python main.py --episodes 50000 --no-play
```

#### How This Works
* `-v "$(pwd)/models":/app/models`: This links your **current** directory's `models` subfolder to the `/app/models` folder *inside* the container.
* `-v "$(pwd)/plots":/app/plots`: This does the same for the `plots` folder.
* `python main.py ...`: This is the new command the container will run.
* When `agent.save()` is called inside the container, it writes to `/app/models/agent.pth`, which is immediately saved to **your** `models/agent.pth` file. Your progress is safe.

**WARNING:** If you run the training command *without* the `-v` volume flags, any new models and plots **will be permanently lost** when the container stops.

## Command-Line Arguments

You can pass any of these arguments to `main.py`, either locally or when using `docker run`.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--episodes` | int | 50000 | Number of episodes to train. |
| `--save-frequency` | int | 1000 | Save checkpoint every N episodes. |
| `--agent-path` | str | `models/agent.pth` | Path to save/load agent. |
| `--no-play` | bool | False | Skip playing after training. |
| `--no-train` | bool | False | Skip training (loads agent and plays). |
| `--force-train` | bool | False | Force training from scratch, ignoring existing models. |
| `--model` | str | `CNN` | Model architecture. Choices: `CNN`, `FFN`. |
| `--learning-rate` | float | 1e-4 | Learning rate for the optimizer. |
| `--memory-size` | int | 50000 | Replay memory size. |
| `--batch-size` | int | 512 | Training batch size. |
| `--epsilon` | float | 1.0 | Initial exploration rate (epsilon). |
| `--epsilon-decay` | float | 0.99997 | Epsilon decay rate. |
| `--epsilon-min` | float | 0.1 | Minimum epsilon value. |
| `--gamma` | float | 0.9 | Discount factor for future rewards. |
| `--target-update-freq` | int | 500 | Steps between target network updates. |
| `--use-double-dqn` | bool | True | Use Double DQN. |
| `--n-step` | int | 3 | Number of steps for n-step returns. |