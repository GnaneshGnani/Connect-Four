import os
import time
import argparse
import warnings
import numpy as np
from tqdm import tqdm

from dqn import DQNAgent
from utils import plot_metrics
from environment import ConnectFourEnv

from clearml import Task, Model

warnings.filterwarnings("ignore", category = FutureWarning, message = ".*weights_only.*")

PROJECT_NAME = "Connect Four"

def train_agent(agent, episodes = 10000, save_path = "models/agent.pth", save_frequency = None, task = None, quick_step_reward = False):
    logger = task.get_logger() if task else None
    save_frequency = episodes // 10 if save_frequency is None else save_frequency
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok = True)

    env = ConnectFourEnv(render_mode = None, quick_step_reward = quick_step_reward)

    metrics = {
        "ep_rewards": [],
        "ep_lengths": [],
        "avg_losses": [],
        "epsilon_history": [],
        "td_error_history": [],
        "action_distributions": []
    }

    last_checkpoint_ep = 0
    pbar = tqdm(range(episodes), desc = "Training Progress")
    
    for episode in pbar:
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_losses = []

        while not done:
            current_player = env.current_player
            valid_actions = env.get_valid_actions()

            # Agent always sees the board from its own perspective
            action = agent.get_action(state * current_player, valid_actions, train = True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.remember(
                state * current_player,
                action,
                reward,
                # Next state is from the opponent's perspective
                next_state * env.current_player if not done else None,
                done
            )

            state = next_state
            step += 1
            agent.train_step_counter += 1

            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
                    
                    if agent.td_errors:
                        last_td_error = agent.td_errors[-1]
                        metrics["td_error_history"].append(last_td_error)
                        if logger:
                            logger.report_scalar(
                                title = "Per-Step Metrics", series = "TD Error",
                                value = last_td_error, iteration = agent.train_step_counter
                            )
                            logger.report_scalar(
                                title = "Per-Step Metrics", series = "Batch Loss",
                                value = loss, iteration = agent.train_step_counter
                            )

        # --- End of Episode ---
        
        # Epsilon Decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *=  agent.epsilon_decay

        # Log per-episode metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        metrics["ep_rewards"].append(total_reward)
        metrics["ep_lengths"].append(step)
        metrics["avg_losses"].append(avg_loss)
        metrics["epsilon_history"].append(agent.epsilon)
        metrics["action_distributions"].append(agent.action_counts.copy())
        
        if logger:
            logger.report_scalar(title = "Per-Episode Metrics", series = "Avg Reward", value = total_reward, iteration = episode)
            logger.report_scalar(title = "Per-Episode Metrics", series = "Avg Loss", value = avg_loss, iteration = episode)
            logger.report_scalar(title = "Per-Episode Metrics", series = "Episode Length", value = step, iteration = episode)
            logger.report_scalar(title = "State", series = "Epsilon", value = agent.epsilon, iteration = episode)

        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "Reward": f"{total_reward:.2f}",
            "Epsilon": f"{agent.epsilon:.3f}",
            "Last Checkpoint": f"Ep {last_checkpoint_ep}"
        })

        # Saving Periodically
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = save_path.replace(".pth", f"_checkpoint_ep{episode + 1}.pth")
            agent.save(checkpoint_path, include_memory = False)
            last_checkpoint_ep = episode + 1

            if task:
                task.upload_artifact(
                    name = f'checkpoint_ep{episode + 1}',
                    artifact_object = checkpoint_path
                )

    print("Training finished!")
    _finalize_training(
        agent = agent,
        task = task,
        metrics = metrics,
        episodes = episodes,
        final_save_path = save_path,
        model_tags = [agent.model_type, "final"]
    )

    # Reset agent stats for potential play session
    agent.random_moves = 0
    agent.action_counts = [0] * 7
    return agent


def _finalize_training(agent, task, metrics, episodes, final_save_path, model_tags):
    print(f"Saving final model to: {final_save_path}")
    agent.save(final_save_path, include_memory = False)

    if task and Model:
        print("Registering final model with ClearML...")
        try:
            model = Model(
                project = task.project,
                name = f"{task.name} Final Model"
            )
            model.update_weights(weights_filename = final_save_path)
            model.connect(task)
            model.add_tags(model_tags)
            model.publish()
            print(f"Final model registered in ClearML with ID: {model.id}")
        except Exception as e:
            print(f"Could not register ClearML model: {e}")

    plot_metrics(
        episodes = episodes,
        rewards = metrics["ep_rewards"],
        lengths = metrics["ep_lengths"],
        losses = metrics["avg_losses"],
        epsilon_history = metrics.get("epsilon_history"),
        td_errors = metrics.get("td_error_history"),
        action_distributions = metrics.get("action_distributions"),
        task = task
    )


def play_vs_agent(agent):
    print(f"\n{' = '*60}")
    print("Starting Game vs. Trained Agent")
    print(f"{' = '*60}\n")
    print("You are Player -1 (Yellow). The Agent is Player 1 (Red).")
    print("Enter column numbers (0-6) in the terminal to play.")
    print(f"{' = '*60}\n")

    # Disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        env = ConnectFourEnv(render_mode = "human")
        if env.pygame is None:
            raise Exception("Pygame not initialized")
    except Exception:
        print("GUI not available. Falling back to text mode.")
        env = ConnectFourEnv(render_mode = "text")

    games_played = 0
    player_wins = 0
    agent_wins = 0
    draws = 0

    while True:
        state = env.reset()
        done = False
        env.render()

        while not done:
            if env.current_player == 1:
                # Agent's turn
                print("\nAgent's turn...")
                valid_actions = env.get_valid_actions()
                action = agent.get_action(state * env.current_player, valid_actions, train = False)
                state, _, done, info = env.step(action)
                print(f"Agent plays column {action}")
                env.render()
                if env.render_mode == "human":
                    time.sleep(0.5)
            else:
                # Human's turn
                try:
                    action = None
                    while True:
                        print(f"\nYour turn (Player Yellow)")
                        print(f"Valid moves: {env.get_valid_actions()}")
                        col_str = input("Enter column (0-6): ")
                        
                        if not col_str.isdigit():
                            print("Invalid input! Please enter a number.")
                            continue
                            
                        col = int(col_str)
                        if env.is_valid_action(col):
                            action = col
                            break
                        else:
                            print("Invalid move! That column is full or out of bounds.")

                    state, _, done, info = env.step(action)
                    print(f"You play column {action}")
                    env.render()

                except ValueError:
                    print("Invalid input! Enter a number between 0 and 6.")
                except Exception as e:
                    print(f"Error: {e}")
        
        # --- Game Over ---
        games_played += 1
        print("\n" + " = " * 60)
        print("       GAME OVER!")
        print(" = " * 60)

        winner = info.get("winner")
        if winner == 1:
            print("Winner: Agent (Player 1 - Red)")
            agent_wins += 1
        elif winner == -1:
            print("Winner: You! (Player -1 - Yellow)")
            player_wins += 1
        else:
            print("It's a Draw!")
            draws += 1

        print(f"\nSession Stats: Games = {games_played}, You = {player_wins}, Agent = {agent_wins}, Draws = {draws}")
        
        choice = input("\nPlay again? (y/n): ").strip().lower()
        if choice != "y":
            break

    # Cleanup
    agent.epsilon = original_epsilon
    env.close()
    print("\n" + " = "*60)
    print("Thanks for playing!")
    print(f"Final Score: You {player_wins} - Agent {agent_wins} (Draws: {draws})")
    print(" = "*60 + "\n")


def get_args():
    parser = argparse.ArgumentParser(
        description = "Connect Four DQN Agent - Train and Play",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Env ---
    parser.add_argument("--quick-step-reward", action = "store_false",
                        help = "Allow Quick Step Negative Rewards")

    # --- Execution ---
    parser.add_argument("--episodes", type = int, default = 50000,
                        help = "Number of episodes to train")
    parser.add_argument("--save-frequency", type = int, default = 1000,
                        help = "Save checkpoint every N episodes")
    parser.add_argument("--agent-path", type = str, default = "models/agent.pth",
                        help = "Path to save/load agent")
    parser.add_argument("--no-play", action = "store_true",
                        help = "Skip playing after training")
    parser.add_argument("--no-train", action = "store_true",
                        help = "Skip training and load agent from --agent-path")
    parser.add_argument("--no-clearml", action = "store_true",
                        help = "Disable ClearML logging")

    # --- Model Hyperparameters ---
    parser.add_argument("--network-model", type = str, default = "CNN", choices = ["CNN", "FFN"],
                        help = "Model architecture")
    parser.add_argument("--learning-rate", type = float, default = 1e-4,
                        help = "Learning rate")
    parser.add_argument("--memory-size", type = int, default = 50000,
                        help = "Replay memory size")
    parser.add_argument("--batch-size", type = int, default = 512,
                        help = "Training batch size")
    parser.add_argument("--epsilon", type = float, default = 1.0,
                        help = "Initial exploration rate")
    parser.add_argument("--epsilon-decay", type = float, default = 0.99997,
                        help = "Epsilon decay rate")
    parser.add_argument("--epsilon-min", type = float, default = 0.1,
                        help = "Minimum epsilon value")
    parser.add_argument("--gamma", type = float, default = 0.9,
                        help = "Discount factor")
    parser.add_argument("--target-update-freq", type = int, default = 500,
                        help = "Steps between target network updates")
    parser.add_argument("--no-double-dqn", action = "store_true",
                        help = "Disable Double DQN (it is enabled by default)")

    args = parser.parse_args()
    
    # Post-process args
    args.use_double_dqn = not args.no_double_dqn
    
    return args


def setup_clearml(args):
    if args.no_train or args.no_clearml or Task is None:
        if args.no_clearml:
            print("ClearML logging disabled by user.")
        elif Task is None:
            print("ClearML not found. Skipping logging. (Install with: pip install clearml)")
        return None
    
    try:
        task = Task.init(
            project_name = PROJECT_NAME,
            task_name = f"{args.network_model} Training - {time.strftime('%Y%m%d-%H%M%S')}"
        )
        task.connect(vars(args), name = 'Script Arguments')
        print(f"ClearML Task initialized: {task.get_output_log_web_page()}")
        return task
    except Exception as e:
        print(f"Error initializing ClearML: {e}")
        return None


def setup_agent(args):
    agent_hyperparams = {
        "model": args.network_model,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "memory_size": args.memory_size,
        "batch_size": args.batch_size,
        "target_update_frequency": args.target_update_freq,
        "use_double_dqn": args.use_double_dqn,
    }

    # Try loading existing agent if path exists
    agent_path = args.agent_path
    if os.path.exists(agent_path):
        print(f"Found existing agent at: {agent_path}")
        if args.no_train:
            print("Loading agent for play mode...")
            try:
                agent = DQNAgent.load(agent_path, load_memory = False, train = False)
                return agent, agent.get_hyperparameters()
            except Exception as e:
                print(f"Error loading agent: {e}. Exiting.")
                return None, None
        else:
            print("Loading agent to continue training...")
            try:
                agent = DQNAgent.load(agent_path, load_memory = True, train = True)
                print("Agent loaded successfully.")
                return agent, agent.get_hyperparameters()
            except Exception as e:
                print(f"Error loading agent: {e}. Creating a new agent instead.")
    
    # Create a new agent if no_train is False and no agent was loaded
    if args.no_train:
        print(f"Error: --no-train specified but no agent found at {agent_path}")
        return None, None

    print(f"Creating new DQN agent ({args.network_model})...")
    agent = DQNAgent(**agent_hyperparams)
    return agent, agent_hyperparams


def main(args):
    print(f"\n{' = '*60}")
    print("      Connect Four DQN Agent")
    print(f"{' = '*60}")
    
    task = setup_clearml(args)
    
    agent, hyperparams = setup_agent(args)
    
    if agent is None:
        return  # Agent setup failed

    if task and hyperparams:
        task.connect(hyperparams, name = 'Agent Hyperparameters')

    if not args.no_train:
        agent = train_agent(
            agent = agent,
            episodes = args.episodes,
            save_path = args.agent_path,
            save_frequency = args.save_frequency,
            task = task,
            quick_step_reward = args.quick_step_reward
        )
    else:
        print("Skipping training (--no-train flag set)")

    if not args.no_play:
        play_vs_agent(agent)
    else:
        print("Skipping play mode (--no-play flag set)")

    if task:
        task.close()
        
    print("\nDone.")


if __name__ == "__main__":
    args = get_args()
    main(args)