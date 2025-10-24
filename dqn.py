import os
import json
import torch
import random
import numpy as np
from datetime import datetime
from collections import deque

class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()

        self.layer_1 = torch.nn.Linear(6 * 7, 256)
        self.ln_1 = torch.nn.LayerNorm(256)
        
        self.layer_2 = torch.nn.Linear(256, 128)
        self.ln_2 = torch.nn.LayerNorm(128)

        self.layer_3 = torch.nn.Linear(128, 64)
        self.ln_3 = torch.nn.LayerNorm(64)

        self.layer_4 = torch.nn.Linear(64, 7)

    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.ln_1(self.layer_1(x)))
        x = self.relu(self.ln_2(self.layer_2(x)))
        x = self.relu(self.ln_3(self.layer_3(x)))
        x = self.layer_4(x)
        return x
    
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()
        
        self.cnn_1 = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 4)
        self.cnn_2 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.cnn_3 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)

        self.flatten = torch.nn.Flatten()
        self.layer_1 = torch.nn.Linear(256 * 4 * 3, 256)
        self.layer_2 = torch.nn.Linear(256, 7)

    def forward(self, x):
        x = x.unsqueeze(1) # Adding Channel to make it look like an image

        x = self.relu(self.cnn_1(x))
        x = self.relu(self.cnn_2(x))
        x = self.relu(self.cnn_3(x))
        x = self.flatten(x)
        x = self.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class DQNAgent:
    def __init__(self, model = "CNN", epsilon = 1, epsilon_decay = 0.99999, epsilon_min = 0.1, 
                 gamma = 0.99, learning_rate = 1e-2, memory_size = 1000, batch_size = 128,
                 target_update_frequency = 100, use_double_dqn = True):
        self.action_space = [i for i in range(7)]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_type = model
        if model == "CNN":
            self.main_model = CNN()
            self.target_model = CNN()

        else:
            self.main_model = FFN()
            self.target_model = FFN()

        self.update_target_network()
        self.main_model.to(self.device)
        self.target_model.to(self.device)

        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_model.parameters(), lr = learning_rate)

        self.memory_size = memory_size
        self.memory = deque(maxlen = memory_size)

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.target_update_frequency = target_update_frequency
        self.batch_size = batch_size
        self.random_moves = 0
        self.use_double_dqn = use_double_dqn
        self.train_step_counter = 0

        # Tracking
        self.action_counts = [0] * 7  # Action Distribution
        self.td_errors = []  # TD errors
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.main_model.state_dict())

    @torch.no_grad()    
    def get_action(self, state, valid_actions, train = False):
        if train and np.random.uniform(0, 1) <= self.epsilon:
            self.random_moves += 1
            action = np.random.choice(valid_actions)
            self.action_counts[action] += 1  # Add this line
            return action
        
        state = torch.tensor(np.array(state), dtype = torch.float32, device = self.device).unsqueeze(0) # Neural Network expects a batch
        invalid_actions = [action for action in self.action_space if action not in valid_actions]
        
        self.main_model.eval()
        q_values = self.main_model(state)[0]
        q_values[invalid_actions] = -float("inf")

        action = torch.argmax(q_values).item()
        self.action_counts[action] += 1

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def _get_legal_actions_mask(self, states):
        if states.ndim == 2:
            states = states.unsqueeze(0)
        
        top_row = states[:, 0, :]
        legal_mask = (top_row == 0)

        return legal_mask

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.main_model.train()

        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.tensor(np.array(states), dtype = torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype = torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype = torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype = torch.bool).to(self.device)

        non_final_next_states = np.array([next_state for next_state in next_states if next_state is not None])
        non_final_next_states = torch.tensor(non_final_next_states, dtype = torch.float32).to(self.device)

        # Hardcoding Invalid Actions to -1
        legal_mask = self._get_legal_actions_mask(states)
        current_q_values = self.main_model(states)
        current_q_values[~legal_mask] = -1
        
        with torch.no_grad():
            next_q_values = torch.zeros(len(dones)).to(self.device)
            if non_final_next_states.shape[0] > 0:
                # Hardcoding Invalid Actions to -1
                legal_mask = self._get_legal_actions_mask(non_final_next_states)
                if self.use_double_dqn:
                    # Double DQN: use main network to select actions, target network to evaluate
                    next_state_q_values = self.main_model(non_final_next_states)
                    next_state_q_values[~legal_mask] = -1
                    next_state_actions = next_state_q_values.argmax(dim = 1)

                    next_state_target_q_values = self.target_model(non_final_next_states)
                    next_state_target_q_values[~legal_mask] = -1
                    next_q_values[~dones] = next_state_target_q_values[torch.arange(non_final_next_states.shape[0]), next_state_actions]
                else:
                    # Standard DQN: use target network for both selection and evaluation
                    next_state_target_q_values = self.target_model(non_final_next_states)
                    next_state_target_q_values[~legal_mask] = -1
                    next_q_values[~dones] = next_state_target_q_values.max(dim = 1)[0]

        # Subtracting because the opponent"s gain is out loss
        # The next state is the move made by opponent
        expected_q_values = current_q_values.clone()
        expected_q_values[torch.arange(len(actions)), actions] = rewards - (self.gamma * next_q_values)

        # Hardcoding Non-Winning Actions to 0
        # dones_q_values = torch.zeros(dones.shape[0], expected_q_values.shape[1], dtype = torch.float32, device = self.device)
        # dones_q_values[dones, actions[dones]] = 1
        # expected_q_values[dones] = dones_q_values[dones]

        # Tracking
        td_error = torch.abs(
            expected_q_values[torch.arange(len(actions)), actions] - 
            current_q_values[torch.arange(len(actions)), actions]
        )
        self.td_errors.append(td_error.mean().item()) 

        # Back Propagation
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, expected_q_values)
        loss.backward()
        self.optimizer.step()

        # # Epsilon decay
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        
        # Update Target Network
        if self.train_step_counter % self.target_update_frequency == 0:
            self.update_target_network()

        return loss.item()

    def save(self, filepath = None, include_memory = False):
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"agent_checkpoint_{timestamp}.pth"
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok = True)
        
        checkpoint = {
            "model_type": self.model_type,
            "optimizer_state": self.optimizer.state_dict(),
            "main_model_state": self.main_model.state_dict(),
            "target_model_state": self.target_model.state_dict(),

            "hyperparameters": {
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "batch_size": self.batch_size,
                "memory_size": self.memory_size,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "learning_rate": self.learning_rate,
                "use_double_dqn": self.use_double_dqn,
                "target_update_frequency": self.target_update_frequency
            },

            "training_state": {
                "random_moves": self.random_moves,
                "action_counts": self.action_counts,  
                "train_step_counter": self.train_step_counter,
                "td_errors": self.td_errors[-1000:] if self.td_errors else []
            }
        }
        
        if include_memory:
            checkpoint["memory"] = list(self.memory)
        
        torch.save(checkpoint, filepath)
        
        # Save a Human-Readable Config File
        config_path = filepath.replace(".pth", "_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_type": self.model_type,
                "hyperparameters": checkpoint["hyperparameters"],
                "training_state": checkpoint["training_state"],
                "memory_included": include_memory,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent = 4)
                
        return filepath
    
    @classmethod
    def load(cls, filepath, load_memory = False, train = False):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        print(f"Loading agent from: {filepath}")
        
        checkpoint = torch.load(filepath, map_location = "cpu", weights_only = False)

        hyperparams = checkpoint["hyperparameters"]
        
        agent = cls(
            model = checkpoint["model_type"],
            epsilon = hyperparams["epsilon"] if train else 0,
            epsilon_decay = hyperparams["epsilon_decay"],
            epsilon_min = hyperparams["epsilon_min"],
            gamma = hyperparams["gamma"],
            learning_rate = hyperparams["learning_rate"],
            memory_size = hyperparams["memory_size"],
            batch_size = hyperparams["batch_size"],
            target_update_frequency = hyperparams["target_update_frequency"],
            use_double_dqn = hyperparams.get("use_double_dqn", True),
        )
        
        # Load model states
        agent.main_model.load_state_dict(checkpoint["main_model_state"])
        agent.target_model.load_state_dict(checkpoint["target_model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Move models to appropriate device
        agent.main_model.to(agent.device)
        agent.target_model.to(agent.device)

        if train:
            agent.main_model.train()
            agent.target_model.train()
        else:
            agent.main_model.eval()
            agent.target_model.eval()
        
        # Restore training state
        training_state = checkpoint["training_state"]
        agent.train_step_counter = training_state["train_step_counter"]
        agent.random_moves = training_state["random_moves"]
        agent.action_counts = training_state.get("action_counts", [0] * 7)
        agent.td_errors = training_state.get("td_errors", [])
        
        # Load replay memory if available and requested
        if load_memory and "memory" in checkpoint:
            agent.memory = deque(checkpoint["memory"], maxlen = agent.memory_size)
            print(f"Loaded replay memory with {len(agent.memory)} experiences")

        else:
            print("Starting with empty replay memory")
        
        print(f"Agent loaded successfully!")
        print(f" Model type: {agent.model_type}")
        print(f" Epsilon: {agent.epsilon:.4f}")
        print(f" Train steps: {agent.train_step_counter}")
        print(f" Device: {agent.device}")
        
        return agent
    
    def get_hyperparameters(self):
        return {
            "gamma": self.gamma,
            "epsilon_on_init": self.epsilon,
            "batch_size": self.batch_size,
            "memory_size": self.memory_size,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "use_double_dqn": self.use_double_dqn,
            "target_update_frequency": self.target_update_frequency
        }