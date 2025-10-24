import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prepare_dataframe(episodes, rewards, lengths, losses, epsilon_history = None):
    rolling_window = max(1, episodes // 100)  # 1% rolling window
    
    df = pd.DataFrame({
        'episode': range(episodes),
        'reward': rewards,
        'length': lengths,
        'loss': losses
    })
    
    if epsilon_history:
        df['epsilon'] = epsilon_history
    
    df['reward_rolling'] = df['reward'].rolling(window = rolling_window, min_periods = 1).mean()
    df['reward_std'] = df['reward'].rolling(window = rolling_window, min_periods = 1).std()
    df['length_rolling'] = df['length'].rolling(window = rolling_window, min_periods = 1).mean()
    df['loss_rolling'] = df['loss'].rolling(window = rolling_window, min_periods = 1).mean()
        
    return df, rolling_window


def plot_rewards_overview(df, rolling_window, save_path = "plots/rewards.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    episodes_arr = df['episode'].values
    rewards_arr = df['reward'].values
    reward_rolling_arr = df['reward_rolling'].values
    reward_std_arr = df['reward_std'].values
    
    ax.plot(episodes_arr, rewards_arr, alpha = 0.3, color = 'steelblue', label = 'Episode Reward')
    ax.plot(episodes_arr, reward_rolling_arr, color = 'darkblue', linewidth = 2, label = f'Rolling Mean (n={rolling_window})')
    ax.fill_between(
        episodes_arr, 
        reward_rolling_arr - reward_std_arr, 
        reward_rolling_arr + reward_std_arr, 
        alpha = 0.2, 
        color = 'steelblue', 
        label='±1 Std Dev'
    )
    ax.axhline(y = 0, color = 'red', linestyle = '--', alpha = 0.5, label = 'Break-even')
    
    ax.set_title('Episode Rewards Over Time', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend(loc = 'best')
    ax.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_td_error(td_errors, save_path = "plots/td_error.png"):
    if not td_errors:
        print(f"{save_path} (no TD error data)")
        return
    
    fig, ax = plt.subplots(figsize = (12, 6))
    
    episodes_arr = range(len(td_errors))
    rolling_window = max(1, len(td_errors) // 100)
    td_rolling = pd.Series(td_errors).rolling(window=rolling_window, min_periods=1).mean()
    
    ax.plot(episodes_arr, td_errors, alpha = 0.3, color = 'orange', label = 'TD Error')
    ax.plot(episodes_arr, td_rolling.values, color = 'darkorange', linewidth = 2, label = 'Rolling Mean')
    
    ax.set_title('Temporal Difference Error Over Time', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Absolute TD Error')
    ax.legend()
    ax.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_episode_length(df, save_path = "plots/episode_length.png"):
    fig, ax = plt.subplots(figsize = (12, 6))
    
    episodes_arr = df['episode'].values
    length_arr = df['length'].values
    length_rolling_arr = df['length_rolling'].values
    
    ax.plot(episodes_arr, length_arr, alpha = 0.3, color = 'orange', label = 'Episode Length')
    ax.plot(episodes_arr, length_rolling_arr, color = 'darkorange', linewidth = 2, label = 'Rolling Mean')
    
    ax.set_title('Episode Length Over Time', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_training_loss(df, save_path = "plots/training_loss.png"):
    if df['loss'].sum() == 0:
        print(f"{save_path} (no loss data)")
        return
    
    fig, ax = plt.subplots(figsize = (12, 6))
    
    episodes_arr = df['episode'].values
    loss_arr = df['loss'].values
    loss_rolling_arr = df['loss_rolling'].values
    
    ax.plot(episodes_arr, loss_arr, alpha = 0.3, color = 'purple', label = 'Loss')
    ax.plot(episodes_arr, loss_rolling_arr, color = 'darkviolet', linewidth = 2, label = 'Rolling Mean')
    
    ax.set_title('Training Loss Over Time', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Loss')
    ax.legend()
    ax.grid(True, alpha = 0.3)
    ax.set_yscale('log')  # Log scale often better for loss
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_epsilon_decay(df, save_path = "plots/epsilon.png"):
    if 'epsilon' not in df.columns:
        print(f"  ⊘ {save_path} (no epsilon data)")
        return
    
    fig, ax = plt.subplots(figsize = (12, 6))
    
    episodes_arr = df['episode'].values
    epsilon_arr = df['epsilon'].values
    
    ax.plot(episodes_arr, epsilon_arr, color = 'red', linewidth = 2)
    ax.set_title('Exploration Rate (Epsilon) Decay', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_action_distribution(action_distributions, save_path = "plots/action_distribution.png"):
    if not action_distributions:
        print(f"  ⊘ {save_path} (no action data)")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 10))
    
    # Heatmap over time
    action_matrix = np.array(action_distributions).T
    im = ax1.imshow(action_matrix, aspect = 'auto', cmap = 'YlOrRd', interpolation = 'nearest')
    ax1.set_title('Action Selection Heatmap Over Training', fontsize = 14, fontweight = 'bold')
    ax1.set_ylabel('Column')
    ax1.set_xlabel('Episode')
    ax1.set_yticks(range(7))
    plt.colorbar(im, ax = ax1, label = 'Action Count')
    
    # Total distribution
    total_actions = np.sum(action_distributions, axis = 0)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 7))
    bars = ax2.bar(range(7), total_actions, color = colors, edgecolor = 'black')
    ax2.set_title('Overall Action Distribution', fontsize = 14, fontweight = 'bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Total Count')
    ax2.set_xticks(range(7))
    ax2.grid(True, alpha = 0.3, axis = 'y')
    
    # Add percentage labels
    total = sum(total_actions)
    for i, (bar, count) in enumerate(zip(bars, total_actions)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({100*count/total:.1f}%)',
                ha = 'center', va = 'bottom', fontsize = 9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_statistics_summary(df, episodes, save_path = "plots/summary.png"):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Total Episodes', f'{episodes:,}'],
        ['Average Reward', f'{df["reward"].mean():.3f}'],
        ['Std Dev Reward', f'{df["reward"].std():.3f}'],
        ['Max Reward', f'{df["reward"].max():.3f}'],
        ['Min Reward', f'{df["reward"].min():.3f}'],
        ['Avg Episode Length', f'{df["length"].mean():.1f}'],
    ]
    
    if df['loss'].sum() > 0:
        stats_data.append(['Avg Loss', f'{df["loss"].mean():.4f}'])
    
    if 'epsilon' in df.columns:
        stats_data.append(['Final Epsilon', f'{df["epsilon"].iloc[-1]:.4f}'])
    
    table = ax.table(cellText = stats_data, cellLoc = 'left', loc = 'center',
                    colWidths = [0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight = 'bold', color = 'white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Training Statistics Summary', fontsize = 16, fontweight = 'bold', pad = 20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_metrics(episodes, rewards, lengths, losses, 
                 epsilon_history = None, td_errors = None, 
                 action_distributions = None, task = None):
    print("Generating training plots...")
    
    # Prepare data
    df, rolling_window = prepare_dataframe(episodes, rewards, lengths, losses, epsilon_history)
        
    if not os.path.exists("plots"):
        os.mkdir("plots")

    # Generate all plots
    plot_rewards_overview(df, rolling_window)
    plot_episode_length(df)
    plot_training_loss(df)
    plot_epsilon_decay(df)
    plot_td_error(td_errors) if td_errors else None  
    plot_action_distribution(action_distributions) if action_distributions else None  
    plot_statistics_summary(df, episodes)

    if task:
        print("Uploading plots to ClearML...")
        plot_dir = "plots"
        try:
            plot_files = [os.path.join(plot_dir, f) for f in os.listdir(plot_dir) if f.endswith(".png")]
            for plot_file in plot_files:
                plot_name = os.path.basename(plot_file).replace('.png', '').replace('_', ' ').title()
                # Uploads the image file and displays it in the "PLOTS" tab
                task.upload_artifact(
                    name = plot_name,
                    artifact_object = plot_file,
                    auto_pickle = False
                )

        except Exception as e:
            print(f"Error uploading plots to ClearML: {e}")

    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Episodes:          {episodes:,}")
    print(f"Average Reward:    {df['reward'].mean():.3f} ± {df['reward'].std():.3f}")
    print(f"Avg Episode Length: {df['length'].mean():.1f}")
    print("="*60 + "\n")