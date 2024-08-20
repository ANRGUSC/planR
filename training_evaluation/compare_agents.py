import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def downsample(data, factor):
    return data[::factor]

def plot_individual(q_log, dqn_log, metric, title, ylabel, unit, window_size=100, downsample_factor=5):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    if metric not in q_log.columns or metric not in dqn_log.columns:
        print(f"Metric '{metric}' not found in the dataset")
        return

    q_smoothed_data = moving_average(q_log[metric], window_size)
    q_smoothed_data = downsample(q_smoothed_data, downsample_factor)
    dqn_smoothed_data = moving_average(dqn_log[metric], window_size)
    dqn_smoothed_data = downsample(dqn_smoothed_data, downsample_factor)

    sns.lineplot(ax=axs[0], x=downsample(q_log['episode'], downsample_factor), y=q_smoothed_data, label='Q-Learning', color='teal', ci='sd')
    sns.lineplot(ax=axs[1], x=downsample(dqn_log['episode'], downsample_factor), y=dqn_smoothed_data, label='DQN', color='orange', ci='sd')

    axs[0].set_title(f'Q-Learning {title}', fontsize=10)
    axs[1].set_title(f'DQN {title}', fontsize=10)
    for ax in axs:
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel(f'{ylabel} ({unit})', fontsize=10)
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"{metric} - Q-Learning: min={q_smoothed_data.min():.4f}, max={q_smoothed_data.max():.4f}, mean={q_smoothed_data.mean():.4f}")
    print(f"{metric} - DQN: min={dqn_smoothed_data.min():.4f}, max={dqn_smoothed_data.max():.4f}, mean={dqn_smoothed_data.mean():.4f}")

def main():
    qlearning_log = pd.read_csv('training_metrics_q_curious-bird-79_0.2.csv')
    dqn_log = pd.read_csv('training_metrics_dqn_firm-pond-80_0.2.csv')

    metrics = [
        ('cumulative_reward', 'Cumulative Reward', 'Cumulative Reward', 'points'),
        ('average_reward', 'Average Reward', 'Average Reward', 'points/episode'),
        ('discounted_reward', 'Discounted Reward', 'Discounted Reward', 'points'),
        ('q_value_change', 'Q-Value Change', 'Q-Value Change', 'mean squared difference'),
        ('sample_efficiency', 'Unique States Visited', 'Unique States Visited', 'states/episode'),
        ('policy_entropy', 'Policy Entropy', 'Policy Entropy', 'nats')
    ]

    for metric, title, ylabel, unit in metrics:
        plot_individual(qlearning_log, dqn_log, metric, f'training_metrics_{metric}', ylabel, unit)

    summary_data = {
        'Metric': ['Memory Usage', 'Episodes'],
        'Q-Learning': [
            qlearning_log['space_complexity'].max(),
            qlearning_log['episode'].max()
        ],
        'DQN': [
            dqn_log['space_complexity'].max(),
            dqn_log['episode'].max()
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)

    summary_df.to_csv('summary_metrics.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig('summary_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 6))
    memory_usage_data = {
        'Q-Learning': qlearning_log['space_complexity'].max(),
        'DQN': dqn_log['space_complexity'].max()
    }
    plt.bar(memory_usage_data.keys(), memory_usage_data.values(), color=['teal', 'orange'])
    plt.xlabel('Agent', fontsize=10)
    plt.ylabel('Memory Usage (bytes)', fontsize=10)
    plt.title('Memory Usage Comparison', fontsize=10)
    plt.tight_layout()
    plt.savefig('memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
