import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import csv
import random

# === Configurable Parameters ===
ROWS, COLS = 4, 4
START = (3, 0)
GOAL = (0, 3)
OBSTACLES = [(1, 1), (1, 2), (3, 2)]
ACTIONS = ['up', 'down', 'left', 'right']
EPISODES = 25
MAX_STEPS = 15
ALPHA = 0.4
GAMMA = 0.9
EPSILON = 0.2
GUIDANCE_MODE = "override"  # Options: "none", "override", "reward_shaping"

# === Q-table Initialization ===
Q = {
    (r, c): {a: 0.0 for a in ACTIONS}
    for r in range(ROWS) for c in range(COLS) if (r, c) not in OBSTACLES
}

# === Helper Functions ===
def is_obstacle(pos):
    return pos in OBSTACLES

def next_state(pos, action):
    r, c = pos
    if action == 'up': r -= 1
    elif action == 'down': r += 1
    elif action == 'left': c -= 1
    elif action == 'right': c += 1
    if 0 <= r < ROWS and 0 <= c < COLS and not is_obstacle((r, c)):
        return (r, c)
    return pos

def choose_action(state):
    return random.choice(ACTIONS) if np.random.rand() < EPSILON else max(Q[state], key=Q[state].get)

def show_board(state):
    print("\nCurrent Board:")
    for r in range(ROWS):
        print("+" + "----+" * COLS)
        row = "|"
        for c in range(COLS):
            cell = (r, c)
            if cell == state:
                row += " A  |"
            elif cell == START:
                row += " S  |"
            elif cell == GOAL:
                row += " G  |"
            elif is_obstacle(cell):
                row += " X  |"
            else:
                row += "    |"
        print(row)
    print("+" + "----+" * COLS)

def compute_learning_curve(rewards, window=10):
    return [np.mean(rewards[max(0, i-window+1):i+1]) if i >= window-1 else None for i in range(len(rewards))]

def show_values(Q_table):
    print("\nLearned Q-values (max per state):")
    for r in range(ROWS):
        print("+" + "--------+" * COLS)
        row = "|"
        for c in range(COLS):
            if (r, c) in Q_table:
                best_q = max(Q_table[(r, c)].values())
                row += f" {best_q:5.2f} |"
            elif (r, c) in OBSTACLES:
                row += " XXXXX |"
            else:
                row += "        |"
        print(row)
    print("+" + "--------+" * COLS)

def show_policy(Q_table):
    print("\nFinal Learned Policy (best actions):")
    symbol = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    for r in range(ROWS):
        print("+" + "----+" * COLS)
        row = "|"
        for c in range(COLS):
            pos = (r, c)
            if pos == GOAL:
                row += " G  |"
            elif pos in OBSTACLES:
                row += " X  |"
            elif pos in Q:
                row += f" {symbol[max(Q[pos], key=Q[pos].get)]}  |"
            else:
                row += " .  |"
        print(row)
    print("+" + "----+" * COLS)

def show_qtable(Q_table):
    print("\nDetailed Q-Table:")
    for state in sorted(Q_table):
        q = Q_table[state]
        print(f"{state}: U={q['up']:.2f} D={q['down']:.2f} L={q['left']:.2f} R={q['right']:.2f}")

def print_summary(metrics):
    total = len(metrics['rewards'])
    steps = sum(metrics['steps'])
    rewards = sum(metrics['rewards'])
    print("\nTraining Summary:")
    print(f"Total Episodes     : {total}")
    print(f"Total Steps        : {steps}")
    print(f"Average Steps/Ep   : {steps/total:.2f}")
    print(f"Total Reward       : {rewards:.2f}")
    print(f"Average Reward/Ep  : {rewards/total:.2f}")
    print(f"Cumulative Reward  : {np.cumsum(metrics['rewards'])[-1]:.2f}")

def export_results(Q_table, metrics, name='delivery_task'):
    with open(f'{name}_policy.pkl', 'wb') as f:
        pickle.dump(Q_table, f)
    with open(f'{name}_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Steps', 'Reward', 'CumulativeReward', 'LearningCurveReward'])
        curve = compute_learning_curve(metrics['rewards'])
        cum = np.cumsum(metrics['rewards'])
        for i in range(len(metrics['rewards'])):
            writer.writerow([i+1, metrics['steps'][i], metrics['rewards'][i], cum[i], curve[i] if curve[i] else ""])

def plot_metrics(metrics):
    episodes = list(range(1, len(metrics['rewards']) + 1))
    plt.plot(episodes, metrics['steps'])
    plt.title("Steps per Episode")
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.grid(True); plt.show()

    plt.plot(episodes, metrics['rewards'])
    plt.title("Reward per Episode")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.grid(True); plt.show()

    plt.plot(episodes, np.cumsum(metrics['rewards']))
    plt.title("Cumulative Reward")
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.grid(True); plt.show()

def plot_qvalue_heatmap(Q_table, title='Max Q-Value per State'):
    grid = np.full((ROWS, COLS), np.nan)
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in Q_table:
                grid[r, c] = max(Q_table[(r, c)].values())
    plt.figure(figsize=(6, 5))
    sns.heatmap(grid, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, linecolor='gray')
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.gca().invert_yaxis()
    plt.show()

def plot_learning_curve(metrics, window=10):
    rewards = metrics['rewards']
    episodes = list(range(1, len(rewards) + 1))
    learning_curve = [np.mean(rewards[max(0, i - window + 1):i + 1]) if i >= window - 1 else None for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, learning_curve, marker='o')
    plt.title(f"Learning Curve (Moving Avg over {window} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()

# === Training Loop ===
metrics = {'steps': [], 'rewards': []}
override_count = 0

for episode in range(EPISODES):
    state = START
    total_reward = 0
    step_count = 0
    print(f"\n=== Episode {episode + 1} ===")
    show_board(state)

    while state != GOAL:
        action = choose_action(state)

        if GUIDANCE_MODE == "override":
            print(f"Suggested action: {action}")
            while True:
                fb = input("Is this action (g)ood or (b)ad? ").strip().lower()
                if fb == 'b':
                    action = input("Enter new action (up/down/left/right): ").strip().lower()
                    override_count += 1
                    break
                elif fb == 'g':
                    break
                else:
                    print("Please enter 'g' or 'b'.")

        next_s = next_state(state, action)
        reward = 1 if next_s == GOAL else -0.04

        if GUIDANCE_MODE == "reward_shaping":
            while True:
                fb = input(f"Is state {next_s} (g)ood or (b)ad? ").strip().lower()
                if fb == 'g':
                    reward += 0.1
                    break
                elif fb == 'b':
                    reward -= 0.1
                    break
                else:
                    print("Please enter 'g' or 'b'.")

        if next_s in Q:
            max_future_q = max(Q[next_s].values())
        else:
            max_future_q = 0.0

        Q[state][action] += ALPHA * (reward + GAMMA * max_future_q - Q[state][action])
        state = next_s
        total_reward += reward
        step_count += 1
        show_board(state)

        if step_count >= MAX_STEPS:
            print("Max steps reached.")
            break

    metrics['steps'].append(step_count)
    metrics['rewards'].append(total_reward)

# === Results ===
print(f"\n🧠 Total Human Overrides during training: {override_count}")
print_summary(metrics)
export_results(Q, metrics)
show_values(Q)
show_policy(Q)
show_qtable(Q)
plot_metrics(metrics)
plot_qvalue_heatmap(Q)
plot_learning_curve(metrics)
