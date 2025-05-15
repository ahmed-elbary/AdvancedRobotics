# Re-run everything after kernel reset, including agents and training + updated bar chart

import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------- ENVIRONMENT ----------------
class RestaurantGrid:
    def __init__(self):
        self.rows, self.cols = 3, 4
        self.kitchen = (2, 0)
        self.correct_table = (0, 3)
        self.wrong_table = (1, 3)
        self.blocked = [(1, 1)]
        self.actions = ['up', 'down', 'left', 'right']
        self.reset()

    def reset(self):
        self.state = self.kitchen
        return self.state

    def transition(self, action):
        r, c = self.state
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        dr, dc = moves[action]
        new_r, new_c = r + dr, c + dc
        new_state = (new_r, new_c)
        if 0 <= new_r < self.rows and 0 <= new_c < self.cols and new_state not in self.blocked:
            self.state = new_state
        return self.state

    def simulate_transition(self, state, action):
        r, c = state
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        dr, dc = moves[action]
        new_r, new_c = r + dr, c + dc
        new_state = (new_r, new_c)
        if 0 <= new_r < self.rows and 0 <= new_c < self.cols and new_state not in self.blocked:
            return new_state
        return state

    def get_reward(self):
        if self.state == self.correct_table:
            return 1
        elif self.state == self.wrong_table:
            return -1
        else:
            return -0.05

    def is_terminal(self):
        return self.state in [self.correct_table, self.wrong_table]

# ---------------- AGENTS ----------------
class BaseAgent:
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=0.3):
        self.env = env
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = {(r, c): {a: 0.0 for a in env.actions}
                  for r in range(env.rows)
                  for c in range(env.cols)
                  if (r, c) not in env.blocked}

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        return max(self.Q[state], key=self.Q[state].get)

    def update_q(self, state, action, reward, next_state):
        max_q = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_q - self.Q[state][action])

    def shaped_feedback(self, state):
        if state == self.env.correct_table:
            return 0.8
        elif state == self.env.wrong_table:
            return 0
        else:
            return 0.0

class IRL(BaseAgent):
    def play_episode(self, max_steps=15):
        state = self.env.reset()
        total_reward, steps = 0, 0
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state = self.env.transition(action)
            reward = self.env.get_reward() + self.shaped_feedback(next_state)
            self.update_q(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            steps += 1
            if self.env.is_terminal():
                break
        return total_reward, steps, state == self.env.correct_table

class TAMER(BaseAgent):
    def play_episode(self, max_steps=15):
        state = self.env.reset()
        total_reward, steps = 0, 0
        for _ in range(max_steps):
            for _ in range(5):  # simulate pre-approval
                action = self.select_action(state)
                if self.env.simulate_transition(state, action) != self.env.wrong_table:
                    break
            next_state = self.env.transition(action)
            reward = self.env.get_reward() + self.shaped_feedback(next_state)
            self.update_q(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            steps += 1
            if self.env.is_terminal():
                break
        return total_reward, steps, state == self.env.correct_table

# ---------------- TRAINING ----------------
def run_training(agent, episodes=30):
    rewards, steps, successes, times = [], [], [], []
    for _ in range(episodes):
        start = time.time()
        r, s, win = agent.play_episode()
        rewards.append(r)
        steps.append(s)
        successes.append(win)
        times.append(time.time() - start)
    return rewards, steps, successes, times

env = RestaurantGrid()
agent_irl = IRL(env)
agent_tamer = TAMER(env)

r1, s1, win1, t1 = run_training(agent_irl)
r2, s2, win2, t2 = run_training(agent_tamer)

# ---------------- BAR CHART ----------------
avg_reward_irl = np.mean(r1)
avg_reward_tamer = np.mean(r2)
avg_steps_irl = np.mean(s1)
avg_steps_tamer = np.mean(s2)
success_rate_irl = np.sum(win1) / len(win1)
success_rate_tamer = np.sum(win2) / len(win2)

metrics = ['Avg Reward', 'Avg Steps', 'Success Rate']
irl_values = [avg_reward_irl, avg_steps_irl, success_rate_irl]
tamer_values = [avg_reward_tamer, avg_steps_tamer, success_rate_tamer]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, irl_values, width, label='IRL')
plt.bar(x + width/2, tamer_values, width, label='TAMER')
plt.ylabel('Metric Value')
plt.title('Average Performance Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("average_comparison_bar_chart.png")
plt.show()
