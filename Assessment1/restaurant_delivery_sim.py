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
            return 0.2
        elif state == self.env.wrong_table:
            return -0.2
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

# ---------------- TRAINING & PLOTTING ----------------
def run_training(agent, episodes=100):
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

episodes = np.arange(1, 101)

# Plot rewards
plt.figure()
plt.plot(episodes, r1, label="IRL")
plt.plot(episodes, r2, label="TAMER")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("reward_plot.png")
plt.close()

# Plot steps
plt.figure()
plt.plot(episodes, s1, label="IRL")
plt.plot(episodes, s2, label="TAMER")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per Episode")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("steps_plot.png")
plt.close()

# Plot success rate (cumulative)
plt.figure()
plt.plot(episodes, np.cumsum(win1)/episodes, label="IRL")
plt.plot(episodes, np.cumsum(win2)/episodes, label="TAMER")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Cumulative Success Rate")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("success_rate_plot.png")
plt.close()

"Training complete. Plots saved as 'reward_plot.png', 'steps_plot.png', and 'success_rate_plot.png'."
