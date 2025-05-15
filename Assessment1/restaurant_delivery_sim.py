class IRLAgent(BaseAgent):
    def play_episode(self, max_steps=15):
        state = self.env.reset()
        total_reward = 0
        for step in range(max_steps):
            self.env.render()
            action = self.select_move(state)
            next_state = self.env.transition(action)

            feedback = input(f"Rate this state {next_state} (g = good, b = bad, n = neutral): ").strip()
            shaped_reward = {'g': 0.2, 'b': -0.2, 'n': 0.0}.get(feedback, 0.0)
            final_reward = self.env.get_reward() + shaped_reward

            self.update_q(state, action, final_reward, next_state)
            total_reward += final_reward
            state = next_state
            if self.env.is_terminal():
                break
        return total_reward


class TAMERAgent(BaseAgent):
    def play_episode(self, max_steps=15):
        state = self.env.reset()
        total_reward = 0
        for step in range(max_steps):
            self.env.render()
            while True:
                action = self.select_move(state)
                print(f"Proposed action: {action}")
                fb = input("Approve? (g = yes, b = no): ").strip()
                if fb == 'g':
                    break
                else:
                    print("Suggesting new action...")
            next_state = self.env.transition(action)
            feedback = input(f"Is this state {next_state} good? (g/b/n): ").strip()
            shaped_reward = {'g': 0.2, 'b': -0.2, 'n': 0.0}.get(feedback, 0.0)
            final_reward = self.env.get_reward() + shaped_reward

            self.update_q(state, action, final_reward, next_state)
            total_reward += final_reward
            state = next_state
            if self.env.is_terminal():
                break
        return total_reward
