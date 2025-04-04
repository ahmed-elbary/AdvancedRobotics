#
# Advanced Robotics, 2024-2025
# Paul Baxter
# Workshop Week 3
#
# Extension of the week 2 simulation
#
# Run with:     python lecture3-simulation.py
#
# May need following depedencies (if pip is installed):
#   pip install numpy
#   pip install matplotlib
#
# Two classes in this file:
#  1. State: the board/maze (same as in week 2) - line 42
#  2. IRLAgent: a basic IRL agent, to be extended/modified - line 105
#


import numpy as np
import matplotlib.pyplot as plt

# global variables
# simulation parameters
N_IRLAGENT = 3          #number of iterations of IRLplus to perform
EXPLORE = 0.3           #the explore proportion: (1-EXPLORE) for exloit
MANUAL_FEEDBACK = 0.1   #reward feedback from human: + and -
NEUTRAL_FEEDBACK = 0.05 #if no feedback, this reward applied (+)
LOGGING = True         #set full logging to terminal or not...

# maze setup - leave alone for now
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)          #third row, first column


##########################################################
# The maze environment
##########################################################
class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if action == "up":
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)
        # if next state legal
        if (nxtState[0] >= 0) and (nxtState[0] <= 2):
            if (nxtState[1] >= 0) and (nxtState[1] <= 3):
                if nxtState != (1, 1):
                    return nxtState
        return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'o'
                if self.board[i, j] == -1:
                    token = 'X'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-----------------')


##########################################################
# Interactive RL agent
##########################################################
class IRLAgent:

    def __init__(self):
        self.states = []
        self.numStates = []
        self.rewards = []
        self.cumulativeReward = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = EXPLORE

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0

    def chooseAction(self):
        # choose action with greatest expected value in resulting state
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            #random action
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=3):
        i = 0
        print ("")
        print ("IRL START")
        print ("")
        stepCounter = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                self.rewards.append(reward)
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print ("Episode ", i, ": ")
                print ("--------------------- End Reward:", reward)
                print ("--------------------- Num Steps Used: ", stepCounter)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                self.numStates.append(stepCounter)
                self.showValues()
                stepCounter = 0
                i += 1
            else:
                print("___________________________________________________")
                stepCounter += 1
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                current_state = self.State.state    #current state before action is executed
                if (LOGGING):
                    print("  current position {} action {}".format(current_state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                if (LOGGING):
                    print ("    |--> next state", self.State.state)
                # for IRL allow user to define reward:
                #  - get reward from user:
                self.State.showBoard()
                feedback = input("      is this new state (g)ood or (b)ad to be in: ")
                u_reward = 0
                if feedback == "g":
                    u_reward = MANUAL_FEEDBACK
                elif feedback == "b":
                    u_reward = -MANUAL_FEEDBACK
                else:
                    #not recognised, assume ok...
                    u_reward = NEUTRAL_FEEDBACK
                #  - update the value of the current state only (state *after* action was executed)
                reward = self.state_values[self.State.state] + self.lr * (u_reward - self.state_values[self.State.state])
                self.state_values[self.State.state] = round(reward, 3)

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)        
        print ("")
        
    def cumulativeRewards(self):
        self.cumulativeReward = [0]
        for r in self.rewards:
            self.cumulativeReward.append(self.cumulativeReward[-1] + r)
            
    def resultsPlots(self):
        #NOTE: for the IRLagent, these plots only really useful with multiple episodes...
        self.cumulativeRewards()
        plt.plot(self.numStates, 'o')
        plt.title("Number of steps per episode")
        plt.xlabel("Episode number")
        plt.ylabel("Number of steps")
        plt.show()
        
        plt.plot(self.rewards, 'o')
        plt.title("Reward per episode")
        plt.xlabel("Episode number")
        plt.ylabel("Reward at end of episode")
        plt.show()
        
        plt.plot(self.cumulativeReward, 'o')
        plt.title("Cumulative reward")
        plt.xlabel("Episode number")
        plt.ylabel("Total reward")
        plt.show()


##########################################################
# Main
##########################################################
if __name__ == "__main__":
    irl = IRLAgent()
    irl.play(N_IRLAGENT)

    print ("_________________________________________________")
    print ("")
    print ("IRL agent: ", N_IRLAGENT, " iterations")
    print(irl.showValues())
    print ("")
    irl.resultsPlots()
    print("")
