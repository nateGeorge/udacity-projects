'''
references:
https://github.com/rahulravindran0108/Smart-cab
'''

import time
import random
import pickle as pk
import numpy as np

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from numpy.random import randint
from collections import namedtuple

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.legal_actions = Environment.valid_actions

        self.state = None
        
        # initialize variables
        self.qDict = dict()
        self.alpha = 0.1
        self.gamma = 0.8
        
        self.previous_state = None
        self.previous_action = None    
        self.previous_reward = None
        
        self.next_waypoint = None
        
        self.penalties = 0
        self.total_penalties = 0
        self.rewards = 0
        self.total_rewards = 0
        
        self.non_ideal_moves = 0
        self.total_non_ideal_moves = 0
        self.actions = 0
        self.total_actions = 0
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.previous_state = None
        self.state = None
        self.previous_action = None
        self.epsilon = 0.0
        self.total_rewards += self.rewards
        self.rewards = 0
        self.total_penalties += self.penalties
        self.penalties = 0
        self.non_ideal_moves = 0
        self.actions = 0
        
    def setAG(self, alpha, gamma):
        '''
        sets alpha and gamma
        '''
        self.alpha = alpha
        self.gamma = gamma
        return

    def flipCoin(self, p ):
        r = random.random()
        return r < p

    def getQValue(self, state, action):
        '''
        input: (state,action)
        output: Q value for the (state,action)
        returns 0 if the value is not present in the dictionary.
        '''
        return self.qDict.get((state, action), 0)

    def getValue(self, state):
        '''
         Returns max_action Q(state,action)
         where the max is over legal actions
        '''
        bestQValue = -np.inf
        
        for action in self.legal_actions:
            if(self.getQValue(state, action) > bestQValue):
                bestQValue = self.getQValue(state, action)

        return bestQValue

    def getPolicy(self, state):
        '''
        Compute the best action to take in a state.
        input: state
        output: best possible action(policy maps states to action)
        Working:
        From all the legal actions, return the action that has the bestQvalue.
        if two actions are tied, flip a coin and choose randomly between the two.
        '''
        bestAction, bestQValue = None, -np.inf
        
        for action in self.legal_actions:
            if(self.getQValue(state, action) > bestQValue):
                bestQValue = self.getQValue(state, action)
                bestAction = action

            if(self.getQValue(state, action) == bestQValue):
                if(self.flipCoin(.5)):
                    bestQValue = self.getQValue(state, action)
                    bestAction = action

        return bestAction

    def setState(self, stateInput):
        '''
        sets state of the agent for the qDictionary
        input: State
        ouput: Named State tuple
        '''
        State = namedtuple('State', ['waypoint', 'light', 'oncoming', 'left', 'right'])
        self.state = State(self.next_waypoint, stateInput['light'], stateInput['oncoming'], stateInput['left'], stateInput['right'])
    
    def printStats(self):
        '''
        prints statistics from trial
        '''
        print 'penlaties, rewards:', self.penalties, self.rewards
        print 'non ideal moves:', self.non_ideal_moves
        print 'percent non-ideal:', round(self.non_ideal_moves/float(self.actions), 2)*100
        print ''
        print ''
    
    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        self.setState(self.env.sense(self))

        # get the current best action based on q table
        action = self.getPolicy(self.state)
        self.actions += 1
        self.total_actions += 1
        non_ideal = self.checkMove(action)
        reward = self.env.act(self, action)
        
        if reward < 0:
            self.penalties += reward
            if not non_ideal:
                print 'penalty!'
                print 'action, waypoint', action, self.next_waypoint
                print self.state

        '''print ''
        print 'waypoint:', self.next_waypoint
        print 'action:', action
        print 'reward:', reward
        print self.state'''
        
        ## in case of initial configuration don't update the q table, else update q table
        if self.previous_reward is not None:
            self.updateQTable(self.previous_state,self.previous_action,self.state,self.previous_reward)

        # store the previous action and state so that we can update the q table on the next iteration
        self.previous_action = action
        self.previous_state = self.state
        self.previous_reward = reward
        self.rewards += reward

    def updateQTable(self, state, action, nextState, reward):
        '''
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here
        '''
    
        if (state, action) not in self.qDict:
            self.qDict[(state, action)] = reward
        else:
            # set the previous state's qValue to itself plus alpha*(reward + gamma*value of next state - old q value)
            self.qDict[(state, action)] = self.qDict[(state, action)] + self.alpha*(reward + self.gamma*self.getValue(nextState) - self.qDict[(state, action)])

    def checkMove(self, action):
        '''
        checks viability of an action and compares with ideal action
        This is for checking if agent finds dest in min time
        '''
        move_okay = False
        if action == 'forward':
            if self.state.light != 'green':
                move_okay = True
        elif action == 'left':
            if self.state.light == 'green' and (self.state.oncoming == None or self.state.oncoming == 'left'):
                move_okay = True
            else:
                move_okay = False
        elif action == 'right':
            if self.state.light == 'green' or self.state.left != 'straight':
                move_okay = True
            else:
                move_okay = False
        
        if move_okay and action != self.next_waypoint:
            print 'non-ideal move:'
            print 'action, waypoint', action, self.next_waypoint
            print self.state
            self.non_ideal_moves += 1
            self.total_non_ideal_moves += 1
            return True
        else:
            return False

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0000001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print 'alpha, gamma:', a.alpha, a.gamma
    print 'penalties:', a.total_penalties
    print 'total rewards:', a.total_rewards

def tune_a_g():
    '''
    tries each combination of alpha and gamma from 0.1 to 0.9 in 0.1 increments
    '''
    start = time.time()
    tryRange = range(1, 10)
    tryRange = [i/10.0 for i in tryRange]
    
    rewards = {} # dict of total rewards for different alphas and gammas
    penalties = {}
    total_actions = {} # total number of time steps taken
    non_ideal_actions = {}
    
    for i in tryRange:
        for j in tryRange:
            penalties['a=' + str(i) + ', g=' + str(j)] = 0
            rewards['a=' + str(i) + ', g=' + str(j)] = 0
            total_actions['a=' + str(i) + ', g=' + str(j)] = 0
            non_ideal_actions['a=' + str(i) + ', g=' + str(j)] = 0
            for k in range(10): # run 10 times each because it is stochastic, ideally should run this more times to get better statistics
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                a.setAG(i, j)
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, update_delay=0.0000000001, display=False)  # create simulator (uses pygame when display=True, if available)
                # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                sim.run(n_trials=100)  # run for a specified number of trials
                penalties['a=' + str(i) + ', g=' + str(j)] += a.total_penalties
                rewards['a=' + str(i) + ', g=' + str(j)] += a.total_rewards
                total_actions['a=' + str(i) + ', g=' + str(j)] += a.total_actions
                non_ideal_actions['a=' + str(i) + ', g=' + str(j)] += a.total_non_ideal_moves
            
    # get best alpha/gamma with lowest penalties
    #print 'penalties =', sorted(penalties.items())
    bestP = max(penalties, key=penalties.get)
    print 'best by penalties:', bestP, penalties[bestP]
    
    # get best alpha/gamma with highest total rewards
    #print 'rewards =', sorted(rewards.items())
    bestR = max(rewards, key=rewards.get)
    print 'best by rewards:', bestR, rewards[bestR]
    
    # get best alpha/gamma with highest total rewards
    #print 'total_actions =', sorted(total_actions.items())
    bestTA = min(total_actions, key=total_actions.get)
    print 'best by time steps:', bestTA, total_actions[bestTA]
    
    # get best alpha/gamma with highest total rewards
    #print 'non_ideal_actions =', sorted(non_ideal_actions.items())
    bestNIA = min(non_ideal_actions, key=non_ideal_actions.get)
    print 'best by non_ideal_actions:', bestNIA, non_ideal_actions[bestNIA]
    
    pk.dump(penalties, open('penalties.pk','w'))
    pk.dump(rewards, open('rewards.pk','w'))
    pk.dump(total_actions, open('total_actions.pk','w'))
    pk.dump(non_ideal_actions, open('non_ideal_actions.pk','w'))
    
    end = time.time()
    
    print 'took', int(end-start), 'seconds'
    
if __name__ == '__main__':
    #run()
    tune_a_g()
