# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
import time
import math

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
"""

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
    Your value iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.

    Some useful mdp methods you will use:
        mdp.getStates()
        mdp.getPossibleActions(state)
        mdp.getTransitionStatesAndProbs(state, action)
        mdp.getReward(state, action, nextState)
  """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            update = util.Counter()
            for state in self.mdp.getStates():
                # the possible rewards
                rewards = []
                for action in self.mdp.getPossibleActions(state):
                    rewards.append(self.reward(state, action))
                    update[state] = max(rewards)

            # update to the latest values
            self.values = update.copy()

    def reward(self, state, action):
        """
    calculates the reward based on the action taken in the state
    :param state: the state we are in
    :param action: which action to take
    :param discount: the discount factor of the dmp
    :return: the maximum possible rewards over the possible next states. as per bellmans' function
    """
        reward = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward += self.mdp.getReward(state, action, next_state) + prob * self.discount * self.values[next_state]

        return reward

    def getValue(self, state):
        """
    Return the value of the state (computed in __init__).
    """
        return self.values[state]

    def getQValue(self, state, action):
        """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
      """
        "*** YOUR CODE HERE ***"
        if action is None:
            return self.values[state]

        q_val = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val += self.mdp.getReward(state, action, next_state) + \
                     self.discount * prob * self.values[next_state]

        return q_val

        # return self.values[state]

    def getPolicy(self, state):
        """
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.  Note that if
    there are no legal actions, which is the case at the
    terminal state, you should return None.
    """
        "*** YOUR CODE HERE ***"

        max_val = -math.inf
        arg_max = None

        for action in self.mdp.getPossibleActions(state):
            val = 0
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                val += prob * self.values[next_state]

                if max_val <= val:
                    max_val = val
                    arg_max = action

        return arg_max

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
