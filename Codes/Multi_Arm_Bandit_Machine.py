import numpy as np


class MABM(object):
    """ 
    Introduction:
    - This is a class to simulate the work of multi_arm bandit machine. Given the necessary parameters, 
      the class could simulate the Discrete Markov Chain happens in the machine.
    
    Input:
    - arm_num: a scalar indicating the number of arms for the machine(or the number of total channels)
    - state_num: a scalar indicating the number of state for each arm(channel)(, given that they are all the same)
    - initial_dist: a matrix with dimension (arm_num, state_num), each row of which shows the initial distribution
      of each arm(channel)
    - transition_matrics: a tensor with dimension (arm_num, state_num, state_num), each matrix of which shows
      the transition matrix of each arm(channel)
    """
    
    def __init__(self, arm_num, state_num, initial_dist, transition_matrics):
        """
        Parameters:
        - N: a scalar for arm_numm
        - M: a scalar for state_num
        - W: an (N, M) marix, each row of which shows the current transition probablity of each arm(channel)
        - P: an (N, M, M) tensor, each matrix of which shows the transition matrix of each arm(channel)
        - state: an (N,) array showing the current state of each arm(channel)
        """
        self.tran_prob = initial_dist.copy()
        self.N = arm_num
        self.P = transition_matrics.copy()
        self.M = state_num
        self.state = np.zeros(self.N)
    
    
    def state_update(self):
        """
        Function:
        - Upadte the state of each arm(channel) according to the transition of Markov Chain.
        """
        for n in range(self.N):
            # Sample a scalar state index for arm n.
            self.state[n] = np.random.choice(a=self.M, p=self.tran_prob[n])
        self.state = self.state.astype(int)
    
    def transition_prob_update(self):
        """
        Function:
        - Update the current transition probability for each arm(channel) according to the current states of them.
        """
        self.tran_prob = self.P[(range(self.N),self.state)]
    
    def MABM_update(self):
        """
        Function:
        - Operate the machine once using the above functions.
        """
        self.state_update()
        #print(self.state)
        self.transition_prob_update()
        #print(self.W)
    
