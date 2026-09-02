import numpy as np
from Multi_Arm_Bandit_Machine import *

class Myopic_User(object):
    """
    Introduction:
    - This is a class for the user of multi_arm bandit machine, who uses the Myopic Policy. Given the necessary parameter
      for the machine and the rule of the game, the class could simulate the game process of the user in given time. 
      
    Input:
    - MABM part:
      - arm_num: a scalar indicating the number of arms for the machine(or the number of total channels)
      - state_num: a scalar indicating the number of state for each arm(channel)(, given that they are all the same)
      - initial_dist: a matrix with dimension (arm_num, state_num), each row of which shows the initial distribution 
        of each arm(channel)
      - transition_matrics: a tensor with dimension (arm_num, state_num, state_num), each matrix of which shows
        the transition matrix of each arm(channel)
    
    - Game Rule part:
      - active_num: a scalar indicating the number of active arms(channels)(, that is the arms user could observe)
      - reward_set: an matrix with dimension (arm_num, state_num) showing the reward of each state for each arm(channel)
      - time: a scalar indicating the time user plays the game
    """
    def __init__(self, arm_num, state_num, initial_dist, transition_matrics, active_num, reward_set, time):
        """
        Parameters:
        - N: a scalar for arm_numm
        - W: an (N, M) marix, each row of which shows the current belief vector of each arm(channel)
        - T: a scalar for the playing time
        - action: an (N,) array showing the current action of each arm(channel)(active or passive)
        - ER: an (N,) array showing the current expected reward of each arm(channel)
        - AR: an (T,) array showing the actual reward of each time
        - mabm: an object simulating the Multi_Arm Bandit Machine
        - R: an (N, M) matrix for reward_set
        - K: a scalar for active_num
        """
        self.belief_vec = initial_dist.copy()
        self.N = arm_num
        self.T = time
        self.action = np.zeros(self.N)
        self.ER = np.zeros(self.N)
        self.AR = np.array([])
        self.mabm = MABM(arm_num, state_num, initial_dist, transition_matrics)
        self.R = reward_set
        self.K = active_num

    
    def ER_update(self):
        """
        Function:
        - Compute the expected reward for each arm(channel) of each time according to the belief vector.
        """
        self.ER = np.sum(self.belief_vec*self.R, axis = 1)
        
    def action_AR_update(self):
        """
        Function:
        - Find the active arms(channels) of each time according to the expected rewards of them.
        - Compute actual reward according to the action of each time.
        """
        self.action = np.zeros(self.N)
        self.action[self.ER.argsort()[::-1][0:self.K]] = 1
        self.AR = np.append(self.AR, np.sum(self.R[self.action == 1, self.mabm.state[self.action == 1]]))
        
    def belief_vector_update(self):
        """
        Function:
        - Update the belief vector according to the action of each arm(channel).
        """
        self.belief_vec[self.action == 1] = self.mabm.tran_prob[self.action == 1]
        self.belief_vec[self.action == 0] = (self.belief_vec[self.action == 0].reshape((
            self.N - self.K, 1, self.mabm.M))@self.mabm.P[self.action == 0, :, :]).reshape((self.N - self.K, self.mabm.M))
        
    def User_process(self):
        """
        Function:
        - Simulate the game process of a myopic user with above functions.
        """
        for t in range(self.T):
            self.mabm.MABM_update()
            self.ER_update()
            self.action_AR_update()
            self.belief_vector_update()
                 