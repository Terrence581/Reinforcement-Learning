import numpy as np
from Multi_Arm_Bandit_Machine import *

class Finite_Optimal_User(object):
    """
    Input:
    - MABM part:
      - arm_num: a scalar indicating the number of arms for the machine(or the number of total channels)
      - state_num: a scalar indicating the number of state for each arm(channel)(, given that they are all the same)
      - initial_dist: a matrix with dimension (arm_num, state_num), each row of which shows the initial distribution of each arm(channel)
      - transition_matrics: a tensor with dimension (arm_num, state_num, state_num), each matrix of which shows the transition matrix of each arm(channel)
    
    - Game Rule part:
      - active_num: a scalar indicating the number of active arms(channels)(, that is the arms user could observe)
      - reward_set: an matrix with dimension (arm_num, state_num) showing the reward of each state for each arm(channel)
      - time: a scalar indicating the time user plays the game
    """
    def __init__(self, arm_num, state_num, initial_dist, transition_matrics, active_num, reward_set, time):
        """
        Parameters:
        - mabm: an object simulating the Multi_Arm Bandit Machine
        - action: an (N,) array showing the current action of each arm(channel)(active or passive)
        - N: a scalar for arm_num
        - K: a scalar for active_num
        - belief_vec: an (N, M) marix, each row of which shows the current belief vector of each arm(channel)
        - P: an (N, M, M) tensor for transition_matrics
        - R: an (N, M) matrix showing the reward of each state for each arm(channel)
        - T: a scalar for the playing time
        - AR: an (T,) array showing the actual reward of each time
        - tran_0: an (N,) array showing the transition probablity p_{01} of each arm(channel)
        - tran_1: an (N,) array showing the transition probablity p_{11} of each arm(channel)
        """
        self.mabm = MABM(arm_num, state_num, initial_dist, transition_matrics)
        self.action = np.zeros(arm_num)
        self.belief_vec = initial_dist.copy()
        self.N = arm_num
        self.K = active_num
        self.R = reward_set.copy()
        self.P = transition_matrics.copy()
        self.T = time
        self.AR = np.array([])
        self.tran_0 = transition_matrics[:, 0, 1].copy().reshape(self.N)
        self.tran_1 = transition_matrics[:, 1, 1].copy().reshape(self.N)
    
    def belief_vec_update(self):
        """
        Function:
        - Update the belief vector according to the action of each arm(channel).
        """
        self.belief_vec[self.action == 1] = self.mabm.tran_prob[self.action == 1]
        self.belief_vec[self.action == 0] = (self.belief_vec[self.action == 0].reshape((self.N - self.K, 1, self.mabm.M))@self.P[self.action == 0, :, :]).reshape((self.N - self.K, self.mabm.M))
            
    def dis_reward(self, belief, t, subsidy, beta):
        """
        Function:
        - Compute the maximum discounted reward over all arms(channels) according to the recursion algorithm.

        Input:
        - t: a scalar indicating which slot it is
        - subsidy: a scalar indicating the subsidy for passive arm(channel)
        - beta: a scalar indicating the discounted factor

        Output:
        - index for the arm (channel) with maximum discounted reward
        - maximum discounted reward
        """
        if t == self.T - 1:
            return np.argmax(belief*self.R[:, 1]), np.max(belief*self.R[:, 1])
        else:
            reward_all = np.zeros(self.N) + subsidy*(self.N - self.K)
            for n in range(self.N):
                w = belief.copy()
                w = w*self.tran_1 + (1 - w)*self.tran_0
                w1 = w.copy(); w1[n] = self.tran_1[n]
                w0 = w.copy(); w0[n] = self.tran_0[n]
                _, reward0 = self.dis_reward(w0, t + 1, subsidy, beta)
                _, reward1 = self.dis_reward(w1, t + 1, subsidy, beta)
                reward_all[n] = beta*(belief[n]*reward1 + (1 - belief[n])*reward0) + belief[n]*self.R[n, 1] 
            return np.argmax(reward_all), np.max(reward_all)

    def action_AR_update(self, t, subsidy, beta):
        """
        Function:
        - Find the active arms(channels) of each time according to the index of arm(channel) with maximum discounted reward.
        - Compute actual reward according to the action of each time.
        """
        self.action = np.zeros(self.N)
        n, _ = self.dis_reward(self.belief_vec[:, 1], t, subsidy, beta)
        self.action[n] = 1
        self.AR = np.append(self.AR, np.sum(self.R[self.action == 1, self.mabm.state[self.action == 1]]))
    
    def User_process(self, subsidy, beta):
        """
        Function:
        - Simulate the game process of a optimal user with above functions.
        """
        for t in range(self.T):
            self.mabm.MABM_update()
            #print(self.action)
            #self.dis_reward_update(t, subsidy, beta)
            self.action_AR_update(t, subsidy, beta)
            #print(self.action, "\n")
            self.belief_vec_update()


def dis_reward(belief_origin, reward_origin, N, P_origin, state_num, beta, t, time):
    """
    Function:
    - Compute the maximum discounted reward over all arms(channels) according to the recursion algorithm.

    Input:
    - N: a scalar indicating the number of arms in a bandit machine
    - state_num: a scalar indicating the number of states for those arms
    - belief_origin: a (N, M) matrix, each row of which shows current belief state distribution of each arm
    - P_origin: a (N, M, M) tensor, each matrix of which shows the transition matrix of each arm
    - reward_origin: a (N, M) matrix, each row of which shows the reward of each state for each arm
    - t: a scalar indicating which slot it is
    - beta: a scalar indicating the discounted factor
    - time: a scalar indicating the total length of the game period


    Output:
    - index for the arm (channel) with maximum discounted reward
    - maximum discounted reward
    """
    belief = belief_origin.copy().reshape((N, 1, state_num))
    reward = reward_origin.copy().reshape((N, state_num, 1))
    P = P_origin.copy()
    if t == time:
        return np.argmax((belief@reward).reshape(N)), np.max((belief@reward).reshape(N))
    else:
        reward_all = np.zeros(N)
        w = belief.copy()
        w = (w@P).copy()
        for n in range(N):
            reward_n = np.zeros(state_num)
            for i in range(state_num):
                w_i = w.copy()
                w_i[n] = P[n, i].copy()
                _, reward_i = dis_reward(w_i, reward, N, P, state_num, beta, t + 1, time)
                reward_n[i] = reward_i
            reward_all[n] = beta*belief[n].reshape(state_num)@reward_n + belief[n].reshape(state_num)@reward[n].reshape(state_num)
        return np.argmax(reward_all), np.max(reward_all)