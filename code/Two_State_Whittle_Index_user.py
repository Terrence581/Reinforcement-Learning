import numpy as np
from Multi_Arm_Bandit_Machine import *

class Whittle_Index_two_state:
    """
    Introduction:
    - This is a class for the user of multi_arm bandit machine with 2 states(0 and 1), who uses the Whittle Index Policy. Given the necessary parameters for the machine and the rule of the game, the class could simulate the game process of the user in given time. 
      
    Input:
    - MABM part:
      - arm_num: a scalar indicating the number of arms for the machine(or the number of total channels)
      - state_num: a scalar indicating the number of state for each arm(channel)(, given that they are all 2 in this class)
      - initial_dist: a matrix with dimension (arm_num, state_num), each row of which shows the initial  distribution of each arm(channel)
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
        - belief_vec: an (N,) array showing the current belief state of each arm(channel)
        - tran_0: an (N,) array showing the transition probablity p_{01} of each arm(channel)
        - tran_1: an (N,) array showing the transition probablity p_{11} of each arm(channel)
        - ini_belief: an (N,) array showing the initial belief state w_{0} of each arm(channel)
        - reward: an (N,) array showing the reward of good state for each arm(channel)
        - T: a scalar for the playing time
        - AR: an (T,) array showing the actual reward of each time
        """
        self.mabm = MABM(arm_num, state_num, initial_dist, transition_matrics)
        self.action = np.zeros(arm_num)
        self.belief_vec = initial_dist[:, 1].copy()
        self.N = arm_num
        self.K = active_num
        self.tran_0 = transition_matrics[:, 0, 1].copy().reshape(self.N)
        self.tran_1 = transition_matrics[:, 1, 1].copy().reshape(self.N)
        self.ini_belief = initial_dist[:, 1].copy()
        self.reward = reward_set[:, 1].copy()
        self.T = time
        self.action = np.zeros(self.N)
        self.AR = np.array([])
    
    def belief_vec_update(self):
        """
        Function:
        - Update the belief vector according to the action of each arm(channel).
        """
        self.belief_vec[self.action == 1] = self.mabm.tran_prob[self.action == 1, 1]
        idx0 = (self.action == 0)
        self.belief_vec[idx0] = self.belief_vec[idx0]*self.tran_1[idx0] + (1 - self.belief_vec[idx0])*self.tran_0[idx0]

    def k_step_belief(self, k, belief, tran_0, tran_1):
        """
        Function:
        - Compute the k-step belief state.

        Input:
        - k: a scalar indicating whose belief state the function computes
        - belief: a scalar or an array indicating current belief state
        - tran_0: a scalar or an array showing the transition probablity p_{01}
        - tran_1: a scalar or an array showing the transition probablity p_{11}
        """
        temp = tran_1 - tran_0
        return (tran_0 - (temp**k)*(tran_0 - (1 -temp)*belief))/(1 - temp)

    def fir_cross_t(self, cur, tar, tran_0, tran_1, ini_belief):
        """
        Function:
        - Compute the first crossing time, from current belief state to target belief state, for possive correlated arm.

        Input:
        - cur: a scalar or an array showing the current belief state
        - tar: a scalar or an array showing the target belief state
        - ini_belief: a scalar or an array showing the initial belief state w_{0}
        """
        L = np.zeros(np.size(cur))
        L[(cur <= tar)*(tar >= ini_belief)] = float('inf')
        temp_idx = (cur <= tar)*(tar < ini_belief)
        temp_val = 1 - tran_1[temp_idx] + tran_0[temp_idx]
        num = tran_0[temp_idx] - tar[temp_idx]*temp_val
        deno = tran_0[temp_idx] - cur[temp_idx]*temp_val
        L[temp_idx] = 1 + (np.log(num/deno)/np.log(1 - temp_val)).astype(int)
        #print(L)
        return L
    
    def whittle_idx_pos(self, w, beta, reward, tran_0, tran_1, ini_belief):
        """
        Function:
        - Compute the whittle index for possive correlated arm(channel)

        Input:
        - w: a scalar or an array showing the belief state for the whittle index
        - beta: a scalar indicating the discounted factor
        - reward: a scalar or an array showing the reward of good state
        """

        whittle_idx = np.zeros(np.size(w))

        idx1 = (w <= tran_0) + (w >= tran_1)
        idx2 = (ini_belief <= w)*(w < tran_1)
        idx3 = (tran_0 < w)*(w < ini_belief)

        L = self.fir_cross_t(tran_0[idx3], w[idx3], tran_0[idx3], tran_1[idx3], ini_belief[idx3])
        #print(L)
        T_L = self.k_step_belief(L, tran_0[idx3], tran_0[idx3], tran_1[idx3])
        beta_L = beta**(L + 1)
        temp1 = 1 - beta*tran_1[idx3]
        temp2 = temp1*(1 - beta_L) + (1 - beta)*beta_L*T_L
        C1 = temp1*(1 - beta_L/beta)/temp2
        C2 = T_L*(beta_L/beta)/temp2
        T_1 = self.k_step_belief(1, w[idx3], tran_0[idx3], tran_1[idx3])
        temp3 = w[idx3] - beta*T_1
        temp4 = beta*(temp1 - w[idx3] - temp3)

        whittle_idx[idx1] = w[idx1]*reward[idx1]
        whittle_idx[idx2] = (w[idx2]*reward[idx2])/(1 - beta*tran_1[idx2] + beta*w[idx2])
        whittle_idx[idx3] = reward[idx3]*(temp3 + C2*(1 - beta)*temp4)/(temp1 - C1*temp4)

        return whittle_idx

    def whittle_idx_neg(self, w, beta, reward, tran_0, tran_1, ini_belief):
        """
        Function:
        - Compute the whittle index for possive correlated arm(channel)
        """

        whittle_idx = np.zeros(np.size(w))
        
        T_p = self.k_step_belief(1, tran_1, tran_0, tran_1)
        idx1 = (w <= tran_1) + (w >= tran_0)
        idx2 = (T_p <= w)*(w < tran_0)
        idx3 = (ini_belief <= w)*(w < T_p)
        idx4 = (tran_1 < w)*(w < ini_belief)

        temp1_3 = 1 + (1 + beta)*beta*tran_0[idx3] - T_p[idx3]*(beta**2)
        temp1_4 = 1 + (1 + beta)*beta*tran_0[idx4] - T_p[idx4]*(beta**2)
        C3_3 = (1 - beta*(1 - tran_0[idx3]))/temp1_3
        C3_4 = (1 - beta*(1 - tran_0[idx4]))/temp1_4
        C4_3 = (beta*T_p[idx3]*(1 - beta) + tran_0[idx3]*(beta**2))/temp1_3
        C4_4 = (beta*T_p[idx4]*(1 - beta) + tran_0[idx4]*(beta**2))/temp1_4
        temp2 = beta*tran_0[idx3] + w[idx3]*(1 - beta)
        temp3_3 = 1 - beta*(1 - tran_0[idx3])
        temp3_4 = 1 - beta*(1 - tran_0[idx4])
        T_1 = self.k_step_belief(1, w[idx4], tran_0[idx4], tran_1[idx4])
        temp4 = beta*T_1 - beta*tran_0[idx4] - w[idx4]


        whittle_idx[idx1] = w[idx1]*reward[idx1]
        whittle_idx[idx2] = reward[idx2]*(beta*tran_0[idx2] + w[idx2]*(1 - beta))/(1 + beta*(tran_0[idx2] - w[idx2]))
        whittle_idx[idx3] = reward[idx3]*(1 - beta + beta*C4_3)*temp2/(temp3_3 - C3_3*beta*temp2)
        whittle_idx[idx4] = reward[idx4]*((beta - 1)*temp4 - C4_4*beta*temp4)/(temp3_4 + C3_4*beta*temp4)

        return whittle_idx

    

    
    def action_AR_update(self, beta):
        """
        Function:
        - Find the active arms(channels) of each time according to the whittle indices of them.
        - Compute actual reward according to the action of each time.
        """

        pos_arm = (self.tran_1 > self.tran_0)
        neg_arm = (self.tran_1 <= self.tran_0)
        m = np.zeros(self.N)
        m[pos_arm] = self.whittle_idx_pos(self.belief_vec[pos_arm], beta, self.reward[pos_arm], self.tran_0[pos_arm], self.tran_1[pos_arm], self.ini_belief[pos_arm])
        m[neg_arm] = self.whittle_idx_neg(self.belief_vec[neg_arm], beta, self.reward[neg_arm], self.tran_0[neg_arm], self.tran_1[neg_arm], self.ini_belief[neg_arm])

        self.action = np.zeros(self.N)
        self.action[m.argsort()[::-1][0:self.K]] = 1
        self.AR = np.append(self.AR, np.sum(self.reward[(self.action == 1)*(self.mabm.state== 1)]))
    
    def User_process(self, beta):
        """
        Function:
        - Simulate the game process of a user, who uses whittle index policy, with above functions.
        """
        
        for t in range(self.T):
            self.mabm.MABM_update()
            self.action_AR_update(beta)
            self.belief_vec_update()